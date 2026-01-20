# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
from PIL import Image
import cv2
try:
    import pyspng
except ImportError:
    pyspng = None

from skylibs.envmap import EnvironmentMap
# from . import tonemapping #import TonemapHDR
from training.tonemapping import TonemapHDR
#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        # assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        # assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        self.tonemap = TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
        # self.tonemap = TonemapHDR(gamma=2.4, percentile=99, max_mapping=0.99)

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if (self._file_ext(fname) in PIL.Image.EXTENSION) or self._file_ext(fname) in ['.exr'])
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        # if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
        if resolution is not None and (raw_shape[2] != resolution and raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
                print('oh, no! Be carefully!')
            else: 
                e = EnvironmentMap(os.path.join(self._path, fname), 'latlong')
                image_hdr = e.data
                use_new_tonemapping = True
                if use_new_tonemapping:
                    img_ldr_, alpha, image_hdr_ = self.tonemap(image_hdr)
                    # both are processed, suppose this is the output of G.synthsis
                    img_ldr=img_ldr_*2-1
                    image_hdr = image_hdr_*2-1
                    image_hdr = np.clip(image_hdr, -1, 1e8)    
                    image = np.concatenate((img_ldr, image_hdr), axis=2)
                else:
                    gamma=2.4
                    image = np.clip(image_hdr,1e-10,1e8)

                    is_single_crop = False
                    if is_single_crop:
                        image = torch.power(image, 1 / gamma)*5.0#*255c  ######
                        image = torch.clip(image,0,1)
                    else:
                        level = [0.01,0.02,0.04]
                        aa = torch.clip(torch.pow(image/level[0], 1/gamma), 0, 1)
                        bb = torch.clip(torch.pow(image/level[1], 1/gamma), 0, 1)
                        cc = torch.clip(torch.pow(image/level[2], 1/gamma), 0, 1)
                        image = (aa+bb+cc)/3.0
                    
                    image=image*2-1
                    image = np.concatenate((image, image_hdr), axis=2)
                

        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------


class HDRPhysicalDataset(Dataset):
    """
    물리적 휘도(cd/m²)를 보존하는 FP32 데이터 로더

    DTAM 학습을 위해 톤매핑 없이 선형 HDR 데이터를 직접 로드합니다.
    Stage 1: 구조 학습 (S2R-HDR)
    Stage 2: 물리 보정 (Laval Photometric)
    """
    def __init__(self,
        path,                   # 데이터셋 경로 (디렉토리 또는 zip)
        resolution      = None, # 목표 해상도
        linear_hdr      = True, # True: 톤매핑 없이 선형 HDR 로드
        dataset_type    = 'auto', # 'auto', 's2r_hdr', 'laval'
        target_height   = 512,  # 목표 높이 (512x1024 Equirectangular)
        target_width    = 1024, # 목표 너비
        **super_kwargs,
    ):
        self._path = path
        self._zipfile = None
        self.linear_hdr = linear_hdr
        self.dataset_type = dataset_type
        self.target_height = target_height
        self.target_width = target_width

        # 톤매핑은 LDR 생성 시에만 사용 (비교용)
        self.tonemap = TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path)
                               for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        # HDR 파일만 필터링 (.exr, .hdr)
        self._image_fnames = sorted(
            fname for fname in self._all_fnames
            if self._file_ext(fname) in ['.exr', '.hdr']
        )
        if len(self._image_fnames) == 0:
            raise IOError('No HDR image files (.exr, .hdr) found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)

        if resolution is not None and (raw_shape[2] != resolution and raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')

        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        """
        물리적 휘도를 보존하는 HDR 이미지 로드

        Returns:
            image: (C, H, W) 형태의 FP32 텐서
                - linear_hdr=True: 3채널 선형 HDR (cd/m² 스케일)
                - linear_hdr=False: 6채널 (LDR 3ch + HDR 3ch)
        """
        fname = self._image_fnames[raw_idx]
        file_path = os.path.join(self._path, fname)

        # EnvironmentMap으로 HDR 로드 (선형 휘도 유지)
        e = EnvironmentMap(file_path, 'latlong')
        image_hdr = e.data.astype(np.float32)  # FP32 강제

        # 해상도 조정 (Lanczos 리샘플링)
        if image_hdr.shape[0] != self.target_height or image_hdr.shape[1] != self.target_width:
            image_hdr = self._resize_hdr(image_hdr, self.target_height, self.target_width)

        if self.linear_hdr:
            # 선형 HDR만 반환 (톤매핑 없음)
            # 음수 값 방지 (물리적 휘도는 0 이상)
            image_hdr = np.maximum(image_hdr, 0.0)
            image = image_hdr
        else:
            # LDR + HDR 6채널 출력 (기존 방식과 호환)
            img_ldr_, alpha, image_hdr_ = self.tonemap(image_hdr)
            img_ldr = img_ldr_ * 2 - 1  # [-1, 1]
            image_hdr_normalized = image_hdr_ * 2 - 1
            image_hdr_normalized = np.clip(image_hdr_normalized, -1, 1e8)
            image = np.concatenate((img_ldr, image_hdr_normalized), axis=2)

        # HWC => CHW
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        image = image.transpose(2, 0, 1)

        return image.astype(np.float32)

    def _resize_hdr(self, image, target_h, target_w):
        """
        HDR 이미지를 Lanczos 리샘플링으로 리사이즈

        주의: 휘도 값 보존을 위해 선형 공간에서 리사이즈
        """
        # OpenCV는 채널별로 리사이즈
        resized = cv2.resize(
            image,
            (target_w, target_h),
            interpolation=cv2.INTER_LANCZOS4
        )
        return resized

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def get_luminance(self, image):
        """
        RGB HDR 이미지에서 휘도(Y) 채널 추출

        ITU-R BT.709 표준 사용:
        Y = 0.2126 * R + 0.7152 * G + 0.0722 * B

        Args:
            image: (3, H, W) 또는 (B, 3, H, W) 형태의 HDR 이미지

        Returns:
            luminance: (1, H, W) 또는 (B, 1, H, W) 형태의 휘도 맵
        """
        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                return 0.2126 * image[0:1] + 0.7152 * image[1:2] + 0.0722 * image[2:3]
            else:
                return 0.2126 * image[:, 0:1] + 0.7152 * image[:, 1:2] + 0.0722 * image[:, 2:3]
        else:  # torch.Tensor
            if image.ndim == 3:
                return 0.2126 * image[0:1] + 0.7152 * image[1:2] + 0.0722 * image[2:3]
            else:
                return 0.2126 * image[:, 0:1] + 0.7152 * image[:, 1:2] + 0.0722 * image[:, 2:3]


#----------------------------------------------------------------------------


