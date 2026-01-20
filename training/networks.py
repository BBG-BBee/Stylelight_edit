# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma

import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# from einops import rearrange, reduce, repeat
import math
#----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    """
    입력 텐서를 2차 모멘트(표준편차와 유사)로 정규화합니다.
    """
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_conv2d(
    x,                          # 입력 텐서: [batch_size, in_channels, in_height, in_width].
    weight,                     # 가중치 텐서: [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # 변조(Modulation) 계수: [batch_size, in_channels].
    noise           = None,     # 출력 활성화에 더할 노이즈 텐서 (선택 사항).
    up              = 1,        # 정수 업샘플링 계수.
    down            = 1,        # 정수 다운샘플링 계수.
    padding         = 0,        # 업샘플링된 이미지에 대한 패딩.
    resample_filter = None,     # 리샘플링 시 적용할 저역 통과 필터. upfirdn2d.setup_filter()로 미리 준비해야 함.
    demodulate      = True,     # 가중치 복조(Demodulation) 적용 여부?
    flip_weight     = True,     # False = convolution, True = correlation (torch.nn.functional.conv2d와 일치).
    fused_modconv   = True,     # 변조, 컨볼루션, 복조를 하나의 융합된 연산으로 수행?
):
    """
    StyleGAN2의 핵심 연산인 Modulated Convolution을 수행합니다.
    스타일 벡터(styles)를 사용하여 컨볼루션 가중치를 변조(scaling)하고, 선택적으로 복조(normalizing)합니다.
    """
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # FP16 오버플로우 방지를 위한 입력 사전 정규화.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # 샘플별 가중치 및 복조 계수 계산.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # 컨볼루션 전후에 스케일링을 적용하여 실행.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # 그룹 컨볼루션을 사용하여 하나의 융합된 연산으로 실행.
    with misc.suppress_tracer_warnings(): # 이 값은 상수로 취급됨
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    """
    기본적인 완전 연결 레이어 (Fully Connected Layer)입니다.
    """
    def __init__(self,
        in_features,                # 입력 특징 수.
        out_features,               # 출력 특징 수.
        bias            = True,     # 활성화 함수 전에 편향(bias)을 더할지 여부.
        activation      = 'linear', # 활성화 함수: 'relu', 'lrelu' 등.
        lr_multiplier   = 1,        # 학습률 승수.
        bias_init       = 0,        # 편향의 초기값.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    """
    기본적인 2D 컨볼루션 레이어입니다. 업샘플링/다운샘플링을 지원합니다.
    """
    def __init__(self,
        in_channels,                    # 입력 채널 수.
        out_channels,                   # 출력 채널 수.
        kernel_size,                    # 컨볼루션 커널의 너비와 높이.
        bias            = True,         # 활성화 함수 전에 편향을 더할지 여부.
        activation      = 'linear',     # 활성화 함수: 'relu', 'lrelu' 등.
        up              = 1,            # 정수 업샘플링 계수.
        down            = 1,            # 정수 다운샘플링 계수.
        resample_filter = [1,3,3,1],    # 리샘플링 시 적용할 저역 통과 필터.
        conv_clamp      = None,         # 출력을 +-X로 클램프, None = 클램핑 비활성화.
        channels_last   = False,        # 입력이 memory_format=channels_last를 갖는지 여부.
        trainable       = True,         # 학습 중에 이 레이어의 가중치를 업데이트할지 여부.
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1) # 약간 더 빠름
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x



#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    """
    [매핑 네트워크 (Mapping Network)]: 잠재벡터 Z를 중간 잠재벡터 W로 매핑합니다.
    """
    def __init__(self,
        z_dim,                      # 입력 잠재(Z) 차원, 0 = 잠재 없음.
        c_dim,                      # 조건부 레이블(C) 차원, 0 = 레이블 없음.
        w_dim,                      # 중간 잠재(W) 차원.
        num_ws,                     # 출력할 중간 잠재 개수, None = 브로드캐스트 안 함.
        num_layers      = 8,        # 매핑 레이어 수.
        embed_features  = None,     # 레이블 임베딩 차원, None = w_dim과 동일.
        layer_features  = None,     # 매핑 레이어의 중간 특징 수, None = w_dim과 동일.
        activation      = 'lrelu',  # 활성화 함수: 'relu', 'lrelu' 등.
        lr_multiplier   = 0.01,     # 매핑 레이어의 학습률 승수.
        w_avg_beta      = 0.995,    # 학습 중 W의 이동 평균 추적을 위한 감쇠(Decay), None = 추적 안 함.
    ):
        super().__init__()
        
        self.z_dim = z_dim     ##
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # 임베딩, 정규화 및 입력 연결.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                # misc.assert_shape(z, [None, self.z_dim])
                # [잠재벡터 z (Latent Vector z)]: 정규 분포에서 샘플링된 512차원 입력 벡터
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # 메인 레이어.
        for idx in range(self.num_layers):
            # print('x shape and idx:', x.shape, idx)
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # W의 이동 평균 업데이트.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # 브로드캐스트.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Truncation 적용.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        # [중간 잠재벡터 w (Intermediate Latent Vector w)]: Mapping Network의 출력으로, 스타일 정보가 분리된 512차원 벡터
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    """
    Synthesis Layer: 스타일 벡터 W를 사용하여 컨볼루션 필터의 가중치를 변조(Modulation)하고, 이를 통해 특징 맵을 생성하는 단일 레이어입니다.
    핵심 연산인 modulated_conv2d를 호출하여 스타일을 이미지에 반영합니다.
    """
    def __init__(self,
        in_channels,                    # 입력 채널 수.
        out_channels,                   # 출력 채널 수.
        w_dim,                          # 중간 잠재(W) 차원.
        resolution,                     # 이 레이어의 해상도.
        kernel_size     = 3,            # 컨볼루션 커널 크기.
        up              = 1,            # 정수 업샘플링 계수.
        use_noise       = True,         # 노이즈 입력 활성화?
        activation      = 'lrelu',      # 활성화 함수: 'relu', 'lrelu' 등.
        resample_filter = [1,3,3,1],    # 리샘플링 시 적용할 저역 통과 필터.
        conv_clamp      = None,         # 컨볼루션 레이어 출력을 +-X로 클램프, None = 클램핑 비활성화.
        channels_last   = False,        # 가중치에 channels_last 포맷 사용?
    ):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, 2*resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.weight.shape[1], in_resolution, 2*in_resolution])
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, 2*self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        # [컨볼루션 레이어 (Convolution Layer)]: Modulated Conv2d를 사용하여 스타일(w)에 따라 필터 가중치를 조절하고 특징 맵을 생성
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):
    """
    특징 맵을 RGB 이미지로 변환하는 레이어입니다.
    """
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x


@persistence.persistent_class
class ToRGBLayerHDR(torch.nn.Module):
    """
    물리적 HDR 출력을 위한 ToRGB 레이어

    Softplus 활성화 함수를 사용하여 물리적 휘도(cd/m²)의
    비음수성(Non-negativity)과 무한한 동적 범위를 지원합니다.

    출력 범위: [0, ∞) - 물리적 휘도 스케일
    """
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1,
                 conv_clamp=None, channels_last=False,
                 activation='softplus',  # 'softplus', 'linear', 'relu'
                 softplus_beta=1.0):     # Softplus의 beta 파라미터
        super().__init__()
        self.conv_clamp = conv_clamp
        self.activation = activation
        self.softplus_beta = softplus_beta
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)

        # Bias 추가 (클램핑 없이)
        x = x + self.bias.to(x.dtype).reshape(1, -1, 1, 1)

        # 활성화 함수 적용
        if self.activation == 'softplus':
            # Softplus: y = (1/beta) * ln(1 + exp(beta * x))
            # 출력: [0, ∞) - 물리적 휘도에 적합
            x = torch.nn.functional.softplus(x, beta=self.softplus_beta)
        elif self.activation == 'relu':
            # ReLU: 음수 제거
            x = torch.nn.functional.relu(x)
        # 'linear': 활성화 없음 (기존 방식과 호환)

        return x
#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):
    """
    Synthesis Block: 특정 해상도를 담당하는 블록입니다. 두 개의 SynthesisLayer와 하나의 ToRGBLayer를 포함할 수 있습니다.
    """
    def __init__(self,
        in_channels,                        # 입력 채널 수, 0 = 첫 번째 블록.
        out_channels,                       # 출력 채널 수.
        w_dim,                              # 중간 잠재(W) 차원.
        resolution,                         # 이 블록의 해상도.
        img_channels,                       # 출력 색상 채널 수.
        is_last,                            # 마지막 블록인가?
        architecture        = 'skip',       # 아키텍처: 'orig', 'skip', 'resnet'.
        resample_filter     = [1,3,3,1],    # 리샘플링 시 적용할 저역 통과 필터.
        conv_clamp          = None,         # 컨볼루션 레이어 출력을 +-X로 클램프, None = 클램핑 비활성화.
        use_fp16            = False,        # 이 블록에 FP16 사용?
        fp16_channels_last  = False,        # FP16과 함께 channels-last 메모리 포맷 사용?
        hdr_mode            = False,        # HDR 물리 모드: Softplus 활성화 사용
        hdr_activation      = 'softplus',   # HDR 모드 활성화 함수: 'softplus', 'relu', 'linear'
        **layer_kwargs,                     # SynthesisLayer에 전달할 인자.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.hdr_mode = hdr_mode
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if resolution>=32:
            out_channels_=out_channels
        else:
            out_channels_=out_channels


        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, 2*resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels_, w_dim=w_dim, resolution=resolution, up=2,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1
        self.conv1 = SynthesisLayer(out_channels_, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            # HDR 모드: Softplus 활성화 사용 (물리적 휘도 출력)
            if hdr_mode and is_last:
                self.torgb = ToRGBLayerHDR(out_channels, img_channels, w_dim=w_dim,
                    conv_clamp=None, channels_last=self.channels_last,
                    activation=hdr_activation)
            else:
                self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                    conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, **layer_kwargs):
        # misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # 이 값은 상수로 취급됨
                fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        # 입력.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            # misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # 메인 레이어.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    """
    SynthesisNetwork: 4x4 해상도부터 최종 해상도까지 SynthesisBlock들을 층층이 쌓아올린 전체 합성 네트워크입니다.
    각 해상도 단계마다 SynthesisLayer를 통해 피쳐 맵을 점진적으로 키우고 디테일을 추가하여 최종 이미지를 생성합니다.
    """
    def __init__(self,
        w_dim,                      # 중간 잠재(W) 차원.
        img_resolution,             # 출력 이미지 해상도.
        img_channels,               # 색상 채널 수.
        channels_dict = {4: 512, 8: 512, 16: 512, 32: 512, 64: 256, 128: 128, 256: 64},  # original 
        channel_base    = 32768,    # 전체 채널 수 승수.
        channel_max     = 512,      # 어떤 레이어에서의 최대 채널 수.
        num_fp16_res    = 0,        # 가장 높은 N개의 해상도에 FP16 사용.
        **block_kwargs,             # SynthesisBlock에 전달할 인자.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    """
    Generator: Mapping Network와 Synthesis Network를 결합한 전체 생성자 모델입니다.
    입력된 잠재 벡터 z를 w로 변환하고, 이를 기반으로 최종적인 고해상도 이미지를 합성하는 전체 과정을 담당합니다.
    """
    def __init__(self,
        z_dim,                      # 입력 잠재(Z) 차원.
        c_dim,                      # 조건부 레이블(C) 차원.
        w_dim,                      # 중간 잠재(W) 차원.
        img_resolution,             # 출력 해상도.
        img_channels,               # 출력 색상 채널 수.
        mapping_kwargs      = {},   # MappingNetwork 인자.
        synthesis_kwargs    = {},   # SynthesisNetwork 인자.
        rank = 'cuda:0'
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        ws = self.mapping(z, None, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        b, num_ws, dim = ws.size()
        img_ = self.synthesis(ws, **synthesis_kwargs)
        return img_


#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    """
    Discriminator Block: 특정 해상도를 담당하는 판별자 블록입니다.
    """
    def __init__(self,
        in_channels,                        # 입력 채널 수, 0 = 첫 번째 블록.
        tmp_channels,                       # 중간 채널 수.
        out_channels,                       # 출력 채널 수.
        resolution,                         # 이 블록의 해상도.
        img_channels,                       # 입력 색상 채널 수.
        first_layer_idx,                    # 첫 번째 레이어의 인덱스.
        architecture        = 'resnet',     # 아키텍처: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # 활성화 함수: 'relu', 'lrelu' 등.
        resample_filter     = [1,3,3,1],    # 리샘플링 시 적용할 저역 통과 필터.
        conv_clamp          = None,         # 컨볼루션 레이어 출력을 +-X로 클램프, None = 클램핑 비활성화.
        use_fp16            = False,        # 이 블록에 FP16 사용?
        fp16_channels_last  = False,        # FP16과 함께 channels-last 메모리 포맷 사용?
        freeze_layers       = 0,            # Freeze-D: 고정할 레이어 수.
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation,
            trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
                trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # 입력.
        if x is not None:
            # print('x.shape:', x.shape)
            # print('self.resolution:', self.resolution)
            misc.assert_shape(x, [None, self.in_channels, self.resolution, 2*self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            # misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

        # 메인 레이어.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    """
    Minibatch Standard Deviation Layer: 미니배치 통계를 사용하여 모드 붕괴(Mode collapse)를 방지합니다.
    """
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor 결과는 상수로 등록됨
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F
        n = N//G
        # print: 8 1 tensor(8) 1 512 32
        # print('self.group_size,self.num_channels,G,F,c,N:',self.group_size, self.num_channels, G,F,c,N)
        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] 미니배치 N을 G 크기의 n개 그룹으로 나누고, 채널 C를 c 크기의 F개 그룹으로 나눕니다.
        y = y - y.mean(dim=0)               # [GnFcHW] 그룹에 대해 평균을 뺍니다.
        y = y.square().mean(dim=0)          # [nFcHW]  그룹에 대해 분산을 계산합니다.
        y = (y + 1e-8).sqrt()               # [nFcHW]  그룹에 대해 표준편차를 계산합니다.
        y = y.mean(dim=[2,3,4])             # [nF]     채널과 픽셀에 대해 평균을 냅니다.

        #y = y.reshape(-1, F, 1, 1)          # [nF11]   누락된 차원을 추가합니다.
        #y = y.repeat(G, 1, H, W)            # [NFHW]   그룹과 픽셀에 대해 복제합니다.

        y = y.reshape(n,1, F, 1, 1)
        y = y.expand(n,G,F, 1, 1).reshape(-1,F,1,1)
        y = y.repeat(1, 1, H, W)

        x = torch.cat([x, y], dim=1)        # [NCHW]   입력에 새로운 채널로 추가합니다.
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    """
    Discriminator Epilogue: 판별자의 마지막 부분으로, MinibatchStdLayer와 최종 분류 레이어를 포함합니다.
    """
    def __init__(self,
        in_channels,                    # 입력 채널 수.
        cmap_dim,                       # 매핑된 조건부 레이블 차원, 0 = 레이블 없음.
        resolution,                     # 이 블록의 해상도.
        img_channels,                   # 입력 색상 채널 수.
        architecture        = 'resnet', # 아키텍처: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # MinibatchStdLayer를 위한 그룹 크기, None = 전체 미니배치.
        mbstd_num_channels  = 1,        # MinibatchStdLayer를 위한 특징 수, 0 = 비활성화.
        activation          = 'lrelu',  # 활성화 함수: 'relu', 'lrelu' 등.
        conv_clamp          = None,     # 컨볼루션 레이어 출력을 +-X로 클램프, None = 클램핑 비활성화.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (2*resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, 2*self.resolution]) # [NCHW]
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, 2*self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # 메인 레이어.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # 조건부 처리 (Conditioning).
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class Discriminator(torch.nn.Module):
    """
    Discriminator: 입력된 이미지가 진짜인지 생성된 가짜인지 판별하는 전체 판별자 네트워크입니다.
    ResNet 구조를 기반으로 이미지를 점진적으로 다운샘플링하며 특징을 추출하고, 
    MinibatchStdLayer를 통해 배치의 다양성을 확인하여 모드 붕괴를 방지합니다.
    StyleLight에서는 LDR용(D)과 HDR용(D_) 두 개의 판별자가 사용됩니다.
    """
    def __init__(self,
        c_dim,                          # 조건부 레이블(C) 차원.
        img_resolution,                 # 입력 해상도.
        img_channels,                   # 입력 색상 채널 수.
        channels_dict= {256: 64, 128: 128, 64: 256, 32: 512, 16: 512, 8: 512, 4: 512},
        architecture        = 'resnet', # 아키텍처: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # 전체 채널 수 승수.
        channel_max         = 512,      # 어떤 레이어에서의 최대 채널 수.
        num_fp16_res        = 0,        # 가장 높은 N개의 해상도에 FP16 사용.
        conv_clamp          = None,     # 컨볼루션 레이어 출력을 +-X로 클램프, None = 클램핑 비활성화.
        cmap_dim            = None,     # 매핑된 조건부 레이블 차원, None = 기본값.
        block_kwargs        = {},       # DiscriminatorBlock에 전달할 인자.
        mapping_kwargs      = {},       # MappingNetwork에 전달할 인자.
        epilogue_kwargs     = {},       # DiscriminatorEpilogue에 전달할 인자.
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        # channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, **block_kwargs):
        x = None
        #######################################################
        batch_size, num_channels, height, width = img.shape
        # print('batch_size, num_channels, height, width:',batch_size, num_channels, height, width)
        random_index = np.random.randint(width)
        img = torch.cat((img[:,:,:,random_index:],img[:,:,:,:random_index]), dim=3)
        #######################################################

        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

#----------------------------------------------------------------------------
