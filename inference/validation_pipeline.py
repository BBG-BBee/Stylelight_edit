"""
검증 파이프라인 (Validation Pipeline)

NFoV 입력 → HDR 파노라마 생성 → 원형어안 HDR 출력

워크플로우:
1. NFoV 이미지 입력 (초점거리에 따라 vFOV 자동 계산)
2. Generator로 512x1024 Equirectangular HDR 생성
3. 전방 180° 영역 크롭 (512x512)
4. SwinIR로 초해상도 (512x512 → 1024x1024)
5. Angular Fisheye 변환 (-vta)
6. Radiance HDR 파일 저장 (evalglare 호환)

evalglare 실행은 사용자가 직접 수행합니다.

사용법:
    python inference/validation_pipeline.py --checkpoint stage2.pkl --input nfov.exr --output fisheye.hdr
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import pickle
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from skylibs.hdrio import imread, imsave
    from skylibs.envmap import EnvironmentMap, rotation_matrix
except ImportError:
    print("Warning: skylibs not found. Some functions may not work.")
    imread = None
    imsave = None
    rotation_matrix = None

from utils.hdr_utils import log_domain_resize


class ValidationPipeline:
    """
    HDR 파노라마 검증 파이프라인

    NFoV 입력으로부터 evalglare 호환 원형어안 HDR 파일을 생성합니다.

    Args:
        generator: 학습된 Generator 모델
        sr_model: 초해상도 모델 (None이면 Lanczos 사용)
        device: 디바이스
    """

    def __init__(self,
                 generator: torch.nn.Module,
                 sr_model: Optional[torch.nn.Module] = None,
                 device: str = 'cuda'):
        self.device = device
        self.G = generator.to(device).eval()
        self.sr_model = sr_model

        if sr_model is not None:
            self.sr_model = sr_model.to(device).eval()

        # 동결
        for param in self.G.parameters():
            param.requires_grad = False

    @classmethod
    def from_checkpoint(cls,
                        checkpoint_path: str,
                        sr_model_path: Optional[str] = None,
                        device: str = 'cuda') -> 'ValidationPipeline':
        """
        체크포인트에서 파이프라인 생성

        Args:
            checkpoint_path: Generator 체크포인트 경로
            sr_model_path: SwinIR 모델 경로 (선택)
            device: 디바이스

        Returns:
            pipeline: ValidationPipeline 인스턴스
        """
        print(f'체크포인트 로드 중: {checkpoint_path}')

        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)

        # Generator 로드 (G_ema 우선)
        if 'G_ema' in data:
            generator = data['G_ema']
        elif 'G' in data:
            generator = data['G']
        else:
            raise KeyError("Checkpoint must contain 'G_ema' or 'G'")

        # SwinIR 로드 (선택)
        sr_model = None
        if sr_model_path is not None:
            try:
                from inference.super_resolution import SwinIRUpscaler
                sr_model = SwinIRUpscaler(
                    scale=2,
                    model_path=sr_model_path,
                    device=device
                )
            except Exception as e:
                print(f"Warning: Failed to load SwinIR: {e}")

        return cls(generator, sr_model, device)

    def generate_panorama(self,
                          z: Optional[torch.Tensor] = None,
                          truncation_psi: float = 0.7) -> torch.Tensor:
        """
        랜덤 또는 지정된 latent로 파노라마 생성

        Args:
            z: latent 벡터 (None이면 랜덤 생성)
            truncation_psi: truncation 파라미터

        Returns:
            panorama: 512x1024 Equirectangular HDR (B, 3, 512, 1024)
        """
        if z is None:
            z = torch.randn(1, 512, device=self.device)

        with torch.no_grad():
            # W space로 매핑
            w = self.G.mapping(z, None, truncation_psi=truncation_psi)
            # 이미지 합성
            panorama = self.G.synthesis(w)

        return panorama

    def crop_forward_hemisphere(self,
                                equirect: torch.Tensor) -> torch.Tensor:
        """
        Equirectangular에서 전방 180° 영역 추출

        Args:
            equirect: Equirectangular 이미지 (B, 3, H, W)
                      W = 360°이므로 전방 180°는 중앙 절반

        Returns:
            cropped: 전방 반구 이미지 (B, 3, H, H)
        """
        B, C, H, W = equirect.shape

        # 전방 180° = 중앙 영역 (W/4 ~ W*3/4)
        # 정확히 H x H 크기로 추출
        start = W // 4
        end = start + H  # H == W // 2

        cropped = equirect[..., start:end]

        return cropped

    def upscale(self, image: torch.Tensor, target_size: int = 1024) -> torch.Tensor:
        """
        이미지 업스케일

        Args:
            image: 입력 이미지 (B, 3, H, W)
            target_size: 목표 크기

        Returns:
            upscaled: 업스케일된 이미지 (B, 3, target_size, target_size)
        """
        if self.sr_model is not None:
            with torch.no_grad():
                upscaled = self.sr_model(image)
        else:
            # Lanczos 폴백 (bicubic으로 근사)
            upscaled = F.interpolate(
                image,
                size=(target_size, target_size),
                mode='bicubic',
                align_corners=False,
                antialias=True
            )

        return upscaled

    def equirect_to_angular_fisheye(self,
                                    equirect: torch.Tensor,
                                    fov: float = 180.0) -> torch.Tensor:
        """
        Equirectangular를 Angular Fisheye (-vta)로 변환

        Angular Fisheye 프로젝션:
        - r = θ (각도에 비례하는 반경)
        - evalglare의 -vta 옵션과 호환

        Args:
            equirect: Equirectangular 이미지 (B, 3, H, W), H==W 가정
            fov: 시야각 (기본값: 180°)

        Returns:
            fisheye: Angular Fisheye 이미지 (B, 3, H, W)
        """
        B, C, H, W = equirect.shape
        assert H == W, "Input must be square for fisheye conversion"

        device = equirect.device
        dtype = equirect.dtype

        # 출력 좌표 그리드 생성 (-1 ~ 1)
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device, dtype=dtype),
            torch.linspace(-1, 1, W, device=device, dtype=dtype),
            indexing='ij'
        )

        # 극좌표 변환
        r = torch.sqrt(x**2 + y**2)
        phi = torch.atan2(y, x)  # 방위각 (-π ~ π)

        # Angular Fisheye: r = θ / (fov/2)
        # θ = r * (fov/2)
        fov_rad = fov * np.pi / 180.0
        theta = r * (fov_rad / 2)  # 천정각 (0 ~ π/2)

        # 유효 영역 마스크 (r <= 1)
        valid_mask = r <= 1.0

        # Equirectangular 좌표로 변환
        # Equirect: u = phi / π, v = (π/2 - theta) / (π/2)
        # 전방 반구만 사용하므로 theta는 0 ~ π/2

        # 정규화된 좌표 (-1 ~ 1)
        u = phi / np.pi  # -1 ~ 1
        v = 1.0 - (2.0 * theta / np.pi)  # 1 ~ -1 (위에서 아래로)

        # 샘플링 그리드
        grid = torch.stack([u, v], dim=-1)  # (H, W, 2)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

        # grid_sample로 리샘플링
        fisheye = F.grid_sample(
            equirect,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )

        # 유효 영역 외부는 0으로 설정
        valid_mask = valid_mask.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
        fisheye = fisheye * valid_mask.float()

        return fisheye

    def save_radiance_hdr(self,
                          image: torch.Tensor,
                          output_path: str,
                          view_params: Optional[Dict] = None) -> str:
        """
        Radiance HDR 형식으로 저장 (evalglare 호환)

        VIEW 헤더에 Angular Fisheye 파라미터를 삽입합니다.

        Args:
            image: HDR 이미지 (B, 3, H, W) 또는 (3, H, W)
            output_path: 출력 파일 경로
            view_params: VIEW 헤더 파라미터 (기본값: -vta -vv 180 -vh 180)

        Returns:
            output_path: 저장된 파일 경로
        """
        # 배치 차원 제거
        if image.dim() == 4:
            image = image[0]

        # (C, H, W) -> (H, W, C) numpy 변환
        image_np = image.cpu().numpy().transpose(1, 2, 0)

        # 음수 방지
        image_np = np.maximum(image_np, 0.0)

        # FP32로 저장
        image_np = image_np.astype(np.float32)

        # skylibs로 저장
        if imsave is not None:
            imsave(output_path, image_np)
        else:
            # 폴백: OpenCV 사용
            import cv2
            # OpenCV는 BGR 순서
            cv2.imwrite(output_path, image_np[:, :, ::-1])

        # VIEW 헤더 삽입 (Radiance .hdr 형식인 경우)
        if output_path.endswith('.hdr'):
            self._inject_view_header(output_path, view_params)

        print(f'HDR 저장 완료: {output_path}')
        return output_path

    def _inject_view_header(self,
                            hdr_path: str,
                            view_params: Optional[Dict] = None):
        """
        Radiance HDR 파일에 VIEW 헤더 삽입

        evalglare가 인식할 수 있는 형식으로 뷰 정보를 추가합니다.

        Args:
            hdr_path: HDR 파일 경로
            view_params: VIEW 파라미터 딕셔너리
        """
        if view_params is None:
            view_params = {
                'type': 'a',  # Angular fisheye
                'vv': 180,    # Vertical FOV
                'vh': 180,    # Horizontal FOV
            }

        # VIEW 문자열 생성
        view_str = f"VIEW= -vt{view_params.get('type', 'a')}"
        view_str += f" -vv {view_params.get('vv', 180)}"
        view_str += f" -vh {view_params.get('vh', 180)}"

        try:
            # 기존 파일 읽기
            with open(hdr_path, 'rb') as f:
                content = f.read()

            # 헤더 끝 찾기 (빈 줄)
            header_end = content.find(b'\n\n')
            if header_end == -1:
                header_end = content.find(b'\n\x0a')

            if header_end != -1:
                # VIEW 헤더 삽입
                header = content[:header_end]
                body = content[header_end:]

                # VIEW가 이미 있는지 확인
                if b'VIEW=' not in header:
                    new_header = header + f'\n{view_str}'.encode('ascii')
                    new_content = new_header + body

                    with open(hdr_path, 'wb') as f:
                        f.write(new_content)

        except Exception as e:
            print(f"Warning: Failed to inject VIEW header: {e}")

    def preprocess_lfov_input(self,
                              image_path: str,
                              vfov: float,
                              azimuth: float = 0.0,
                              elevation: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        NFoV HDR 이미지를 ERP 마스크드 입력으로 변환

        NFoV HDR 이미지를 512×1024 Equirectangular에
        embed하여 Generator 입력(GAN Inversion 등)으로 사용할 수 있는 형태로 변환한다.

        Args:
            image_path: NFoV HDR 이미지 경로 (.exr, .hdr)
            vfov: 수직 화각 (도)
            azimuth: 카메라 방위각 (rad)
            elevation: 카메라 고도각 (rad)

        Returns:
            masked_pano: (1, 3, 512, 1024) 마스크된 ERP 텐서
            mask: (1, 1, 512, 1024) 유효 영역 바이너리 마스크
        """
        import cv2

        # 1. HDR 이미지 로드
        image = imread(image_path).astype(np.float32)
        image = np.maximum(image, 0.0)  # 음수 방지

        # 2. 256×192로 다운스케일 (4:3) — log-domain 리사이즈로 피크 휘도 보존
        image_resized = log_domain_resize(image, 192, 256)

        # 3. 빈 ERP 생성 + 카메라 방향 설정
        env = EnvironmentMap(512, 'latlong')
        dcm = rotation_matrix(azimuth=azimuth, elevation=elevation, roll=0)

        # 4. crop을 ERP에 embed → (512, 1024, 3)
        #    Fov2MaskedPano(mode="normal"): 빈 ERP 좌표에 crop 픽셀 매핑
        masked_pano = env.Fov2MaskedPano(
            image_resized, vfov=vfov, rotation_matrix=dcm,
            ar=4. / 3., resolution=(256, 192),
            projection="perspective", mode="normal"
        )

        # 5. binary mask 획득 — mode="mask"는 (512, 1024) mask만 반환
        mask = env.Fov2MaskedPano(
            image_resized, vfov=vfov, rotation_matrix=dcm,
            ar=4. / 3., resolution=(256, 192),
            projection="perspective", mode="mask"
        )

        # 6. 텐서 변환 (HWC → CHW → batch)
        masked_pano_t = torch.from_numpy(
            masked_pano.transpose(2, 0, 1).copy()
        ).unsqueeze(0).float().to(self.device)
        mask_t = torch.from_numpy(
            mask[np.newaxis, np.newaxis].copy()
        ).float().to(self.device)

        return masked_pano_t, mask_t

    def run(self,
            z: Optional[torch.Tensor] = None,
            input_path: Optional[str] = None,
            vfov: float = 55.1,
            azimuth: float = 0.0,
            elevation: float = 0.0,
            output_path: str = 'output_fisheye.hdr',
            truncation_psi: float = 0.7,
            save_intermediate: bool = False) -> Dict:
        """
        전체 파이프라인 실행

        Args:
            z: latent 벡터 (None이면 랜덤)
            input_path: NFoV HDR 입력 이미지 경로 (지정 시 LFOV 전처리 수행)
            vfov: 수직 화각 (도)
            azimuth: 카메라 방위각 (rad)
            elevation: 카메라 고도각 (rad)
            output_path: 출력 파일 경로
            truncation_psi: truncation 파라미터
            save_intermediate: 중간 결과 저장 여부

        Returns:
            results: {
                'fisheye': 최종 fisheye 이미지 텐서,
                'panorama': Equirectangular 파노라마,
                'cropped': 크롭된 전방 반구,
                'upscaled': 업스케일된 이미지,
                'output_path': 저장된 파일 경로,
                'masked_pano': (입력 지정 시) 마스크된 ERP,
                'mask': (입력 지정 시) 유효 영역 마스크
            }
        """
        results = {}

        # Step 0: LFOV 입력 → GAN Inversion → 파노라마 생성
        if input_path is not None:
            print('Step 0: NFoV HDR 입력 전처리 중...')
            masked_pano, mask = self.preprocess_lfov_input(
                input_path, vfov=vfov, azimuth=azimuth, elevation=elevation
            )
            results['masked_pano'] = masked_pano
            results['mask'] = mask

            # GAN Inversion: masked_pano에서 bbox 영역을 보존하는 최적 w 탐색
            print('Step 0.5: GAN Inversion (초점 마스킹 HDR 역전환) 중...')
            from training.projectors import w_projector

            # mask에서 bbox 추출
            mask_np = mask.squeeze().cpu().numpy()  # (512, 1024)
            mask_rows = np.any(mask_np > 0, axis=1)
            mask_cols = np.any(mask_np > 0, axis=0)
            row_min = int(np.argmax(mask_rows))
            row_max = int(len(mask_rows) - np.argmax(mask_rows[::-1]))
            col_min = int(np.argmax(mask_cols))
            col_max = int(len(mask_cols) - np.argmax(mask_cols[::-1]))
            bbox = [row_min, row_max, col_min, col_max]

            # HDR 입력 텐서 (cd/m² 스케일)
            target_hdr = masked_pano.squeeze(0)  # (3, 512, 1024)

            import copy
            G_copy = copy.deepcopy(self.G).eval().to(self.device)
            w_opt = w_projector.project(
                G_copy, bbox, target_hdr,
                device=self.device,
                w_avg_samples=600,
                num_steps=450,
                w_name='lfov_inversion',
            )
            del G_copy

            # 최적 w로 파노라마 생성
            print('Step 1: 파노라마 생성 중 (GAN Inversion 결과)...')
            with torch.no_grad():
                panorama = self.G.synthesis(w_opt.to(self.device), noise_mode='const', force_fp32=True)
                panorama = torch.clamp(panorama, min=0.0)  # Softplus HDR
            results['panorama'] = panorama

        else:
            # 입력 없으면 랜덤 생성
            print('Step 1: 파노라마 생성 중...')
            panorama = self.generate_panorama(z, truncation_psi)
            results['panorama'] = panorama

        # Step 2: 전방 180° 크롭
        print('Step 2: 전방 반구 크롭 중...')
        cropped = self.crop_forward_hemisphere(panorama)
        results['cropped'] = cropped

        # Step 3: 초해상도
        print('Step 3: 초해상도 처리 중...')
        upscaled = self.upscale(cropped, target_size=1024)
        results['upscaled'] = upscaled

        # Step 4: Angular Fisheye 변환
        print('Step 4: Angular Fisheye 변환 중...')
        fisheye = self.equirect_to_angular_fisheye(upscaled)
        results['fisheye'] = fisheye

        # Step 5: HDR 저장
        print('Step 5: HDR 파일 저장 중...')
        saved_path = self.save_radiance_hdr(fisheye, output_path)
        results['output_path'] = saved_path

        # 중간 결과 저장 (선택)
        if save_intermediate:
            base_path = Path(output_path).stem
            output_dir = Path(output_path).parent

            # 파노라마
            self.save_radiance_hdr(
                panorama,
                str(output_dir / f'{base_path}_panorama.hdr')
            )
            # 크롭
            self.save_radiance_hdr(
                cropped,
                str(output_dir / f'{base_path}_cropped.hdr')
            )
            # 업스케일
            self.save_radiance_hdr(
                upscaled,
                str(output_dir / f'{base_path}_upscaled.hdr')
            )

        print('\n파이프라인 완료!')
        print(f'출력 파일: {saved_path}')
        print('\n다음 단계:')
        print(f'  evalglare -vta {saved_path}')

        return results


def main():
    parser = argparse.ArgumentParser(
        description='HDR 파노라마 → 원형어안 HDR 변환 파이프라인'
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Generator 체크포인트 경로')
    parser.add_argument('--output', type=str, default='output_fisheye.hdr',
                        help='출력 HDR 파일 경로')
    parser.add_argument('--sr_model', type=str, default=None,
                        help='SwinIR 모델 경로 (선택)')
    parser.add_argument('--truncation', type=float, default=0.7,
                        help='Truncation psi (기본값: 0.7)')
    parser.add_argument('--seed', type=int, default=None,
                        help='랜덤 시드')
    parser.add_argument('--save_intermediate', action='store_true',
                        help='중간 결과 저장')
    parser.add_argument('--device', type=str, default='cuda',
                        help='디바이스 (기본값: cuda)')

    # LFOV 입력 전처리 옵션
    parser.add_argument('--input', type=str, default=None,
                        help='NFoV HDR 입력 이미지 경로 (.exr, .hdr)')
    parser.add_argument('--focal_mm', type=float, default=None,
                        help='렌즈 초점거리 (mm, 풀프레임 환산). 미지정 시 대화형 입력')
    parser.add_argument('--vfov', type=float, default=None,
                        help='수직 화각 (도). 직접 지정 시 --focal_mm보다 우선')
    parser.add_argument('--az', type=float, default=0.0,
                        help='카메라 방위각 azimuth (rad)')
    parser.add_argument('--el', type=float, default=0.0,
                        help='카메라 고도각 elevation (rad)')

    args = parser.parse_args()

    # vFOV 결정: --vfov 직접 지정 > --focal_mm > 대화형 입력
    from utils.hdr_utils import focal_to_vfov
    if args.vfov is not None:
        vfov = args.vfov
    elif args.focal_mm is not None:
        vfov = focal_to_vfov(args.focal_mm)
        print(f'[FOV] {args.focal_mm}mm → vFOV={vfov:.1f}°')
    else:
        focal_input = input('렌즈 초점거리 입력 (mm, 풀프레임 환산, 기본값 23): ').strip()
        focal_mm = float(focal_input) if focal_input else 23.0
        vfov = focal_to_vfov(focal_mm)
        print(f'[FOV] {focal_mm}mm → vFOV={vfov:.1f}°')

    # 시드 설정
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # 파이프라인 생성
    pipeline = ValidationPipeline.from_checkpoint(
        args.checkpoint,
        sr_model_path=args.sr_model,
        device=args.device
    )

    # 실행
    results = pipeline.run(
        input_path=args.input,
        vfov=vfov,
        azimuth=args.az,
        elevation=args.el,
        output_path=args.output,
        truncation_psi=args.truncation,
        save_intermediate=args.save_intermediate
    )


if __name__ == '__main__':
    main()
