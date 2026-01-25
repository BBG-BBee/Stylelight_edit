"""
데이터셋 전처리 파이프라인

S2R-HDR 및 Laval Photometric 데이터셋을
물리적 정합성 기반 HDR 파노라마 학습을 위해 전처리합니다.

사용법:
    python prepare_datasets.py --dataset s2r_hdr --input_dir /path/to/s2r --output_dir /path/to/output
    python prepare_datasets.py --dataset laval --input_dir /path/to/laval --output_dir /path/to/output
"""

import os
import sys
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from skylibs.envmap import EnvironmentMap
    from skylibs.hdrio import imread, imsave
except ImportError:
    print("Warning: skylibs not found. Some functions may not work.")


def prepare_s2r_hdr(input_dir: str, output_dir: str,
                    target_resolution: tuple = (512, 1024),
                    validate: bool = True):
    """
    S2R-HDR 데이터셋을 512x1024 Equirectangular로 변환

    S2R-HDR은 Unreal Engine 5로 생성된 합성 HDR 파노라마 데이터셋입니다.
    Stage 1 학습(구조 학습)에 사용됩니다.

    Args:
        input_dir: S2R-HDR 원본 데이터 경로
        output_dir: 변환된 데이터 저장 경로
        target_resolution: (height, width) 형태의 목표 해상도
        validate: 변환 후 유효성 검사 수행 여부

    Returns:
        processed_count: 처리된 이미지 수
    """
    target_h, target_w = target_resolution
    os.makedirs(output_dir, exist_ok=True)

    # 지원 확장자
    hdr_extensions = {'.exr', '.hdr'}

    # 입력 파일 목록
    input_files = []
    for ext in hdr_extensions:
        input_files.extend(Path(input_dir).rglob(f'*{ext}'))

    if len(input_files) == 0:
        print(f"Error: No HDR files found in {input_dir}")
        return 0

    print(f"Found {len(input_files)} HDR files in {input_dir}")
    print(f"Target resolution: {target_h}x{target_w}")

    # 통계 카운터
    skipped_count = 0           # 전체 skip
    resize_only_count = 0       # 리사이즈만
    bit_convert_only_count = 0  # 32비트 변환만
    format_only_count = 0       # 포맷 변환만 (.hdr → .exr)
    full_process_count = 0      # 전체 처리 (리사이즈 + 32비트)
    failed_files = []

    for file_path in tqdm(input_files, desc="Processing S2R-HDR"):
        try:
            # 출력 경로 먼저 계산
            rel_path = file_path.relative_to(input_dir)
            output_path = Path(output_dir) / rel_path.with_suffix('.exr')

            # HDR 이미지 로드 (선형 휘도 유지)
            image_hdr = imread(str(file_path))

            if image_hdr is None:
                failed_files.append(str(file_path))
                continue

            # 조건 체크
            need_resize = (image_hdr.shape[0] != target_h or image_hdr.shape[1] != target_w)
            need_bit_convert = (image_hdr.dtype != np.float32)

            # 둘 다 OK면 전체 skip 또는 포맷 변환만
            if not need_resize and not need_bit_convert:
                if file_path.suffix.lower() == '.exr' and output_path.exists():
                    skipped_count += 1
                    continue
                else:
                    # 포맷 변환만 (.hdr → .exr)
                    format_only_count += 1
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    imsave(str(output_path), image_hdr)
                    continue

            # 통계 카운터 증가
            if need_resize and not need_bit_convert:
                resize_only_count += 1
            elif not need_resize and need_bit_convert:
                bit_convert_only_count += 1
            else:
                full_process_count += 1

            # 32비트 변환 필요시만
            if need_bit_convert:
                image_hdr = image_hdr.astype(np.float32)

            # 음수 값 제거 (물리적 휘도는 0 이상)
            image_hdr = np.maximum(image_hdr, 0.0)

            # 리사이즈 필요시만
            if need_resize:
                image_hdr = cv2.resize(
                    image_hdr,
                    (target_w, target_h),
                    interpolation=cv2.INTER_LANCZOS4
                )

            # 출력 경로 생성
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # EXR 형식으로 저장 (FP32)
            imsave(str(output_path), image_hdr)

            # 유효성 검사
            if validate:
                loaded = imread(str(output_path))
                if loaded is None or loaded.shape != (target_h, target_w, 3):
                    failed_files.append(str(file_path))
                    continue

        except Exception as e:
            print(f"\nError processing {file_path}: {e}")
            failed_files.append(str(file_path))

    # 전처리 결과 출력
    processed_count = format_only_count + resize_only_count + bit_convert_only_count + full_process_count
    print(f"\n=== 전처리 결과 ===")
    print(f"전체 파일: {len(input_files)}개")
    print(f"")
    print(f"건너뜀: {skipped_count}개 (이미 {target_h}x{target_w} 32비트 EXR)")
    print(f"포맷 변환만: {format_only_count}개 (.hdr → .exr)")
    print(f"리사이즈만: {resize_only_count}개")
    print(f"32비트 변환만: {bit_convert_only_count}개")
    print(f"전체 처리: {full_process_count}개 (리사이즈 + 32비트)")
    if failed_files:
        print(f"실패: {len(failed_files)}개")
        for f in failed_files[:5]:
            print(f"  - {f}")
        if len(failed_files) > 5:
            print(f"  ... 외 {len(failed_files) - 5}개")

    return processed_count


def prepare_laval_photometric(input_dir: str, output_dir: str,
                               target_resolution: tuple = (512, 1024),
                               apply_exposure: bool = True):
    """
    Laval Photometric Indoor HDR 데이터셋을 절대 휘도(cd/m²)로 변환

    Laval 데이터셋은 색차계로 보정된 실측 HDR 데이터입니다.
    Stage 2 학습(물리 보정)에 사용됩니다.

    Args:
        input_dir: Laval 원본 데이터 경로
        output_dir: 변환된 데이터 저장 경로
        target_resolution: (height, width) 형태의 목표 해상도
        apply_exposure: HDR 헤더의 노출값을 적용할지 여부

    Returns:
        processed_count: 처리된 이미지 수
    """
    target_h, target_w = target_resolution
    os.makedirs(output_dir, exist_ok=True)

    # Laval 데이터셋은 주로 .hdr 형식
    hdr_extensions = {'.hdr', '.exr'}

    input_files = []
    for ext in hdr_extensions:
        input_files.extend(Path(input_dir).rglob(f'*{ext}'))

    if len(input_files) == 0:
        print(f"Error: No HDR files found in {input_dir}")
        return 0

    print(f"Found {len(input_files)} HDR files in {input_dir}")
    print(f"Target resolution: {target_h}x{target_w}")
    print(f"Apply exposure correction: {apply_exposure}")

    # 통계 카운터
    skipped_count = 0           # 전체 skip
    resize_only_count = 0       # 리사이즈만
    bit_convert_only_count = 0  # 32비트 변환만
    format_only_count = 0       # 포맷 변환만 (.hdr → .exr)
    full_process_count = 0      # 전체 처리 (리사이즈 + 32비트)
    failed_files = []

    # 휘도 통계 수집
    luminance_stats = []

    for file_path in tqdm(input_files, desc="Processing Laval Photometric"):
        try:
            # 출력 경로 먼저 계산
            rel_path = file_path.relative_to(input_dir)
            output_path = Path(output_dir) / rel_path.with_suffix('.exr')

            # HDR 이미지 로드
            image_hdr = imread(str(file_path))

            if image_hdr is None:
                failed_files.append(str(file_path))
                continue

            # 조건 체크
            need_resize = (image_hdr.shape[0] != target_h or image_hdr.shape[1] != target_w)
            need_bit_convert = (image_hdr.dtype != np.float32)

            # 둘 다 OK면 전체 skip 또는 포맷 변환만
            if not need_resize and not need_bit_convert:
                if file_path.suffix.lower() == '.exr' and output_path.exists():
                    skipped_count += 1
                    continue
                else:
                    # 포맷 변환만 (.hdr → .exr)
                    format_only_count += 1
                    # 노출값 적용
                    if apply_exposure:
                        exposure = _read_hdr_exposure(str(file_path))
                        if exposure is not None:
                            image_hdr = image_hdr * exposure
                    image_hdr = np.maximum(image_hdr, 0.0)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    imsave(str(output_path), image_hdr)
                    continue

            # 통계 카운터 증가
            if need_resize and not need_bit_convert:
                resize_only_count += 1
            elif not need_resize and need_bit_convert:
                bit_convert_only_count += 1
            else:
                full_process_count += 1

            # 32비트 변환 필요시만
            if need_bit_convert:
                image_hdr = image_hdr.astype(np.float32)

            # 노출값 적용 (Laval 데이터셋의 경우 헤더에 노출 정보가 있음)
            if apply_exposure:
                exposure = _read_hdr_exposure(str(file_path))
                if exposure is not None:
                    image_hdr = image_hdr * exposure

            # 음수 값 제거
            image_hdr = np.maximum(image_hdr, 0.0)

            # 휘도 계산 (ITU-R BT.709)
            luminance = 0.2126 * image_hdr[:,:,0] + 0.7152 * image_hdr[:,:,1] + 0.0722 * image_hdr[:,:,2]
            luminance_stats.append({
                'file': str(file_path),
                'min': float(np.min(luminance)),
                'max': float(np.max(luminance)),
                'mean': float(np.mean(luminance)),
                'median': float(np.median(luminance))
            })

            # 리사이즈 필요시만
            if need_resize:
                image_hdr = cv2.resize(
                    image_hdr,
                    (target_w, target_h),
                    interpolation=cv2.INTER_LANCZOS4
                )

            # 출력 경로 생성
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # EXR 형식으로 저장
            imsave(str(output_path), image_hdr)

        except Exception as e:
            print(f"\nError processing {file_path}: {e}")
            failed_files.append(str(file_path))

    # 휘도 통계 출력
    if luminance_stats:
        all_max = [s['max'] for s in luminance_stats]
        all_mean = [s['mean'] for s in luminance_stats]
        print(f"\n휘도 통계 (cd/m²):")
        print(f"  최대 휘도 범위: {min(all_max):.1f} - {max(all_max):.1f}")
        print(f"  평균 휘도 범위: {min(all_mean):.1f} - {max(all_mean):.1f}")

    # 전처리 결과 출력
    processed_count = format_only_count + resize_only_count + bit_convert_only_count + full_process_count
    print(f"\n=== 전처리 결과 ===")
    print(f"전체 파일: {len(input_files)}개")
    print(f"")
    print(f"건너뜀: {skipped_count}개 (이미 {target_h}x{target_w} 32비트 EXR)")
    print(f"포맷 변환만: {format_only_count}개 (.hdr → .exr)")
    print(f"리사이즈만: {resize_only_count}개")
    print(f"32비트 변환만: {bit_convert_only_count}개")
    print(f"전체 처리: {full_process_count}개 (리사이즈 + 32비트)")
    if failed_files:
        print(f"실패: {len(failed_files)}개")
        for f in failed_files[:5]:
            print(f"  - {f}")
        if len(failed_files) > 5:
            print(f"  ... 외 {len(failed_files) - 5}개")

    return processed_count


def _read_hdr_exposure(file_path: str) -> float:
    """
    HDR 파일 헤더에서 노출값(EXPOSURE) 읽기

    Radiance HDR 형식의 EXPOSURE 키워드를 파싱합니다.
    """
    try:
        with open(file_path, 'rb') as f:
            header = b''
            while True:
                line = f.readline()
                if line == b'\n':
                    break
                header += line
                if len(header) > 10000:  # 헤더가 너무 길면 중단
                    break

            header_str = header.decode('ascii', errors='ignore')

            # EXPOSURE= 키워드 찾기
            for line in header_str.split('\n'):
                if line.upper().startswith('EXPOSURE='):
                    try:
                        return float(line.split('=')[1].strip())
                    except:
                        pass

    except Exception:
        pass

    return None


def validate_dataset(data_dir: str, expected_shape: tuple = (512, 1024, 3)):
    """
    전처리된 데이터셋의 유효성 검사

    Args:
        data_dir: 데이터셋 경로
        expected_shape: 예상 이미지 형태 (H, W, C)

    Returns:
        valid_count: 유효한 이미지 수
        invalid_files: 유효하지 않은 파일 목록
    """
    hdr_files = list(Path(data_dir).rglob('*.exr')) + list(Path(data_dir).rglob('*.hdr'))

    valid_count = 0
    invalid_files = []

    for file_path in tqdm(hdr_files, desc="Validating"):
        try:
            image = imread(str(file_path))

            if image is None:
                invalid_files.append((str(file_path), "Failed to load"))
                continue

            if image.shape != expected_shape:
                invalid_files.append((str(file_path), f"Shape mismatch: {image.shape}"))
                continue

            if not np.isfinite(image).all():
                invalid_files.append((str(file_path), "Contains NaN or Inf"))
                continue

            if np.min(image) < 0:
                invalid_files.append((str(file_path), f"Negative values: min={np.min(image):.4f}"))
                continue

            valid_count += 1

        except Exception as e:
            invalid_files.append((str(file_path), str(e)))

    print(f"\nValidation Results:")
    print(f"  Valid: {valid_count}/{len(hdr_files)}")
    print(f"  Invalid: {len(invalid_files)}")

    if invalid_files:
        print("\nInvalid files:")
        for path, reason in invalid_files[:10]:
            print(f"  - {path}: {reason}")
        if len(invalid_files) > 10:
            print(f"  ... and {len(invalid_files) - 10} more")

    return valid_count, invalid_files


def generate_dataset_json(data_dir: str, output_path: str = None):
    """
    StyleGAN 학습을 위한 dataset.json 생성

    Args:
        data_dir: 데이터셋 경로
        output_path: JSON 저장 경로 (기본값: data_dir/dataset.json)
    """
    import json

    if output_path is None:
        output_path = os.path.join(data_dir, 'dataset.json')

    hdr_files = list(Path(data_dir).rglob('*.exr')) + list(Path(data_dir).rglob('*.hdr'))

    # 레이블은 사용하지 않으므로 빈 배열 생성
    labels = {}
    for file_path in hdr_files:
        rel_path = str(file_path.relative_to(data_dir))
        labels[rel_path] = []  # 빈 레이블

    dataset_json = {'labels': list(labels.items())}

    with open(output_path, 'w') as f:
        json.dump(dataset_json, f, indent=2)

    print(f"Generated dataset.json with {len(labels)} entries at {output_path}")


def main():
    parser = argparse.ArgumentParser(description='HDR Dataset Preprocessing')

    parser.add_argument('--dataset', type=str, required=True,
                        choices=['s2r_hdr', 'laval', 'validate'],
                        help='Dataset type to process')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing raw HDR files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for processed files')
    parser.add_argument('--height', type=int, default=512,
                        help='Target height (default: 512)')
    parser.add_argument('--width', type=int, default=1024,
                        help='Target width (default: 1024)')
    parser.add_argument('--no-exposure', action='store_true',
                        help='Do not apply exposure correction (Laval only)')
    parser.add_argument('--generate-json', action='store_true',
                        help='Generate dataset.json after processing')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir + '_processed'

    target_resolution = (args.height, args.width)

    if args.dataset == 's2r_hdr':
        count = prepare_s2r_hdr(
            args.input_dir,
            args.output_dir,
            target_resolution=target_resolution
        )
    elif args.dataset == 'laval':
        count = prepare_laval_photometric(
            args.input_dir,
            args.output_dir,
            target_resolution=target_resolution,
            apply_exposure=not args.no_exposure
        )
    elif args.dataset == 'validate':
        validate_dataset(args.input_dir, expected_shape=(*target_resolution, 3))
        return

    if args.generate_json and count > 0:
        generate_dataset_json(args.output_dir)


if __name__ == '__main__':
    main()
