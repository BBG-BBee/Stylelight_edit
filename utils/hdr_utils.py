"""HDR 이미지 처리 유틸리티"""
import numpy as np
import cv2


def log_domain_resize(image_hdr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    로그 도메인에서 HDR 이미지 리사이즈 (기하평균 기반)

    선형 공간의 산술평균 대신 로그 공간에서 산술평균을 수행하여
    HDR 피크 휘도를 보존하고 Lanczos 링잉 아티팩트를 방지한다.

    원리:
        log(image + eps) → cv2.resize(INTER_AREA) → exp(result) - eps
        로그 공간의 산술평균 = 선형 공간의 기하평균
        예) 선형 평균: (50,000 + 100) / 2 = 25,050  ← 밝은 쪽 편향
            기하 평균: sqrt(50,000 × 100) = 2,236    ← 지각적 중간점

    Args:
        image_hdr: (H, W, C) FP32 HDR 이미지, 값 >= 0
        target_h: 목표 높이
        target_w: 목표 너비

    Returns:
        (target_h, target_w, C) FP32 리사이즈된 HDR 이미지
    """
    eps = 1e-6
    log_img = np.log(image_hdr + eps)
    log_resized = cv2.resize(log_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return np.maximum(np.exp(log_resized) - eps, 0.0)
