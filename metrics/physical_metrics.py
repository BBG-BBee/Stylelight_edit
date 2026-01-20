"""
물리적 정합성 평가 메트릭

DGP(주광 눈부심 확률) 계산을 위한 물리적 휘도 정확도를 평가합니다.

주요 메트릭:
1. ΔEv (수직 조도 오차율): <10% 목표
2. RMSE_trans (전이 구간 RMSE): 300~1000 cd/m² 영역
3. DGP Class Accuracy: 눈부심 등급 분류 정확도

사용법:
    from metrics.physical_metrics import PhysicalMetrics

    metrics = PhysicalMetrics()
    results = metrics.evaluate(pred_hdr, gt_hdr)
    print(f"ΔEv: {results['delta_ev']:.2f}%")
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List


class PhysicalMetrics:
    """
    물리적 정합성 평가 메트릭 클래스

    DGP 계산에 필요한 휘도 정확도를 다양한 관점에서 평가합니다.

    Args:
        T_onset: 전이 구간 시작 임계값 (cd/m²)
        T_peak: 전이 구간 종료 임계값 (cd/m²)
        dgp_thresholds: DGP 등급 임계값
    """

    # DGP 등급 분류 (CIE 표준)
    DGP_CLASSES = {
        'imperceptible': (0.0, 0.35),      # 감지 불가
        'perceptible': (0.35, 0.40),        # 감지 가능
        'disturbing': (0.40, 0.45),         # 방해됨
        'intolerable': (0.45, 1.0),         # 견딜 수 없음
    }

    def __init__(self,
                 T_onset: float = 300.0,
                 T_peak: float = 1000.0,
                 dgp_thresholds: Optional[Dict] = None):
        self.T_onset = T_onset
        self.T_peak = T_peak
        self.dgp_thresholds = dgp_thresholds or self.DGP_CLASSES

    def rgb_to_luminance(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        RGB를 휘도로 변환 (ITU-R BT.709)

        Args:
            rgb: RGB 이미지 (B, 3, H, W) 또는 (3, H, W)

        Returns:
            luminance: 휘도 (B, 1, H, W) 또는 (1, H, W)
        """
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)

        # ITU-R BT.709
        weights = torch.tensor([0.2126, 0.7152, 0.0722],
                              device=rgb.device, dtype=rgb.dtype)
        luminance = (rgb * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)

        return luminance

    def compute_vertical_illuminance(self, hdr_image: torch.Tensor) -> torch.Tensor:
        """
        수직 조도 (Ev) 계산

        반구형 HDR에서 수직면에 입사하는 조도를 계산합니다.
        Ev = ∫∫ L(θ,φ) * cos(θ) * dΩ

        Args:
            hdr_image: HDR 이미지 (B, 3, H, W), Equirectangular 또는 Fisheye

        Returns:
            Ev: 수직 조도 (B,) [lux]
        """
        B, C, H, W = hdr_image.shape
        device = hdr_image.device

        # 휘도 계산
        luminance = self.rgb_to_luminance(hdr_image)  # (B, 1, H, W)

        # 좌표 그리드 생성 (Equirectangular 가정)
        # θ: 0 ~ π (위에서 아래)
        # φ: 0 ~ 2π (왼쪽에서 오른쪽)
        theta = torch.linspace(0, np.pi, H, device=device)
        phi = torch.linspace(0, 2 * np.pi, W, device=device)

        theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')

        # 입체각 요소: dΩ = sin(θ) * dθ * dφ
        d_theta = np.pi / H
        d_phi = 2 * np.pi / W
        solid_angle = torch.sin(theta_grid) * d_theta * d_phi

        # 코사인 가중치 (수직면에 대한 입사각)
        # 전방(φ=π)을 향한 수직면 기준
        cos_weight = torch.sin(theta_grid) * torch.cos(phi_grid - np.pi)
        cos_weight = torch.clamp(cos_weight, min=0)  # 후방은 제외

        # 조도 적분
        weight = solid_angle * cos_weight
        weight = weight.view(1, 1, H, W).expand(B, 1, H, W)

        Ev = (luminance * weight).sum(dim=[2, 3]).squeeze(1)

        return Ev

    def delta_ev(self,
                 pred_hdr: torch.Tensor,
                 gt_hdr: torch.Tensor) -> Dict[str, float]:
        """
        수직 조도 오차율 (ΔEv) 계산

        목표: <10%

        Args:
            pred_hdr: 예측 HDR 이미지
            gt_hdr: 정답 HDR 이미지

        Returns:
            metrics: {
                'pred_ev': 예측 Ev [lux],
                'gt_ev': 정답 Ev [lux],
                'delta_ev_abs': 절대 오차 [lux],
                'delta_ev_rel': 상대 오차율 [%]
            }
        """
        pred_ev = self.compute_vertical_illuminance(pred_hdr)
        gt_ev = self.compute_vertical_illuminance(gt_hdr)

        delta_abs = torch.abs(pred_ev - gt_ev)
        delta_rel = (delta_abs / (gt_ev + 1e-6)) * 100

        return {
            'pred_ev': pred_ev.mean().item(),
            'gt_ev': gt_ev.mean().item(),
            'delta_ev_abs': delta_abs.mean().item(),
            'delta_ev_rel': delta_rel.mean().item()
        }

    def rmse_transition_zone(self,
                             pred_hdr: torch.Tensor,
                             gt_hdr: torch.Tensor) -> Dict[str, float]:
        """
        전이 구간 (T_onset ~ T_peak) RMSE 계산

        DGP 계산에 가장 중요한 휘도 영역의 오차를 측정합니다.

        Args:
            pred_hdr: 예측 HDR 이미지
            gt_hdr: 정답 HDR 이미지

        Returns:
            metrics: {
                'rmse_transition': 전이 구간 RMSE [cd/m²],
                'rmse_background': 배경 구간 RMSE,
                'rmse_glare': 눈부심 구간 RMSE,
                'rmse_total': 전체 RMSE
            }
        """
        pred_lum = self.rgb_to_luminance(pred_hdr)
        gt_lum = self.rgb_to_luminance(gt_hdr)

        # 구간 마스크
        bg_mask = gt_lum < self.T_onset
        trans_mask = (gt_lum >= self.T_onset) & (gt_lum < self.T_peak)
        glare_mask = gt_lum >= self.T_peak

        # 오차 계산
        diff = pred_lum - gt_lum
        sq_diff = diff ** 2

        # 구간별 RMSE
        def safe_rmse(sq_diff, mask):
            if mask.sum() > 0:
                return torch.sqrt(sq_diff[mask].mean()).item()
            return 0.0

        return {
            'rmse_transition': safe_rmse(sq_diff, trans_mask),
            'rmse_background': safe_rmse(sq_diff, bg_mask),
            'rmse_glare': safe_rmse(sq_diff, glare_mask),
            'rmse_total': torch.sqrt(sq_diff.mean()).item(),
            'pixel_count_transition': trans_mask.sum().item(),
            'pixel_count_glare': glare_mask.sum().item()
        }

    def classify_dgp(self, dgp_value: float) -> str:
        """
        DGP 값을 등급으로 분류

        Args:
            dgp_value: DGP 값 (0 ~ 1)

        Returns:
            class_name: 등급 이름
        """
        for class_name, (low, high) in self.dgp_thresholds.items():
            if low <= dgp_value < high:
                return class_name
        return 'intolerable'

    def dgp_class_accuracy(self,
                           pred_dgp_list: List[float],
                           gt_dgp_list: List[float]) -> Dict[str, float]:
        """
        DGP 등급 분류 정확도

        Args:
            pred_dgp_list: 예측 DGP 값 리스트
            gt_dgp_list: 정답 DGP 값 리스트

        Returns:
            metrics: {
                'accuracy': 전체 정확도 [%],
                'class_accuracy': 클래스별 정확도,
                'confusion_matrix': 혼동 행렬
            }
        """
        pred_classes = [self.classify_dgp(d) for d in pred_dgp_list]
        gt_classes = [self.classify_dgp(d) for d in gt_dgp_list]

        # 전체 정확도
        correct = sum(p == g for p, g in zip(pred_classes, gt_classes))
        accuracy = (correct / len(pred_classes)) * 100 if pred_classes else 0.0

        # 클래스별 정확도
        class_names = list(self.dgp_thresholds.keys())
        class_accuracy = {}
        for cls in class_names:
            cls_indices = [i for i, g in enumerate(gt_classes) if g == cls]
            if cls_indices:
                cls_correct = sum(pred_classes[i] == cls for i in cls_indices)
                class_accuracy[cls] = (cls_correct / len(cls_indices)) * 100
            else:
                class_accuracy[cls] = 0.0

        # 혼동 행렬
        confusion = {gt: {pred: 0 for pred in class_names} for gt in class_names}
        for p, g in zip(pred_classes, gt_classes):
            confusion[g][p] += 1

        return {
            'accuracy': accuracy,
            'class_accuracy': class_accuracy,
            'confusion_matrix': confusion
        }

    def luminance_histogram_similarity(self,
                                       pred_hdr: torch.Tensor,
                                       gt_hdr: torch.Tensor,
                                       n_bins: int = 100) -> Dict[str, float]:
        """
        휘도 히스토그램 유사도

        로그 스케일 휘도 분포의 유사성을 측정합니다.

        Args:
            pred_hdr: 예측 HDR
            gt_hdr: 정답 HDR
            n_bins: 히스토그램 빈 수

        Returns:
            metrics: {
                'histogram_intersection': 히스토그램 교차 (높을수록 좋음),
                'kl_divergence': KL 발산 (낮을수록 좋음),
                'earth_mover_distance': EMD (낮을수록 좋음)
            }
        """
        pred_lum = self.rgb_to_luminance(pred_hdr).flatten()
        gt_lum = self.rgb_to_luminance(gt_hdr).flatten()

        # 로그 스케일 변환
        pred_log = torch.log10(pred_lum + 1e-6)
        gt_log = torch.log10(gt_lum + 1e-6)

        # 공통 범위 결정
        min_val = min(pred_log.min().item(), gt_log.min().item())
        max_val = max(pred_log.max().item(), gt_log.max().item())

        # 히스토그램 계산
        pred_hist = torch.histc(pred_log, bins=n_bins, min=min_val, max=max_val)
        gt_hist = torch.histc(gt_log, bins=n_bins, min=min_val, max=max_val)

        # 정규화
        pred_hist = pred_hist / (pred_hist.sum() + 1e-6)
        gt_hist = gt_hist / (gt_hist.sum() + 1e-6)

        # 히스토그램 교차
        intersection = torch.min(pred_hist, gt_hist).sum().item()

        # KL 발산 (pred || gt)
        kl_div = (pred_hist * torch.log((pred_hist + 1e-10) / (gt_hist + 1e-10))).sum().item()

        # EMD (1D Wasserstein distance)
        pred_cdf = torch.cumsum(pred_hist, dim=0)
        gt_cdf = torch.cumsum(gt_hist, dim=0)
        emd = torch.abs(pred_cdf - gt_cdf).sum().item() / n_bins

        return {
            'histogram_intersection': intersection,
            'kl_divergence': kl_div,
            'earth_mover_distance': emd
        }

    def peak_luminance_accuracy(self,
                                pred_hdr: torch.Tensor,
                                gt_hdr: torch.Tensor,
                                top_k: int = 100) -> Dict[str, float]:
        """
        피크 휘도 정확도

        눈부심 소스(광원) 영역의 휘도 정확도를 측정합니다.

        Args:
            pred_hdr: 예측 HDR
            gt_hdr: 정답 HDR
            top_k: 상위 K개 픽셀 고려

        Returns:
            metrics: {
                'peak_pred': 예측 최대 휘도,
                'peak_gt': 정답 최대 휘도,
                'peak_error_rel': 피크 상대 오차 [%],
                'top_k_rmse': 상위 K개 픽셀 RMSE
            }
        """
        pred_lum = self.rgb_to_luminance(pred_hdr).flatten()
        gt_lum = self.rgb_to_luminance(gt_hdr).flatten()

        # 최대값
        peak_pred = pred_lum.max().item()
        peak_gt = gt_lum.max().item()
        peak_error = abs(peak_pred - peak_gt) / (peak_gt + 1e-6) * 100

        # 상위 K개 픽셀
        _, gt_top_indices = torch.topk(gt_lum, min(top_k, len(gt_lum)))
        pred_top_values = pred_lum[gt_top_indices]
        gt_top_values = gt_lum[gt_top_indices]

        top_k_rmse = torch.sqrt(((pred_top_values - gt_top_values) ** 2).mean()).item()

        return {
            'peak_pred': peak_pred,
            'peak_gt': peak_gt,
            'peak_error_rel': peak_error,
            'top_k_rmse': top_k_rmse
        }

    def evaluate(self,
                 pred_hdr: torch.Tensor,
                 gt_hdr: torch.Tensor) -> Dict[str, any]:
        """
        종합 평가

        모든 물리적 메트릭을 계산합니다.

        Args:
            pred_hdr: 예측 HDR 이미지 (B, 3, H, W)
            gt_hdr: 정답 HDR 이미지 (B, 3, H, W)

        Returns:
            metrics: 모든 메트릭 딕셔너리
        """
        results = {}

        # 수직 조도 오차
        ev_metrics = self.delta_ev(pred_hdr, gt_hdr)
        results.update({f'ev_{k}': v for k, v in ev_metrics.items()})

        # 전이 구간 RMSE
        rmse_metrics = self.rmse_transition_zone(pred_hdr, gt_hdr)
        results.update({f'rmse_{k}': v for k, v in rmse_metrics.items()})

        # 히스토그램 유사도
        hist_metrics = self.luminance_histogram_similarity(pred_hdr, gt_hdr)
        results.update({f'hist_{k}': v for k, v in hist_metrics.items()})

        # 피크 휘도 정확도
        peak_metrics = self.peak_luminance_accuracy(pred_hdr, gt_hdr)
        results.update({f'peak_{k}': v for k, v in peak_metrics.items()})

        # 요약
        results['summary'] = {
            'delta_ev_rel': ev_metrics['delta_ev_rel'],
            'rmse_transition': rmse_metrics['rmse_transition'],
            'histogram_intersection': hist_metrics['histogram_intersection'],
            'peak_error_rel': peak_metrics['peak_error_rel']
        }

        return results

    def print_report(self, metrics: Dict[str, any]):
        """평가 결과 출력"""
        print('\n' + '=' * 60)
        print('물리적 정합성 평가 결과')
        print('=' * 60)

        print('\n[수직 조도 (Ev)]')
        print(f"  예측 Ev: {metrics.get('ev_pred_ev', 0):.2f} lux")
        print(f"  정답 Ev: {metrics.get('ev_gt_ev', 0):.2f} lux")
        print(f"  ΔEv (상대): {metrics.get('ev_delta_ev_rel', 0):.2f}% {'✓' if metrics.get('ev_delta_ev_rel', 100) < 10 else '✗'}")

        print('\n[RMSE by Zone]')
        print(f"  배경 (< {self.T_onset} cd/m²): {metrics.get('rmse_rmse_background', 0):.2f} cd/m²")
        print(f"  전이 ({self.T_onset}-{self.T_peak} cd/m²): {metrics.get('rmse_rmse_transition', 0):.2f} cd/m²")
        print(f"  눈부심 (≥ {self.T_peak} cd/m²): {metrics.get('rmse_rmse_glare', 0):.2f} cd/m²")
        print(f"  전체: {metrics.get('rmse_rmse_total', 0):.2f} cd/m²")

        print('\n[휘도 분포]')
        print(f"  히스토그램 교차: {metrics.get('hist_histogram_intersection', 0):.4f}")
        print(f"  KL 발산: {metrics.get('hist_kl_divergence', 0):.4f}")
        print(f"  EMD: {metrics.get('hist_earth_mover_distance', 0):.4f}")

        print('\n[피크 휘도]')
        print(f"  예측 최대: {metrics.get('peak_peak_pred', 0):.2f} cd/m²")
        print(f"  정답 최대: {metrics.get('peak_peak_gt', 0):.2f} cd/m²")
        print(f"  피크 오차: {metrics.get('peak_peak_error_rel', 0):.2f}%")

        print('\n' + '=' * 60)
