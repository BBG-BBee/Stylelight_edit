# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# TTA Augmentation
from .tta_augment import (
    ExpAug,
    WBAug,
    PermAug,
    FlipAug,
    IdentityAug,
    TTAAugmentor,
)

# S2R-Adapter
from .s2r_adapter import (
    S2RAdapterLinear,
    S2RAdapterFullyConnected,
    apply_s2r_adapter_to_generator,
    get_s2r_adapter_parameters,
    freeze_non_adapter_parameters,
    set_adapter_scales,
    get_adapter_scales,
    make_scales_learnable,
    # TTA functions
    compute_uncertainty_from_outputs,
    adaptive_scale_from_uncertainty,
    apply_adaptive_scales,
)

__all__ = [
    # TTA Augmentation
    'ExpAug',
    'WBAug',
    'PermAug',
    'FlipAug',
    'IdentityAug',
    'TTAAugmentor',
    # S2R-Adapter
    'S2RAdapterLinear',
    'S2RAdapterFullyConnected',
    'apply_s2r_adapter_to_generator',
    'get_s2r_adapter_parameters',
    'freeze_non_adapter_parameters',
    'set_adapter_scales',
    'get_adapter_scales',
    'make_scales_learnable',
    'compute_uncertainty_from_outputs',
    'adaptive_scale_from_uncertainty',
    'apply_adaptive_scales',
]
