"""
Benchmark Dataset Feature Extraction & Processing Pipeline
===========================================================
Extracts, aligns, and standardizes multi-modal behavioral and environmental
features from real-world biometric benchmarks:
  - RT-BENE (Real-Time Blink Detection in Natural Environments)
  - MPIIFaceGaze / MPIIGaze (3D Gaze Deviation & Head Pose)
  - DROZY Multimodal Fatigue Database (PERCLOS & Karolinska Sleepiness Scale)
  - CEW (Closed Eyes in the Wild - Eyelid Occlusion)
"""

import os
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Project Feature Specification
RAW_FEATURE_NAMES = [
    'd_avg', 'd_min', 'I_avg', 't_dur',
    'blink_rate', 'incomplete_blink_ratio',
    'gaze_dev', 'head_pose', 'cam_conf'
]

def compute_ear_from_landmarks(landmarks: List[Tuple[float, float]], eye_indices: List[int]) -> float:
    """
    Computes Eye Aspect Ratio (EAR) using Soukupová & Čech (2016) formulation.
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    p1, p2, p3, p4, p5, p6 = [landmarks[idx] for idx in eye_indices]
    
    vertical_1 = math.hypot(p2[0] - p6[0], p2[1] - p6[1])
    vertical_2 = math.hypot(p3[0] - p5[0], p3[1] - p5[1])
    horizontal = math.hypot(p1[0] - p4[0], p1[1] - p4[1])
    
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def process_benchmark_video_window(
    ear_sequence: List[float],
    gaze_angles: List[float],
    head_poses: List[float],
    sampling_rate_fps: float = 30.0,
    full_blink_ear_threshold: float = 0.20,
    incomplete_blink_ear_threshold: float = 0.26
) -> Dict[str, float]:
    """
    Processes temporal window of biometric signals (calibrated on RT-BENE & DROZY)
    into high-level behavioral telemetry features.
    """
    ear_arr = np.array(ear_sequence)
    duration_sec = len(ear_arr) / sampling_rate_fps
    
    # Identify blink events
    blinks = 0
    incomplete_blinks = 0
    is_blinking = False
    
    for val in ear_arr:
        if val < full_blink_ear_threshold:
            if not is_blinking:
                is_blinking = True
        else:
            if is_blinking:
                is_blinking = False
                blinks += 1
            elif val < incomplete_blink_ear_threshold:
                incomplete_blinks += 1
                
    total_blinks = blinks + incomplete_blinks
    blink_rate_bpm = (blinks / duration_sec) * 60.0 if duration_sec > 0 else 0.0
    incomplete_blink_ratio = (incomplete_blinks / total_blinks) if total_blinks > 0 else 0.0
    
    return {
        "blink_rate": round(float(blink_rate_bpm), 2),
        "incomplete_blink_ratio": round(float(incomplete_blink_ratio), 3),
        "gaze_dev": round(float(np.mean(gaze_angles)), 2) if len(gaze_angles) > 0 else 0.0,
        "head_pose": round(float(np.mean(head_poses)), 2) if len(head_poses) > 0 else 0.0,
        "t_dur": round(float(duration_sec), 2)
    }

def align_telemetry_with_user_baseline(
    raw_features: Dict[str, float],
    baseline: Dict[str, float]
) -> Dict[str, float]:
    """
    Computes per-user personal deviation vector: Δ = F - B(U).
    """
    combined = dict(raw_features)
    for feat in RAW_FEATURE_NAMES:
        combined[f"{feat}_dev"] = round(raw_features[feat] - baseline.get(feat, 0.0), 3)
    return combined
