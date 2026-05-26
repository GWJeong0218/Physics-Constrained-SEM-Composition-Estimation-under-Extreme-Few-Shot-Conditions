"""
Synthetic SEM image generator
-----------------------------

Generates SEM-like synthetic images for composition-estimation experiments.
The generator uses a small set of explicit assumptions:

- composition vectors are non-negative and sum to one
- image intensity is linked to a Z-contrast-inspired proxy
- each class represents a different morphology regime
- SEM-style artifacts are added after the morphology/intensity step
- final image tone is matched to a small set of real SEM reference images

This is a reference script for the experimental pipeline, not a general SEM
simulator.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


# Global defaults

IMG_SIZE = 224
NUM_CLASSES = 4
IMAGES_PER_CLASS = 500

ELEMENTS = ["O", "Si", "Co", "Pd", "C"]
Z_MAP = {"O": 8, "Si": 14, "Co": 27, "Pd": 46, "C": 6}
Z_VEC = np.array([Z_MAP[e] for e in ELEMENTS], dtype=np.float32)


# Tone matching

def compute_real_tone_stats(real_paths: Iterable[Path]) -> Dict[str, float]:
    vals = []

    for path in real_paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Failed to read real SEM reference: {path}")
            continue
        vals.append(img.reshape(-1))

    if not vals:
        raise RuntimeError(
            "No real SEM reference images could be read. "
            "Check the paths passed to --real-ref."
        )

    vals = np.concatenate(vals, axis=0).astype(np.float32)
    p1, p5, p50, p95, p99 = np.percentile(vals, [1, 5, 50, 95, 99])

    return {
        "p1": float(p1),
        "p5": float(p5),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
    }


def match_real_tone(img01: np.ndarray, tone: Dict[str, float]) -> np.ndarray:
    x = np.clip(img01, 0, 1).astype(np.float32)
    x_u8 = x * 255.0

    s5, _, s95 = np.percentile(x_u8, [5, 50, 95])
    r5, r50, r95 = tone["p5"], tone["p50"], tone["p95"]

    scale = (r95 - r5) / (s95 - s5 + 1e-6)
    y = (x_u8 - s5) * scale + r5

    cur50 = np.percentile(y, 50)
    y = y + 0.6 * (r50 - cur50)
    y = np.clip(y, tone["p1"], tone["p99"])

    return y.astype(np.uint8)

# Composition and intensity model

def sample_composition(cls: int, base_conc: float = 18.0, class_strength: float = 0.10) -> np.ndarray:
    base = np.ones(len(ELEMENTS), dtype=np.float32)
    weak_bias = {
        0: np.array([1.05, 1.05, 0.95, 0.95, 1.00], dtype=np.float32),
        1: np.array([0.95, 1.00, 1.05, 1.00, 1.00], dtype=np.float32),
        2: np.array([1.00, 0.95, 1.00, 1.05, 1.00], dtype=np.float32),
        3: np.array([1.05, 0.95, 0.95, 0.95, 1.10], dtype=np.float32),
    }[cls]

    bias = (1.0 - class_strength) * base + class_strength * weak_bias
    alpha = bias / bias.sum() * base_conc
    return np.random.dirichlet(alpha).astype(np.float32)


def zeff_from_comp(y: np.ndarray, p_range: Tuple[float, float] = (0.6, 2.2)) -> float:
    p = np.random.uniform(*p_range)
    return float(np.sum(y * (Z_VEC ** p)) ** (1.0 / p))


def intensity_from_zeff(zeff: float) -> float:
    zmin, zmax = 6.0, 46.0
    x = (zeff - zmin) / (zmax - zmin + 1e-6)
    x = np.clip(x, 0, 1)

    gamma = np.random.uniform(0.7, 2.2)
    x = x ** gamma

    a = np.random.uniform(0.6, 1.5)
    b = np.random.uniform(-0.12, 0.12)
    x = np.clip(a * x + b, 0, 1)

    k = np.random.uniform(2.0, 6.0)
    x = 1.0 / (1.0 + np.exp(-k * (x - 0.5)))

    return float(np.clip(x, 0, 1))

# Shape utilities

def irregular_blob(canvas: np.ndarray, cx: int, cy: int, r: int, n_verts: int = 16) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, n_verts, endpoint=False)
    angles += np.random.uniform(-0.18, 0.18, size=n_verts)

    rad = r * (1.0 + np.random.normal(0, 0.20, size=n_verts))
    rad = np.clip(rad, r * 0.55, r * 1.7)

    pts = []
    for a, rr in zip(angles, rad):
        x = int(cx + rr * np.cos(a))
        y = int(cy + rr * np.sin(a))
        pts.append([np.clip(x, 0, IMG_SIZE - 1), np.clip(y, 0, IMG_SIZE - 1)])

    pts = np.array([pts], dtype=np.int32)
    cv2.fillPoly(canvas, pts, 1.0)
    return canvas


def roughen_mask(mask: np.ndarray, roughness: float = 0.7) -> np.ndarray:
    k1 = int(np.random.choice([3, 5, 7]))
    ker1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1))

    if np.random.rand() < 0.5:
        mask = cv2.erode(mask, ker1, iterations=int(np.random.randint(1, 3)))
    else:
        mask = cv2.dilate(mask, ker1, iterations=int(np.random.randint(1, 3)))

    grid = int(np.random.choice([6, 8, 10]))
    field = cv2.resize(
        np.random.randn(grid, grid).astype(np.float32),
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_CUBIC,
    )
    field = (field - field.mean()) / (field.std() + 1e-6)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=np.random.uniform(1.0, 2.5))

    thresh = 0.5 + roughness * 0.10 * field
    out = (mask > thresh).astype(np.float32)

    if np.random.rand() < 0.8:
        out = cv2.GaussianBlur(out, (3, 3), 0)

    return np.clip(out, 0, 1)


def add_intragranular_texture(img: np.ndarray, strength_range: Tuple[float, float] = (0.03, 0.06)) -> np.ndarray:
    s = np.random.uniform(*strength_range)

    n1 = np.random.randn(IMG_SIZE, IMG_SIZE).astype(np.float32)
    n1 = cv2.GaussianBlur(n1, (0, 0), sigmaX=np.random.uniform(0.6, 1.2))

    grid = int(np.random.choice([10, 14, 18]))
    n2 = cv2.resize(
        np.random.randn(grid, grid).astype(np.float32),
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_CUBIC,
    )
    n2 = cv2.GaussianBlur(n2, (0, 0), sigmaX=np.random.uniform(1.5, 3.0))

    field = 0.7 * n1 + 0.3 * n2
    field = field / (field.std() + 1e-6)

    out = img * (1.0 + s * field)
    out = out + np.random.normal(0, s * 0.15, size=img.shape).astype(np.float32)
    return np.clip(out, 0, 1)


def anisotropic_stretch(img: np.ndarray, sx: float, sy: float) -> np.ndarray:
    h, w = img.shape
    mat = np.array(
        [[sx, 0, (1 - sx) * w / 2], [0, sy, (1 - sy) * h / 2]],
        dtype=np.float32,
    )
    out = cv2.warpAffine(
        img,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return np.clip(out, 0, 1)


def add_streak_and_bias(
    img: np.ndarray,
    streak_strength_range: Tuple[float, float] = (0.015, 0.045),
    bias_strength_range: Tuple[float, float] = (0.03, 0.10),
) -> np.ndarray:
    h, w = img.shape

    bx = np.random.uniform(-1, 1)
    by = np.random.uniform(-1, 1)
    xs = np.linspace(-1, 1, w, dtype=np.float32)
    ys = np.linspace(-1, 1, h, dtype=np.float32)
    xgrid, ygrid = np.meshgrid(xs, ys)

    grad = bx * xgrid + by * ygrid
    grad = grad / (np.max(np.abs(grad)) + 1e-6)

    b = np.random.uniform(*bias_strength_range)
    out = img * (1.0 + b * grad)

    s = np.random.uniform(*streak_strength_range)
    period = int(np.random.randint(24, 80))
    row = np.arange(h, dtype=np.float32)
    band = np.sin(2 * np.pi * row / period) + 0.25 * np.random.randn(h).astype(np.float32)
    band = (band - band.mean()) / (band.std() + 1e-6)

    out = out * (1.0 + s * band[:, None])
    return np.clip(out, 0, 1)


def shear_warp(img: np.ndarray, shx: float = 0.12, shy: float = 0.0) -> np.ndarray:
    h, w = img.shape
    mat = np.array(
        [[1, shx, -shx * w / 2], [shy, 1, -shy * h / 2]],
        dtype=np.float32,
    )
    out = cv2.warpAffine(
        img,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return np.clip(out, 0, 1)

# Morphology generators

def gen_fine_dense() -> np.ndarray:
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    n = int(np.random.randint(320, 520))

    for _ in range(n):
        r = int(np.random.randint(2, 5))
        x, y = np.random.randint(0, IMG_SIZE, 2)
        if np.random.rand() < 0.85:
            img = irregular_blob(img, int(x), int(y), r, n_verts=int(np.random.randint(10, 18)))
        else:
            cv2.circle(img, (int(x), int(y)), r, 1, -1)

    return roughen_mask(img, roughness=np.random.uniform(0.35, 0.75))


def gen_coarse_growth() -> np.ndarray:
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    n = int(np.random.randint(25, 70))

    for _ in range(n):
        r = int(np.random.randint(10, 22))
        x, y = np.random.randint(0, IMG_SIZE, 2)
        img = irregular_blob(img, int(x), int(y), r, n_verts=int(np.random.randint(12, 22)))

    return roughen_mask(img, roughness=np.random.uniform(0.45, 0.90))


def gen_agglomerated() -> np.ndarray:
    cx = IMG_SIZE // 2 + int(np.random.randint(-25, 25))
    cy = IMG_SIZE // 2 + int(np.random.randint(-25, 25))

    core = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    core_n = int(np.random.randint(120, 260))

    for _ in range(core_n):
        r = int(np.random.randint(3, 9))
        x = int(np.random.normal(cx, 18))
        y = int(np.random.normal(cy, 18))
        if 0 <= x < IMG_SIZE and 0 <= y < IMG_SIZE:
            core = irregular_blob(core, x, y, r, n_verts=int(np.random.randint(10, 18)))

    if np.random.rand() < 0.85:
        core = shear_warp(
            core,
            shx=np.random.uniform(-0.18, 0.18),
            shy=np.random.uniform(-0.05, 0.05),
        )

    sat = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    sat_n = int(np.random.randint(60, 140))

    for _ in range(sat_n):
        r = int(np.random.randint(2, 6))
        x, y = np.random.randint(0, IMG_SIZE, 2)
        if np.random.rand() < 0.7:
            sat = irregular_blob(sat, int(x), int(y), r, n_verts=int(np.random.randint(10, 16)))
        else:
            cv2.circle(sat, (int(x), int(y)), r, 1, -1)

    img = np.maximum(core, sat)
    return roughen_mask(img, roughness=np.random.uniform(0.55, 1.00))


def gen_sparse_charged() -> np.ndarray:
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    n = int(np.random.randint(15, 50))

    for _ in range(n):
        r = int(np.random.randint(6, 14))
        x, y = np.random.randint(0, IMG_SIZE, 2)
        img = irregular_blob(img, int(x), int(y), r, n_verts=int(np.random.randint(10, 18)))

    return roughen_mask(img, roughness=np.random.uniform(0.30, 0.80))


GEN_BY_CLASS = {
    0: gen_fine_dense,
    1: gen_coarse_growth,
    2: gen_agglomerated,
    3: gen_sparse_charged,
}

# Phase and imaging model

def make_precip_mask(num: int, r_range: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    for _ in range(num):
        r = int(np.random.randint(r_range[0], r_range[1]))
        x, y = np.random.randint(0, IMG_SIZE, 2)
        cv2.circle(mask, (int(x), int(y)), r, 1, -1)

    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return np.clip(mask, 0, 1)


def perturb_composition(y: np.ndarray, sigma: float) -> np.ndarray:
    y2 = y + np.random.normal(0, sigma, size=y.shape).astype(np.float32)
    y2 = np.clip(y2, 1e-4, None)
    return y2 / y2.sum()


def add_sem_background(
    img: np.ndarray,
    base_range: Tuple[float, float] = (0.06, 0.18),
    lf_strength_range: Tuple[float, float] = (0.03, 0.12),
) -> np.ndarray:
    h, w = img.shape
    out = img + np.random.uniform(*base_range)

    grid = int(np.random.choice([6, 8, 10]))
    small = np.random.randn(grid, grid).astype(np.float32)
    field = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    field = (field - field.mean()) / (field.std() + 1e-6)

    out = out + np.random.uniform(*lf_strength_range) * field
    out = out + np.random.normal(0, 0.01, size=(h, w)).astype(np.float32)
    return np.clip(out, 0, 1)


def apply_blur(img: np.ndarray, prob: float) -> np.ndarray:
    if np.random.rand() < prob:
        k = int(np.random.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img


def apply_intensity_variation(img: np.ndarray, strength: float) -> np.ndarray:
    alpha = np.random.uniform(1.0 - strength, 1.0 + strength)
    beta = np.random.uniform(-0.2 * strength, 0.2 * strength)
    return np.clip(alpha * img + beta, 0, 1)


def apply_shot_noise(img: np.ndarray, exposure_range: Tuple[float, float]) -> np.ndarray:
    exposure = np.random.uniform(*exposure_range)
    lam = np.clip(img, 0, 1) * exposure
    noisy = np.random.poisson(lam).astype(np.float32) / exposure
    return np.clip(noisy, 0, 1)


def apply_lowfreq_gradient(img: np.ndarray, strength: float, grid: int) -> np.ndarray:
    small = np.random.randn(grid, grid).astype(np.float32)
    field = cv2.resize(small, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    field = (field - field.min()) / (field.max() - field.min() + 1e-6)
    field = (field - 0.5) * 2.0

    out = img * (1.0 + strength * field)
    out += np.random.uniform(-0.05, 0.05) * strength
    return np.clip(out, 0, 1)


def apply_drift(img: np.ndarray, max_shift: float) -> np.ndarray:
    dx = np.random.uniform(-max_shift, max_shift)
    dy = np.random.uniform(-max_shift, max_shift)
    mat = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)

    out = cv2.warpAffine(
        img,
        mat,
        (IMG_SIZE, IMG_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return np.clip(out, 0, 1)


def apply_scanline(
    img: np.ndarray,
    strength: float,
    period_range: Tuple[int, int] = (20, 70),
) -> np.ndarray:
    period = int(np.random.randint(period_range[0], period_range[1] + 1))
    row = np.arange(IMG_SIZE, dtype=np.float32)
    band = np.sin(2.0 * np.pi * row / period)
    band = band + 0.3 * np.random.randn(IMG_SIZE).astype(np.float32)
    band = (band - band.min()) / (band.max() - band.min() + 1e-6)
    band = 1.0 + strength * ((band - 0.5) * 2.0)

    return np.clip(img * band[:, None], 0, 1)


def apply_partial_occlusion(img: np.ndarray, max_occ_ratio: float = 0.18) -> np.ndarray:
    h, w = img.shape
    occ_h = int(np.random.uniform(0.06, max_occ_ratio) * h)
    occ_w = int(np.random.uniform(0.06, max_occ_ratio) * w)
    x0 = int(np.random.randint(0, w - occ_w))
    y0 = int(np.random.randint(0, h - occ_h))

    mask = np.zeros((h, w), dtype=np.float32)
    mask[y0:y0 + occ_h, x0:x0 + occ_w] = 1.0
    k = int(np.random.choice([9, 13, 17]))
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask = np.clip(mask, 0, 1)

    bg = max(float(np.percentile(img, 50)), 0.05)
    dark = np.random.uniform(0.25, 0.55)
    stain = img * dark + bg * np.random.uniform(0.8, 1.2)
    stain = stain + np.random.normal(0, 0.01, size=img.shape).astype(np.float32)
    stain = np.clip(stain, 0, 1)

    out = img * (1.0 - mask) + stain * mask
    return np.clip(out, 0, 1)


def apply_scale_variation(
    img: np.ndarray,
    min_scale: float = 0.85,
    max_scale: float = 1.15,
) -> np.ndarray:
    scale = np.random.uniform(min_scale, max_scale)
    h, w = img.shape
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    new_h = max(2, new_h)
    new_w = max(2, new_w)

    img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if scale < 1.0:
        pad_top = (h - new_h) // 2
        pad_bottom = h - new_h - pad_top
        pad_left = (w - new_w) // 2
        pad_right = w - new_w - pad_left

        out = cv2.copyMakeBorder(
            img_scaled,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )
        return np.clip(out[:h, :w], 0, 1)

    start_h = (new_h - h) // 2
    start_w = (new_w - w) // 2
    out = img_scaled[start_h:start_h + h, start_w:start_w + w]
    return np.clip(out, 0, 1)


CLASS_PARAMS = {
    0: dict(blur_prob=0.25, intensity=0.10, exposure=(90, 240), drift_p=0.20, scan_p=0.20, scan_s=(0.01, 0.04)),
    1: dict(blur_prob=0.35, intensity=0.16, exposure=(80, 220), drift_p=0.25, scan_p=0.25, scan_s=(0.01, 0.05)),
    2: dict(blur_prob=0.45, intensity=0.20, exposure=(70, 200), drift_p=0.30, scan_p=0.30, scan_s=(0.02, 0.06)),
    3: dict(blur_prob=0.55, intensity=0.26, exposure=(55, 170), drift_p=0.55, scan_p=0.60, scan_s=(0.03, 0.09)),
}


def class_phase_params(cls: int):
    if cls == 0:
        return int(np.random.randint(30, 120)), (2, 7), np.random.uniform(0.015, 0.040)
    if cls == 1:
        return int(np.random.randint(10, 60)), (4, 12), np.random.uniform(0.015, 0.050)
    if cls == 2:
        return int(np.random.randint(20, 100)), (3, 10), np.random.uniform(0.020, 0.060)
    return int(np.random.randint(8, 40)), (5, 14), np.random.uniform(0.015, 0.045)


def render_one(cls: int, tone: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    y = sample_composition(
        cls,
        base_conc=np.random.uniform(14.0, 26.0),
        class_strength=0.10,
    )

    base = GEN_BY_CLASS[cls]()

    num_precip, r_range, sigma = class_phase_params(cls)
    mask = make_precip_mask(num_precip, r_range)
    y2 = perturb_composition(y, sigma=sigma)

    i1 = intensity_from_zeff(zeff_from_comp(y))
    i2 = intensity_from_zeff(zeff_from_comp(y2))
    img = base * ((1.0 - mask) * i1 + mask * i2)

    params = CLASS_PARAMS[cls]
    img = add_sem_background(img)
    img = apply_blur(img, prob=params["blur_prob"])
    img = apply_intensity_variation(img, strength=params["intensity"])
    img = apply_shot_noise(img, exposure_range=params["exposure"])
    img = apply_lowfreq_gradient(
        img,
        strength=np.random.uniform(0.04, 0.14),
        grid=int(np.random.choice([6, 8, 10])),
    )

    if np.random.rand() < params["drift_p"]:
        img = apply_drift(img, max_shift=np.random.uniform(0.5, 2.5))
    if np.random.rand() < params["scan_p"]:
        img = apply_scanline(
            img,
            strength=np.random.uniform(*params["scan_s"]),
            period_range=(20, 70),
        )

    if np.random.rand() < 0.35:
        img = apply_partial_occlusion(img, max_occ_ratio=0.18)
    if np.random.rand() < 0.60:
        img = apply_scale_variation(img, min_scale=0.85, max_scale=1.15)

    img = add_intragranular_texture(img, strength_range=(0.03, 0.06))

    if cls == 3:
        sx = np.random.uniform(1.05, 1.15)
        sy = np.random.uniform(0.92, 0.98)
        if np.random.rand() < 0.5:
            sx, sy = sy, sx
        img = anisotropic_stretch(img, sx=sx, sy=sy)
        img = add_streak_and_bias(img)

    img_u8 = match_real_tone(img, tone)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    return img_bgr, y

# Dataset generation

def generate_dataset(
    out_dir: Path,
    real_refs: List[Path],
    num_classes: int = NUM_CLASSES,
    images_per_class: int = IMAGES_PER_CLASS,
    seed: int = 42,
) -> None:
    if num_classes < 1 or num_classes > len(GEN_BY_CLASS):
        raise ValueError(f"num_classes must be between 1 and {len(GEN_BY_CLASS)}.")
    if images_per_class < 1:
        raise ValueError("images_per_class must be at least 1.")

    np.random.seed(seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    for cls in range(num_classes):
        (out_dir / f"class_{cls}").mkdir(exist_ok=True)

    real_tone = compute_real_tone_stats(real_refs)
    labels_csv = out_dir / "labels.csv"

    print("[INFO] Real SEM tone statistics:", real_tone)
    print("[INFO] Output directory:", out_dir)
    print("[INFO] Images per class:", images_per_class)
    print("[INFO] Classes:", num_classes)

    with labels_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path"] + ELEMENTS)

        for cls in range(num_classes):
            for i in tqdm(range(images_per_class), desc=f"class_{cls}"):
                img_bgr, y = render_one(cls, real_tone)

                out_path = out_dir / f"class_{cls}" / f"sim_{cls}_{i:04d}.png"
                ok = cv2.imwrite(str(out_path), img_bgr)
                if not ok:
                    raise RuntimeError(f"Failed to write image: {out_path}")

                writer.writerow([str(out_path)] + [float(v) for v in y])

    print("[INFO] Generation finished")
    print(f"[INFO] Labels saved to: {labels_csv}")

# CLI

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic SEM-like images for composition-estimation experiments."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="sim_images",
        help="Output directory for generated images and labels.csv.",
    )
    parser.add_argument(
        "--real-ref",
        type=str,
        nargs="+",
        required=True,
        help="Real SEM reference images used for percentile-based tone matching.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=NUM_CLASSES,
        help=f"Number of morphology classes to generate. Max: {NUM_CLASSES}.",
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=IMAGES_PER_CLASS,
        help="Number of synthetic images per morphology class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    real_refs = [Path(p) for p in args.real_ref]
    generate_dataset(
        out_dir=Path(args.out_dir),
        real_refs=real_refs,
        num_classes=args.num_classes,
        images_per_class=args.images_per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
