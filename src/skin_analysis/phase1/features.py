from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def _to_uint8(image_bgr: np.ndarray) -> np.ndarray:
    """Accept normalized float images or uint8 images and convert to uint8 safely."""
    if image_bgr.dtype == np.uint8:
        return image_bgr

    image = np.asarray(image_bgr, dtype=np.float32)
    if image.max() <= 1.0:
        image = image * 255.0
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def extract_color_histogram(image_bgr: np.ndarray, hist_bins: int = 32) -> np.ndarray:
    """
    Extract normalized HSV histograms channel-wise.

    Color histograms help capture redness-sensitive cues such as rosacea and
    pigmentation changes such as dark spots.
    """
    image_uint8 = _to_uint8(image_bgr)
    hsv_image = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2HSV)

    histograms = []
    channel_ranges = [(0, 180), (0, 256), (0, 256)]
    for channel_index, channel_range in enumerate(channel_ranges):
        hist = cv2.calcHist([hsv_image], [channel_index], None, [hist_bins], channel_range)
        hist = cv2.normalize(hist, hist).flatten()
        histograms.append(hist)
    return np.concatenate(histograms).astype(np.float32)


def extract_lbp_histogram(
    image_bgr: np.ndarray,
    lbp_points: int = 24,
    lbp_radius: int = 3,
) -> np.ndarray:
    """
    Extract a normalized LBP histogram from the grayscale image.

    LBP is valuable for texture-heavy classes such as eczema because dry or flaky
    skin patterns are better described by local texture than by color alone.
    """
    image_uint8 = _to_uint8(image_bgr)
    grayscale = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(grayscale, P=lbp_points, R=lbp_radius, method="uniform")
    n_bins = lbp_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-12
    return hist


def extract_glcm_features(
    image_bgr: np.ndarray,
    distances: Iterable[int] = (1, 2),
    angles: Iterable[float] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
) -> np.ndarray:
    """Extract flattened GLCM statistics from the grayscale image."""
    image_uint8 = _to_uint8(image_bgr)
    grayscale = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(
        grayscale,
        distances=list(distances),
        angles=list(angles),
        levels=256,
        symmetric=True,
        normed=True,
    )

    properties = ("contrast", "homogeneity", "energy", "correlation")
    feature_blocks = [graycoprops(glcm, prop).flatten() for prop in properties]
    return np.concatenate(feature_blocks).astype(np.float32)


def extract_features(
    image_bgr: np.ndarray,
    hist_bins: int = 32,
    lbp_points: int = 24,
    lbp_radius: int = 3,
    glcm_distances: Iterable[int] = (1, 2),
    glcm_angles: Iterable[float] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
) -> np.ndarray:
    """Combine color and texture features into one classical ML feature vector."""
    color_features = extract_color_histogram(image_bgr=image_bgr, hist_bins=hist_bins)
    lbp_features = extract_lbp_histogram(
        image_bgr=image_bgr,
        lbp_points=lbp_points,
        lbp_radius=lbp_radius,
    )
    glcm_features = extract_glcm_features(
        image_bgr=image_bgr,
        distances=glcm_distances,
        angles=glcm_angles,
    )
    return np.concatenate([color_features, lbp_features, glcm_features]).astype(np.float32)


def build_feature_matrix(
    images: list[np.ndarray],
    hist_bins: int = 32,
    lbp_points: int = 24,
    lbp_radius: int = 3,
    glcm_distances: Iterable[int] = (1, 2),
    glcm_angles: Iterable[float] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
) -> np.ndarray:
    """Extract features for every image and return a 2D feature matrix."""
    features = [
        extract_features(
            image_bgr=image,
            hist_bins=hist_bins,
            lbp_points=lbp_points,
            lbp_radius=lbp_radius,
            glcm_distances=glcm_distances,
            glcm_angles=glcm_angles,
        )
        for image in images
    ]
    return np.vstack(features).astype(np.float32)
