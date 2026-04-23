"""Phase 2 deep learning pipeline: EfficientNet-B3 + CBAM for skin-condition classification."""

from importlib import import_module

__all__ = [
    "EfficientNetB3CBAM",
    "Phase2Config",
    "generate_gradcam_overlays",
    "run_phase2_pipeline",
]


def __getattr__(name: str):
    if name == "EfficientNetB3CBAM":
        return import_module(".model", __name__).EfficientNetB3CBAM
    if name == "Phase2Config":
        return import_module(".config", __name__).Phase2Config
    if name == "generate_gradcam_overlays":
        return import_module(".gradcam", __name__).generate_gradcam_overlays
    if name == "run_phase2_pipeline":
        return import_module(".run_phase2", __name__).run_phase2_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
