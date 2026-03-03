"""Reusable Phase 1 utilities for skin-condition classification."""

from importlib import import_module

__all__ = ["load_data", "extract_features", "train_model", "evaluate_model", "run_phase1_pipeline"]


def __getattr__(name: str):
    if name == "load_data":
        return import_module(".data", __name__).load_data
    if name == "extract_features":
        return import_module(".features", __name__).extract_features
    if name == "train_model":
        return import_module(".models", __name__).train_model
    if name == "evaluate_model":
        return import_module(".evaluate", __name__).evaluate_model
    if name == "run_phase1_pipeline":
        return import_module(".pipeline", __name__).run_phase1_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
