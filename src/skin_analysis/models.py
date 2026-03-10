from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_model(model_name: str, random_state: int = 42) -> Pipeline | RandomForestClassifier:
    """Create one of the supported classical ML baseline models."""
    normalized_name = model_name.strip().lower()

    if normalized_name == "svm":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if normalized_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported model name: {model_name}")


def train_model(model_name: str, X_train, y_train, random_state: int = 42):
    """Fit one of the supported models and return the trained estimator."""
    model = build_model(model_name=model_name, random_state=random_state)
    model.fit(X_train, y_train)
    return model
