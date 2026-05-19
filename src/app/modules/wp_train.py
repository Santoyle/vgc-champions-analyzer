"""
Pipeline de entrenamiento del modelo Win Probability (WP) para VGC.

El modelo WP predice la probabilidad de victoria de p1 dado el estado
del equipo y metadatos de la batalla. El pipeline completo es:

  1. Convertir ReplayFeatures a DataFrame (features_to_dataframe).
  2. Split 70/15/15 estratificado por label.
  3. Entrenar XGBoost con early stopping en validation set.
  4. Calibrar probabilidades con IsotonicRegression post-hoc.
  5. Evaluar AUC y Brier score en test set.
  6. Gate de calidad: AUC >= 0.65 Y Brier <= 0.20 (ambas condiciones).
  7. Si pasa el gate: guardar en models/wp/reg={id}/{timestamp}/ y current/.

El gate AUC/Brier previene que modelos con entrenamiento insuficiente
(pocos replays, clases desbalanceadas) lleguen a producción. Un modelo que
no pasa el gate se loggea como advertencia pero no se persiste.

current/ siempre apunta al último modelo que pasó el gate para esa
regulación. Las versiones anteriores quedan disponibles con su timestamp
para auditoría y rollback.

load_model reconstruye IsotonicRegression desde JSON (no pickle) para
evitar problemas de compatibilidad entre versiones de scikit-learn.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

from src.app.modules.wp import (
    ReplayFeatures,
    features_to_dataframe,
    get_feature_names,
)

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_MODELS_DIR = _PROJECT_ROOT / "models" / "wp"

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

AUC_GATE: float = 0.65
BRIER_GATE: float = 0.20
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15
RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    """
    Resultado del entrenamiento del modelo WP.

    Attributes:
        auc: AUC-ROC en test set (mayor es mejor, 0.5 = azar).
        brier: Brier score en test set (menor es mejor, 0 = perfecto).
        log_loss_val: Log-loss en validation set para monitoreo.
        n_train: Muestras usadas en entrenamiento.
        n_val: Muestras usadas en validación (calibración + early stopping).
        n_test: Muestras usadas en evaluación final.
        feature_names: Lista de features usadas en el orden del modelo.
        passed_gate: True si auc >= AUC_GATE y brier <= BRIER_GATE (ambas).
        model_path: Path al directorio versionado si se guardó. None si no pasó el gate.
        trained_at: ISO 8601 timestamp del inicio del entrenamiento.
        regulation_id: Regulación para la que se entrenó el modelo.
    """

    auc: float
    brier: float
    log_loss_val: float
    n_train: int
    n_val: int
    n_test: int
    feature_names: list[str]
    passed_gate: bool
    model_path: Path | None
    trained_at: str
    regulation_id: str


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------


def _split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide el DataFrame en train/val/test con ratios 70/15/15.

    Usa estratificación por label para mantener balance entre victorias (1.0)
    y derrotas (0.0) en cada split.

    Args:
        df: DataFrame con columna "label" (0.0 / 1.0 / NaN).

    Returns:
        Tupla (df_train, df_val, df_test) — solo filas con label conocido.

    Raises:
        ValueError: Si hay menos de 20 muestras con label conocido.
    """
    df_labeled = df[df["label"].notna()].copy()

    if len(df_labeled) < 20:
        raise ValueError(
            f"Insuficientes muestras con label: {len(df_labeled)} (mínimo 20)"
        )

    df_train, df_temp = train_test_split(
        df_labeled,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_STATE,
        stratify=df_labeled["label"],
    )

    val_size_relative = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1.0 - val_size_relative),
        random_state=RANDOM_STATE,
        stratify=df_temp["label"],
    )

    log.info(
        "Split: train=%d val=%d test=%d", len(df_train), len(df_val), len(df_test)
    )
    return df_train, df_val, df_test


def _train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
) -> xgb.XGBClassifier:
    """
    Entrena XGBoost con early stopping en validation set.

    Hiperparámetros fijos para MVP (optimizados para datasets <10k replays):
      n_estimators=500, max_depth=4, learning_rate=0.05,
      subsample=0.8, colsample_bytree=0.8, min_child_weight=5.
    scale_pos_weight se calcula automáticamente para balancear clases.

    Args:
        X_train: Matrix de features de entrenamiento (float32).
        y_train: Labels de entrenamiento (float32, 0 o 1).
        X_val: Matrix de features de validación.
        y_val: Labels de validación.
        feature_names: Nombres de las features para el modelo.

    Returns:
        XGBClassifier entrenado con el mejor número de iteraciones.
    """
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=RANDOM_STATE,
        verbosity=0,
        feature_names=feature_names,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    log.info("XGBoost entrenado: best_iteration=%d", model.best_iteration)
    return model


def _calibrate_isotonic(
    model: xgb.XGBClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> IsotonicRegression:
    """
    Calibra las probabilidades del modelo XGBoost usando IsotonicRegression.

    La calibración isotónica post-hoc ajusta las probabilidades predichas para
    que reflejen frecuencias observadas. Es más flexible que Platt scaling para
    distribuciones no-Gaussianas como las de VGC.

    Args:
        model: XGBClassifier ya entrenado.
        X_val: Features de validación (usadas para ajustar el calibrador).
        y_val: Labels verdaderos de validación.

    Returns:
        IsotonicRegression ajustado a las predicciones del modelo en val set.
    """
    raw_probs: np.ndarray = model.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probs, y_val)
    log.info(
        "Calibración isotónica ajustada sobre %d muestras de validación", len(y_val)
    )
    return calibrator


def _evaluate(
    model: xgb.XGBClassifier,
    calibrator: IsotonicRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float]:
    """
    Evalúa el modelo calibrado en el test set.

    Args:
        model: XGBClassifier entrenado.
        calibrator: IsotonicRegression ajustado.
        X_test: Features de test (nunca vistas durante entrenamiento).
        y_test: Labels verdaderos de test.

    Returns:
        Tupla (auc, brier) calculados sobre probabilidades calibradas.
    """
    raw_probs: np.ndarray = model.predict_proba(X_test)[:, 1]
    cal_probs: np.ndarray = calibrator.predict(raw_probs)

    auc = float(roc_auc_score(y_test, cal_probs))
    brier = float(brier_score_loss(y_test, cal_probs))

    log.info("Evaluación test: AUC=%.4f Brier=%.4f", auc, brier)
    return auc, brier


def _save_model(
    model: xgb.XGBClassifier,
    calibrator: IsotonicRegression,
    regulation_id: str,
    metadata: dict[str, Any],
) -> Path:
    """
    Guarda el modelo y su metadata en el directorio versionado y en current/.

    Estructura creada:
        models/wp/reg={reg_id}/{YYYY-MM-DDTHHMMSS}/
            model.xgb          — modelo XGBoost en formato binario nativo
            calibrator.json    — IsotonicRegression serializado como JSON
            metadata.json      — métricas, config y timestamp
        models/wp/reg={reg_id}/current/
            model.xgb          — copia del modelo de esta versión
            calibrator.json    — copia del calibrador de esta versión
            metadata.json      — copia de los metadatos de esta versión

    current/ se sobreescribe siempre, apuntando al último modelo que pasó el gate.

    Args:
        model: XGBClassifier a persistir.
        calibrator: IsotonicRegression a persistir (como JSON, no pickle).
        regulation_id: ID de la regulación.
        metadata: Dict con métricas, feature_names y configuración.

    Returns:
        Path al directorio versionado recién creado.
    """
    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    reg_dir = _MODELS_DIR / f"reg={regulation_id}"
    version_dir = reg_dir / timestamp
    current_dir = reg_dir / "current"

    version_dir.mkdir(parents=True, exist_ok=True)
    current_dir.mkdir(parents=True, exist_ok=True)

    # Guardar XGBoost en formato binario nativo (no pickle)
    model_path = version_dir / "model.xgb"
    model.save_model(str(model_path))

    # Serializar IsotonicRegression como JSON (compatible entre versiones sklearn)
    calibrator_path = version_dir / "calibrator.json"
    calibrator_data: dict[str, Any] = {
        "X_thresholds": calibrator.X_thresholds_.tolist(),
        "y_thresholds": calibrator.y_thresholds_.tolist(),
        "out_of_bounds": "clip",
    }
    calibrator_path.write_text(json.dumps(calibrator_data, indent=2), encoding="utf-8")

    # Guardar metadata
    metadata_path = version_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Sobreescribir current/ con esta versión
    (current_dir / "model.xgb").write_bytes(model_path.read_bytes())
    (current_dir / "calibrator.json").write_text(
        calibrator_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (current_dir / "metadata.json").write_text(
        metadata_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    log.info("Modelo guardado en %s y current/", version_dir)
    return version_dir


# ---------------------------------------------------------------------------
# Funciones públicas
# ---------------------------------------------------------------------------


def train_wp_model(
    features_list: list[ReplayFeatures],
    regulation_id: str,
    save_if_passes_gate: bool = True,
) -> TrainResult:
    """
    Ejecuta el pipeline completo de entrenamiento del modelo WP.

    Pipeline:
      1. Convertir features a DataFrame.
      2. Split 70/15/15 estratificado por label.
      3. Entrenar XGBoost con early stopping.
      4. Calibrar con IsotonicRegression.
      5. Evaluar AUC y Brier en test set.
      6. Gate: AUC >= 0.65 Y Brier <= 0.20 (ambas condiciones).
      7. Si pasa el gate y save_if_passes_gate=True: persistir modelo.

    Args:
        features_list: Lista de ReplayFeatures extraídas de replays.
        regulation_id: Regulación para la que se entrena.
        save_if_passes_gate: Si True, persiste el modelo cuando pasa el gate.

    Returns:
        TrainResult con métricas, resultado del gate y path del modelo.

    Raises:
        ValueError: Si features_list está vacío o hay menos de 20 muestras
                    con label conocido.
    """
    trained_at = datetime.now().isoformat()
    feature_names = get_feature_names()

    df = features_to_dataframe(features_list)
    if df.empty:
        raise ValueError("features_list vacío — sin datos para entrenar")

    df_train, df_val, df_test = _split_data(df)

    X_train = df_train[feature_names].values.astype(np.float32)
    y_train = df_train["label"].values.astype(np.float32)
    X_val = df_val[feature_names].values.astype(np.float32)
    y_val = df_val["label"].values.astype(np.float32)
    X_test = df_test[feature_names].values.astype(np.float32)
    y_test = df_test["label"].values.astype(np.float32)

    model = _train_xgboost(X_train, y_train, X_val, y_val, feature_names)
    calibrator = _calibrate_isotonic(model, X_val, y_val)
    auc, brier = _evaluate(model, calibrator, X_test, y_test)

    ll_val: float
    try:
        raw_val: np.ndarray = model.predict_proba(X_val)[:, 1]
        cal_val: np.ndarray = calibrator.predict(raw_val)
        ll_val = float(log_loss(y_val, cal_val))
    except Exception:  # noqa: BLE001
        ll_val = float("nan")

    # Gate: ambas condiciones deben cumplirse
    passed_gate = auc >= AUC_GATE and brier <= BRIER_GATE

    log.info(
        "Gate: AUC=%.4f (>=%.2f) Brier=%.4f (<=%.2f) → %s",
        auc,
        AUC_GATE,
        brier,
        BRIER_GATE,
        "PASS" if passed_gate else "FAIL",
    )

    metadata: dict[str, Any] = {
        "regulation_id": regulation_id,
        "auc": auc,
        "brier": brier,
        "log_loss_val": ll_val,
        "n_train": len(df_train),
        "n_val": len(df_val),
        "n_test": len(df_test),
        "feature_names": feature_names,
        "passed_gate": passed_gate,
        "trained_at": trained_at,
        "auc_gate": AUC_GATE,
        "brier_gate": BRIER_GATE,
    }

    model_path: Path | None = None
    if passed_gate and save_if_passes_gate:
        model_path = _save_model(model, calibrator, regulation_id, metadata)
    elif not passed_gate:
        log.warning(
            "Modelo NO guardado — no pasó el gate. AUC=%.4f Brier=%.4f", auc, brier
        )

    return TrainResult(
        auc=auc,
        brier=brier,
        log_loss_val=ll_val,
        n_train=len(df_train),
        n_val=len(df_val),
        n_test=len(df_test),
        feature_names=feature_names,
        passed_gate=passed_gate,
        model_path=model_path,
        trained_at=trained_at,
        regulation_id=regulation_id,
    )


def load_model(
    regulation_id: str,
    version: str = "current",
) -> tuple[xgb.XGBClassifier, IsotonicRegression] | None:
    """
    Carga el modelo WP desde disco.

    Reconstruye IsotonicRegression desde JSON para evitar problemas de
    compatibilidad con pickle entre versiones de scikit-learn.

    Args:
        regulation_id: ID de la regulación.
        version: "current" para el último modelo que pasó el gate, o un
                 timestamp específico en formato YYYY-MM-DDTHHMMSS.

    Returns:
        Tupla (XGBClassifier, IsotonicRegression) si existe el modelo.
        None si el directorio o los archivos no existen.
    """
    model_dir = _MODELS_DIR / f"reg={regulation_id}" / version
    model_path = model_dir / "model.xgb"
    calibrator_path = model_dir / "calibrator.json"

    if not model_path.exists():
        log.info("No hay modelo en %s", model_dir)
        return None

    try:
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))

        cal_data: dict[str, Any] = json.loads(
            calibrator_path.read_text(encoding="utf-8")
        )
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.X_thresholds_ = np.array(cal_data["X_thresholds"], dtype=np.float64)
        calibrator.y_thresholds_ = np.array(cal_data["y_thresholds"], dtype=np.float64)

        log.info("Modelo cargado desde %s", model_dir)
        return model, calibrator

    except Exception as exc:  # noqa: BLE001
        log.error("Error cargando modelo desde %s: %s", model_dir, exc)
        return None


def predict_proba(
    features_list: list[ReplayFeatures],
    regulation_id: str,
    version: str = "current",
) -> np.ndarray | None:
    """
    Predice la probabilidad de victoria de p1 para una lista de replays.

    Carga el modelo desde disco, extrae las features del DataFrame y aplica
    las probabilidades calibradas.

    Args:
        features_list: Lista de ReplayFeatures a predecir.
        regulation_id: ID de la regulación del modelo a cargar.
        version: Versión del modelo ("current" o timestamp).

    Returns:
        Array numpy de float en [0, 1] con probabilidades de victoria de p1.
        None si no hay modelo disponible o features_list está vacío.
    """
    loaded = load_model(regulation_id, version)
    if loaded is None:
        log.warning("No hay modelo WP para %s/%s", regulation_id, version)
        return None

    model, calibrator = loaded
    feature_names = get_feature_names()
    df = features_to_dataframe(features_list)

    if df.empty:
        return None

    X = df[feature_names].values.astype(np.float32)
    raw_probs: np.ndarray = model.predict_proba(X)[:, 1]
    cal_probs: np.ndarray = calibrator.predict(raw_probs)
    return cal_probs


__all__ = [
    "TrainResult",
    "train_wp_model",
    "load_model",
    "predict_proba",
    "AUC_GATE",
    "BRIER_GATE",
]
