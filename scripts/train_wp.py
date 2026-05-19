"""
CLI de entrenamiento del modelo Win Probability (WP) para VGC.

Uso:
    # Regulación activa (auto-detect):
    python scripts/train_wp.py

    # Regulación específica:
    python scripts/train_wp.py --reg M-A

    # Entrenar sin guardar aunque pase el gate:
    python scripts/train_wp.py --dry-run

    # Datos sintéticos para testing del pipeline:
    python scripts/train_wp.py --synthetic-data

Exit codes:
    0 — el modelo pasó el gate AUC >= 0.65 y Brier <= 0.20.
    1 — el modelo no pasó el gate, falló el entrenamiento,
        o no hay datos disponibles.

El exit code 1 permite que GitHub Actions marque el step como
fallido en CI si el modelo entrenado no tiene calidad suficiente.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.app.core.regulation_active import get_active_regulation
from src.app.modules.wp import (
    ALL_TYPES,
    ReplayFeatures,
    extract_features,
)
from src.app.modules.wp_train import (
    AUC_GATE,
    BRIER_GATE,
    train_wp_model,
)

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_replays_from_parquet(regulation_id: str) -> list[object]:
    """
    Carga replays desde los Parquets de Showdown en
    data/raw/reg={id}/source=showdown/*.parquet.

    Construye objetos duck-typed compatibles con extract_features()
    a partir de los dicts de cada fila del Parquet.

    Args:
        regulation_id: ID de la regulación.

    Returns:
        Lista de objetos con atributos: replay_id, regulation_id, p1, p2,
        winner, rating, upload_time, team_p1, team_p2, raw_log.
        Lista vacía si no hay Parquets o el directorio no existe.
    """
    import json
    from dataclasses import dataclass

    @dataclass
    class ReplayRecord:
        replay_id: str
        regulation_id: str
        p1: str
        p2: str
        winner: str | None
        rating: int
        upload_time: int
        team_p1: list[str]
        team_p2: list[str]
        raw_log: str = ""

    parquet_dir = _DATA_DIR / "raw" / f"reg={regulation_id}" / "source=showdown"

    if not parquet_dir.exists():
        log.warning("No hay directorio de replays: %s", parquet_dir)
        return []

    parquet_files = list(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        log.warning("Sin Parquets en %s", parquet_dir)
        return []

    records: list[object] = []
    for pq_file in parquet_files:
        try:
            df = pd.read_parquet(pq_file)
            for _, row in df.iterrows():
                try:
                    team_p1: list[str] = json.loads(
                        str(row.get("team_p1_json", "[]"))
                    )
                    team_p2: list[str] = json.loads(
                        str(row.get("team_p2_json", "[]"))
                    )
                    winner_val = row.get("winner")
                    records.append(
                        ReplayRecord(
                            replay_id=str(row.get("replay_id", "")),
                            regulation_id=regulation_id,
                            p1=str(row.get("p1", "")),
                            p2=str(row.get("p2", "")),
                            winner=str(winner_val) if winner_val is not None else None,
                            rating=int(row.get("rating", 1500) or 1500),
                            upload_time=int(row.get("upload_time", 0) or 0),
                            team_p1=team_p1,
                            team_p2=team_p2,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    log.debug("Fila inválida en %s: %s", pq_file.name, exc)
        except Exception as exc:  # noqa: BLE001
            log.warning("Error leyendo %s: %s", pq_file, exc)

    log.info(
        "Cargados %d replays desde %d Parquets", len(records), len(parquet_files)
    )
    return records


def _generate_synthetic_features(
    regulation_id: str,
    n: int = 200,
) -> list[ReplayFeatures]:
    """
    Genera features sintéticas con señal para testing del pipeline.

    La señal proviene de que p1 gana más frecuentemente cuando tiene
    rating_norm más alto (correlación débil + ruido 30%). Esto permite
    que XGBoost aprenda algo sin datos reales.

    Args:
        regulation_id: ID para etiquetar las features generadas.
        n: Número de features a generar.

    Returns:
        Lista de ReplayFeatures con label sintético y features aleatorias.
    """
    import random

    rng = random.Random(42)
    n_types = len(ALL_TYPES)
    features: list[ReplayFeatures] = []

    for i in range(n):
        p1_rating = rng.uniform(1500.0, 2000.0)
        p2_rating = rng.uniform(1500.0, 2000.0)
        # Señal: jugador con más rating gana, con 30% de ruido
        label = 1.0 if p1_rating > p2_rating else 0.0
        if rng.random() < 0.3:
            label = 1.0 - label

        features.append(
            ReplayFeatures(
                replay_id=f"synthetic-{i}",
                regulation_id=regulation_id,
                label=label,
                rating_norm=(p1_rating - 1500.0) / 500.0,
                month_idx=rng.randint(0, 3),
                p1_type_coverage=[rng.random() for _ in range(n_types)],
                p2_type_coverage=[rng.random() for _ in range(n_types)],
                p1_has_fake_out=float(rng.random() > 0.5),
                p2_has_fake_out=float(rng.random() > 0.5),
                p1_has_trick_room=float(rng.random() > 0.8),
                p2_has_trick_room=float(rng.random() > 0.8),
                p1_has_intimidate=float(rng.random() > 0.6),
                p2_has_intimidate=float(rng.random() > 0.6),
                p1_has_redirection=float(rng.random() > 0.7),
                p2_has_redirection=float(rng.random() > 0.7),
                p1_has_tailwind=float(rng.random() > 0.7),
                p2_has_tailwind=float(rng.random() > 0.7),
                p1_speed_control=float(rng.random() > 0.6),
                p2_speed_control=float(rng.random() > 0.6),
                p1_team_size=float(rng.randint(4, 6)),
                p2_team_size=float(rng.randint(4, 6)),
            )
        )

    return features


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    """
    Entry point del script de entrenamiento WP.

    Returns:
        0 si el modelo pasó el gate, 1 en cualquier caso de fallo.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Entrenamiento del modelo Win Probability"
    )
    parser.add_argument(
        "--reg",
        default=None,
        help="Regulación a entrenar. Default: auto-detect.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Entrenar sin guardar el modelo aunque pase el gate.",
    )
    parser.add_argument(
        "--synthetic-data",
        action="store_true",
        dest="synthetic",
        help="Usar datos sintéticos para testing. Solo para desarrollo.",
    )
    args = parser.parse_args()

    # Resolver regulación
    reg_id: str = args.reg or ""
    if not reg_id:
        try:
            active = get_active_regulation()
            reg_id = active.regulation_id
            log.info("Regulación activa: %s", reg_id)
        except Exception as exc:  # noqa: BLE001
            log.error("No se pudo detectar regulación activa: %s", exc)
            return 1

    # Cargar o generar features
    features_list: list[ReplayFeatures]
    if args.synthetic:
        log.info("Usando datos sintéticos para testing")
        features_list = _generate_synthetic_features(reg_id, n=200)
    else:
        replays = _load_replays_from_parquet(reg_id)
        if not replays:
            log.error(
                "Sin replays para %s. Ejecuta el pipeline LIVE primero.", reg_id
            )
            return 1

        reg_start = "2026-04" if reg_id == "M-A" else "2025-05"
        features_list = []
        for replay in replays:
            feat = extract_features(replay, regulation_start_month=reg_start)
            if feat is not None:
                features_list.append(feat)

        log.info(
            "%d features extraídas de %d replays", len(features_list), len(replays)
        )

    if len(features_list) < 20:
        log.error(
            "Insuficientes features con label conocido para entrenar: %d",
            len(features_list),
        )
        return 1

    # Entrenar
    try:
        result = train_wp_model(
            features_list,
            regulation_id=reg_id,
            save_if_passes_gate=not args.dry_run,
        )
    except ValueError as exc:
        log.error("Error en entrenamiento: %s", exc)
        return 1

    # Reporte
    gate_str = "PASS" if result.passed_gate else "FAIL"
    saved_str = str(result.model_path) if result.model_path else "NO (dry-run o gate falló)"
    print(
        f"\n{'='*50}\n"
        f"RESULTADO ENTRENAMIENTO WP -- {reg_id}\n"
        f"{'='*50}\n"
        f"AUC:      {result.auc:.4f} (gate >= {AUC_GATE})\n"
        f"Brier:    {result.brier:.4f} (gate <= {BRIER_GATE})\n"
        f"Train:    {result.n_train} muestras\n"
        f"Val:      {result.n_val} muestras\n"
        f"Test:     {result.n_test} muestras\n"
        f"Gate:     {gate_str}\n"
        f"Guardado: {saved_str}\n"
        f"{'='*50}\n"
    )

    if not result.passed_gate:
        log.error(
            "El modelo NO pasó el gate. "
            "AUC=%.4f (necesita >= %.2f), Brier=%.4f (necesita <= %.2f)",
            result.auc,
            AUC_GATE,
            result.brier,
            BRIER_GATE,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
