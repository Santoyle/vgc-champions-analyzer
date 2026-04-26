"""
Página de predicciones WP (Win Probability) con SHAP feature importance
y calculadora de matchup interactiva.

Requiere un modelo entrenado en models/wp/reg={id}/current/.
Si no existe, muestra instrucciones para entrenarlo.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.app.modules.wp import (
    ALL_TYPES,
    ReplayFeatures,
    features_to_dataframe,
    get_feature_names,
)
from src.app.modules.wp_train import (
    AUC_GATE,
    BRIER_GATE,
    load_model,
)
from src.app.modules.ga import (
    Chromosome,
    chromosome_to_dict,
    _load_pokemon_master,
)
from src.app.modules.ga_nsga2 import (
    run_nsga2,
    GAResult,
)
from src.app.modules.ga_warmstart import (
    build_warm_start_population,
    WarmStartConfig,
)
from src.app.modules.ga_blending import (
    blended_fitness,
    count_replays_for_regulation,
    compute_lambda,
)
from src.app.utils.session import init_session

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_MODELS_DIR = _PROJECT_ROOT / "models" / "wp"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_model_metadata(regulation_id: str) -> dict[str, Any] | None:
    """
    Lee metadata.json del modelo current para una regulación.

    Args:
        regulation_id: ID de la regulación.

    Returns:
        Dict con metadata del modelo. None si no existe o hay error.
    """
    import json

    metadata_path = (
        _MODELS_DIR / f"reg={regulation_id}" / "current" / "metadata.json"
    )
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text())  # type: ignore[no-any-return]
    except Exception:  # noqa: BLE001
        return None


def build_demo_features(regulation_id: str) -> ReplayFeatures:
    """
    Construye un ReplayFeatures de demo para mostrar el SHAP plot sin datos reales.

    Representa un matchup típico de Champions:
    Incineroar + Garchomp (p1) vs Sneasler + Sinistcha (p2).

    Args:
        regulation_id: Para etiquetar las features.

    Returns:
        ReplayFeatures con valores representativos de Champions.
    """
    n_types = len(ALL_TYPES)
    p1_coverage = [0.0] * n_types
    p2_coverage = [0.0] * n_types

    fire_idx = ALL_TYPES.index("Fire")
    dark_idx = ALL_TYPES.index("Dark")
    dragon_idx = ALL_TYPES.index("Dragon")
    ground_idx = ALL_TYPES.index("Ground")
    fight_idx = ALL_TYPES.index("Fighting")
    poison_idx = ALL_TYPES.index("Poison")
    grass_idx = ALL_TYPES.index("Grass")
    ghost_idx = ALL_TYPES.index("Ghost")

    p1_coverage[fire_idx] = 0.5
    p1_coverage[dark_idx] = 0.5
    p1_coverage[dragon_idx] = 0.5
    p1_coverage[ground_idx] = 0.5

    p2_coverage[fight_idx] = 0.5
    p2_coverage[poison_idx] = 0.5
    p2_coverage[grass_idx] = 0.5
    p2_coverage[ghost_idx] = 0.5

    return ReplayFeatures(
        replay_id="demo",
        regulation_id=regulation_id,
        label=None,
        rating_norm=0.5,
        month_idx=0,
        p1_type_coverage=p1_coverage,
        p2_type_coverage=p2_coverage,
        p1_has_fake_out=1.0,
        p2_has_fake_out=0.0,
        p1_has_trick_room=0.0,
        p2_has_trick_room=0.0,
        p1_has_intimidate=1.0,
        p2_has_intimidate=0.0,
        p1_has_redirection=0.0,
        p2_has_redirection=1.0,
        p1_has_tailwind=0.0,
        p2_has_tailwind=1.0,
        p1_speed_control=0.0,
        p2_speed_control=1.0,
        p1_team_size=6.0,
        p2_team_size=6.0,
    )


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------

init_session()
reg_id: str = st.session_state["selected_reg_id"]

st.title("Predictions")
st.caption(
    f"Regulación: **{reg_id}** · Modelo Win Probability + SHAP"
)

st.divider()

# --- Cargar metadata del modelo ---

metadata = get_model_metadata(reg_id)

if metadata is None:
    st.warning(
        f"No hay modelo WP entrenado para **{reg_id}**.\n\n"
        f"Para entrenar el modelo:\n"
        f"```\n"
        f"python scripts/train_wp.py --reg {reg_id}\n"
        f"```\n"
        f"O con datos sintéticos para probar:\n"
        f"```\n"
        f"python scripts/train_wp.py --synthetic-data\n"
        f"```"
    )
    st.info(
        f"El modelo requiere AUC >= {AUC_GATE} "
        f"y Brier <= {BRIER_GATE} para ser promovido a producción."
    )
    st.stop()

# --- Banner de métricas del modelo actual ---

st.subheader("Modelo en producción")

col1, col2, col3, col4 = st.columns(4)
with col1:
    auc_val = float(metadata.get("auc", 0))
    delta_auc = (
        f"+{auc_val - AUC_GATE:.3f}"
        if auc_val >= AUC_GATE
        else f"{auc_val - AUC_GATE:.3f}"
    )
    st.metric("AUC-ROC", f"{auc_val:.4f}", delta=delta_auc)
with col2:
    brier_val = float(metadata.get("brier", 1))
    delta_brier = (
        f"-{BRIER_GATE - brier_val:.3f}"
        if brier_val <= BRIER_GATE
        else f"+{brier_val - BRIER_GATE:.3f}"
    )
    st.metric(
        "Brier Score",
        f"{brier_val:.4f}",
        delta=delta_brier,
        delta_color="inverse",
    )
with col3:
    st.metric("Muestras train", metadata.get("n_train", "—"))
with col4:
    trained_at = str(metadata.get("trained_at", "—"))[:10]
    st.metric("Entrenado", trained_at)

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_pareto, tab_shap, tab_matchup = st.tabs([
    "🧬 Pareto GA",
    "🧠 SHAP Feature Importance",
    "⚔️ Matchup Calculator",
])

# ── Tab Pareto GA ─────────────────────────────────────────────────────────

with tab_pareto:
    st.subheader("Generador de equipos NSGA-II")
    st.caption(
        "El GA optimiza simultáneamente 4 objetivos: "
        "cobertura defensiva, sinergia ofensiva, "
        "anti-meta y control de velocidad."
    )

    # Métricas de contexto del blending
    reg_config_obj = st.session_state.get("regulation_config")
    n_replays = count_replays_for_regulation(reg_id)
    lam = compute_lambda(n_replays)

    bl_col1, bl_col2, bl_col3 = st.columns(3)
    with bl_col1:
        st.metric("Replays disponibles", n_replays)
    with bl_col2:
        st.metric(
            "λ (prior weight)",
            f"{lam:.3f}",
            help=(
                "Alto λ = confiar en datos históricos. "
                "Bajo λ = confiar en datos actuales."
            ),
        )
    with bl_col3:
        st.metric("Data weight", f"{1 - lam:.3f}")

    st.divider()

    # Controles del GA
    with st.expander("⚙️ Configuración del GA", expanded=False):
        ga_col1, ga_col2 = st.columns(2)
        with ga_col1:
            pop_size = st.slider(
                "Tamaño de población",
                min_value=20,
                max_value=120,
                value=60,
                step=20,
                key="ga_pop_size",
                help="120 = producción, 60 = demo rápida",
            )
            n_gen = st.slider(
                "Generaciones",
                min_value=5,
                max_value=60,
                value=20,
                step=5,
                key="ga_n_gen",
            )
        with ga_col2:
            use_warm_start = st.checkbox(
                "Usar warm-start histórico",
                value=True,
                key="ga_warm_start",
                help=(
                    "Inyecta equipos de regulaciones "
                    "anteriores para acelerar la convergencia."
                ),
            )
            ga_seed = st.number_input(
                "Semilla aleatoria",
                min_value=1,
                max_value=9999,
                value=42,
                key="ga_seed",
            )

    # Botón principal
    run_ga = st.button(
        "🧬 Generar equipos Pareto",
        type="primary",
        key="btn_run_ga",
        use_container_width=True,
    )

    # Estado del resultado GA en session
    if "ga_result" not in st.session_state:
        st.session_state["ga_result"] = None
    if "ga_reg_id" not in st.session_state:
        st.session_state["ga_reg_id"] = None

    # Ejecutar GA
    if run_ga:
        # Limpiar resultado anterior si cambió la regulación
        if st.session_state["ga_reg_id"] != reg_id:
            st.session_state["ga_result"] = None
            st.session_state["ga_reg_id"] = reg_id

        pm = _load_pokemon_master()

        # Warm-start
        warm_chroms: list[Chromosome] = []
        if use_warm_start:
            with st.spinner("Cargando equipos históricos..."):
                try:
                    ws_config = WarmStartConfig(
                        min_rating=1500,
                        max_teams=100,
                        warm_fraction=0.35,
                    )
                    warm_chroms = build_warm_start_population(
                        reg_config_obj,
                        config=ws_config,
                        pokemon_master=pm,
                    )
                    if warm_chroms:
                        st.toast(
                            f"✅ {len(warm_chroms)} equipos históricos cargados"
                        )
                except Exception as exc:  # noqa: BLE001
                    log.warning("Warm-start falló: %s", exc)

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        def _progress_cb(
            gen: int,
            total: int,
            front: list[Chromosome],
        ) -> None:
            pct = int(gen / total * 100)
            progress_bar.progress(pct)
            status_text.text(
                f"Gen {gen}/{total} — {len(front)} equipos en Pareto"
            )

        with st.spinner(f"Evolucionando {n_gen} generaciones..."):
            try:
                result = run_nsga2(
                    reg=reg_config_obj,
                    pokemon_master=pm,
                    pop_size=pop_size,
                    n_gen=n_gen,
                    seed=int(ga_seed),
                    warm_start_chromosomes=warm_chroms or None,
                    progress_callback=_progress_cb,
                )
                st.session_state["ga_result"] = result
                st.session_state["ga_reg_id"] = reg_id
                progress_bar.progress(100)
                status_text.text(
                    f"✅ Completado: "
                    f"{len(result.pareto_front)} equipos en el frente de Pareto"
                )
                st.success(
                    f"🎯 {len(result.pareto_front)} equipos óptimos generados."
                )
            except Exception as exc:  # noqa: BLE001
                log.error("Error en run_nsga2: %s", exc)
                st.error(
                    f"Error ejecutando GA: {exc}\n\n"
                    "Intenta con menos generaciones "
                    "o verifica que la regulación "
                    "tiene pokemon_legales definidos."
                )

    # Mostrar resultados del Pareto
    ga_result: GAResult | None = st.session_state.get("ga_result")

    if ga_result is None:
        st.info(
            "Haz click en **Generar equipos Pareto** "
            "para iniciar la optimización.\n\n"
            f"Regulación activa: **{reg_id}** · "
            f"{n_replays} replays disponibles · "
            f"λ = {lam:.3f}"
        )
    else:
        st.divider()
        st.subheader(
            f"Frente de Pareto — {len(ga_result.pareto_front)} equipos"
        )

        # Gráfico de convergencia
        if ga_result.logbook:
            import plotly.graph_objects as go

            gens = [d["gen"] for d in ga_result.logbook]
            obj_names = [
                "Cobertura def",
                "Sinergia of",
                "Anti-meta",
                "Speed control",
            ]
            fig_conv = go.Figure()
            for obj_idx, obj_name in enumerate(obj_names):
                vals = []
                for d in ga_result.logbook:
                    max_vals = d.get("max", [0, 0, 0, 0])
                    vals.append(
                        max_vals[obj_idx]
                        if obj_idx < len(max_vals)
                        else 0.0
                    )
                fig_conv.add_trace(
                    go.Scatter(x=gens, y=vals, name=obj_name, mode="lines")
                )
            fig_conv.update_layout(
                title="Convergencia del GA",
                xaxis_title="Generación",
                yaxis_title="Fitness máximo",
                height=300,
            )
            st.plotly_chart(fig_conv, use_container_width=True)

        # Cards de equipos Pareto
        pm_display = _load_pokemon_master()
        for rank, chrom in enumerate(ga_result.pareto_front[:10], start=1):
            chrom_dict = chromosome_to_dict(chrom, pm_display)
            slots = chrom_dict.get("slots", [])

            fit = getattr(chrom, "fitness", None)
            if fit is not None and hasattr(fit, "values") and len(fit.values) >= 4:
                f1, f2, f3, f4 = (
                    fit.values[0],
                    fit.values[1],
                    fit.values[2],
                    fit.values[3],
                )
            else:
                f1 = f2 = f3 = f4 = 0.0

            with st.expander(
                f"🏅 Equipo #{rank} — "
                f"f1:{f1:.2f} f2:{f2:.2f} f3:{f3:.2f} f4:{f4:.2f}",
                expanded=(rank == 1),
            ):
                import plotly.graph_objects as go

                radar_fig = go.Figure(
                    go.Scatterpolar(
                        r=[f1, f2, f3, f4, f1],
                        theta=[
                            "Def Coverage",
                            "Off Synergy",
                            "Anti-Meta",
                            "Speed Control",
                            "Def Coverage",
                        ],
                        fill="toself",
                        name=f"Equipo #{rank}",
                    )
                )
                radar_fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    height=300,
                    showlegend=False,
                )
                st.plotly_chart(radar_fig, use_container_width=True)

                # Pokémon del equipo
                n_cols = min(6, len(slots)) if slots else 1
                pkm_cols = st.columns(n_cols)
                for i, slot in enumerate(slots[:6]):
                    with pkm_cols[i % n_cols]:
                        name = slot.get(
                            "species_name",
                            f"#{slot.get('species_id', '?')}",
                        )
                        item = slot.get("item", "")
                        st.markdown(f"**{name}**")
                        if item:
                            st.caption(f"@ {item}")

                # Blended fitness si hay replays disponibles
                if n_replays > 0:
                    try:
                        bf = blended_fitness(
                            chrom,
                            n_replays=n_replays,
                            pokemon_master=pm_display,
                        )
                        st.caption(
                            f"Blended (λ={bf.lambda_used:.3f}): "
                            f"f1={bf.f1:.3f} "
                            f"f2={bf.f2:.3f} "
                            f"f3={bf.f3:.3f} "
                            f"f4={bf.f4:.3f}"
                        )
                    except Exception:  # noqa: BLE001
                        pass


# ── Tab SHAP ──────────────────────────────────────────────────────────────

with tab_shap:
    st.subheader("Importancia de features (SHAP)")
    st.caption(
        "Las features con mayor valor SHAP son las más influyentes "
        "en las predicciones del modelo."
    )

    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        loaded = load_model(reg_id)
        if loaded is None:
            st.warning("No se pudo cargar el modelo.")
            st.stop()

        model, calibrator = loaded

        feature_names = get_feature_names()
        demo_list = [build_demo_features(reg_id) for _ in range(10)]
        df_demo = features_to_dataframe(demo_list)
        X_demo = df_demo[feature_names].values.astype(np.float32)

        with st.spinner("Calculando SHAP values..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_demo)

        fig_shap, _ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X_demo,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        st.pyplot(fig_shap, use_container_width=True)
        plt.close(fig_shap)

        st.caption(f"SHAP values calculados sobre {len(demo_list)} ejemplos de demo.")

        if hasattr(model, "feature_importances_"):
            importance_df = (
                pd.DataFrame({
                    "feature": feature_names,
                    "importance": model.feature_importances_,
                })
                .sort_values("importance", ascending=False)
                .head(15)
            )
            st.dataframe(importance_df, use_container_width=True, hide_index=True)

    except ImportError:
        st.warning(
            "shap o matplotlib no disponibles. "
            "Instala con: pip install shap matplotlib"
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Error calculando SHAP para %s: %s", reg_id, exc)
        st.warning(
            f"Error calculando SHAP: {exc}\n\n"
            "El modelo puede necesitar más datos de entrenamiento."
        )

# ── Tab Matchup Calculator ────────────────────────────────────────────────

with tab_matchup:
    st.subheader("Calculadora de Matchup")
    st.caption(
        "Ingresa dos equipos para calcular la probabilidad de victoria "
        "del equipo 1."
    )

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.write("**Equipo 1 (p1)**")
        team1_pokemon: list[str] = []
        for i in range(6):
            pkm = st.text_input(
                f"Pokemon {i + 1}",
                placeholder="ej: Incineroar",
                key=f"matchup_t1_{i}",
                label_visibility="collapsed",
            )
            if pkm.strip():
                team1_pokemon.append(pkm.strip())

        has_fake_out_t1 = st.checkbox("Tiene Fake Out", key="t1_fake_out")
        has_intimidate_t1 = st.checkbox("Tiene Intimidate", key="t1_intimidate")
        has_tr_t1 = st.checkbox("Tiene Trick Room", key="t1_tr")

    with col_t2:
        st.write("**Equipo 2 (p2)**")
        team2_pokemon: list[str] = []
        for i in range(6):
            pkm = st.text_input(
                f"Pokemon {i + 1}",
                placeholder="ej: Sneasler",
                key=f"matchup_t2_{i}",
                label_visibility="collapsed",
            )
            if pkm.strip():
                team2_pokemon.append(pkm.strip())

        has_fake_out_t2 = st.checkbox("Tiene Fake Out", key="t2_fake_out")
        has_intimidate_t2 = st.checkbox("Tiene Intimidate", key="t2_intimidate")
        has_tr_t2 = st.checkbox("Tiene Trick Room", key="t2_tr")

    if st.button("Calcular Win Probability", type="primary", key="btn_calc_wp"):
        if not team1_pokemon or not team2_pokemon:
            st.warning("Ingresa al menos 1 Pokemon en cada equipo.")
        else:
            n_types = len(ALL_TYPES)
            matchup_feat = ReplayFeatures(
                replay_id="matchup-calc",
                regulation_id=reg_id,
                label=None,
                rating_norm=0.5,
                month_idx=0,
                p1_type_coverage=[0.0] * n_types,
                p2_type_coverage=[0.0] * n_types,
                p1_has_fake_out=float(has_fake_out_t1),
                p2_has_fake_out=float(has_fake_out_t2),
                p1_has_trick_room=float(has_tr_t1),
                p2_has_trick_room=float(has_tr_t2),
                p1_has_intimidate=float(has_intimidate_t1),
                p2_has_intimidate=float(has_intimidate_t2),
                p1_has_redirection=0.0,
                p2_has_redirection=0.0,
                p1_has_tailwind=0.0,
                p2_has_tailwind=0.0,
                p1_speed_control=0.0,
                p2_speed_control=0.0,
                p1_team_size=float(len(team1_pokemon)),
                p2_team_size=float(len(team2_pokemon)),
            )

            try:
                from src.app.modules.wp_train import predict_proba

                probs = predict_proba([matchup_feat], reg_id)

                if probs is not None and len(probs) > 0:
                    wp_p1 = float(probs[0])
                    wp_p2 = 1.0 - wp_p1

                    st.divider()
                    st.subheader("Resultado")

                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        sign1 = "+" if wp_p1 > 0.5 else ""
                        st.metric(
                            "Win Prob Equipo 1",
                            f"{wp_p1 * 100:.1f}%",
                            delta=f"{sign1}{(wp_p1 - 0.5) * 100:.1f}%",
                        )
                    with res_col2:
                        sign2 = "+" if wp_p2 > 0.5 else ""
                        st.metric(
                            "Win Prob Equipo 2",
                            f"{wp_p2 * 100:.1f}%",
                            delta=f"{sign2}{(wp_p2 - 0.5) * 100:.1f}%",
                        )

                    st.progress(wp_p1)
                    if wp_p1 > 0.5:
                        winner_label = "Equipo 1 favorito"
                    elif wp_p2 > 0.5:
                        winner_label = "Equipo 2 favorito"
                    else:
                        winner_label = "Matchup equilibrado"
                    st.caption(
                        f"{winner_label} "
                        f"(WP basada en roles y tamano de equipo)"
                    )
                else:
                    st.error(
                        "No se pudo calcular WP. "
                        "Verifica que el modelo este entrenado."
                    )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Error calculando WP: {exc}")
