# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```powershell
# Run the app
streamlit run streamlit_app.py

# Run all tests
python -m pytest

# Run a single test file
python -m pytest tests/test_metrics_v2.py -v

# Run a single test
python -m pytest tests/test_metrics_v2.py::test_mlwr_basic -v

# Lint + format
ruff check src/ tests/ --fix
ruff format src/ tests/

# Type checking
mypy src/

# Install dev dependencies
pip install -r requirements-dev.txt

# Install only pipeline dependencies (CI ingestion, no ML)
pip install -r requirements-pipeline.txt
```

Pre-commit hooks (ruff lint + format) run automatically on commit. Direct commits to `main` are blocked.

## Architecture

### Data flow

```
scrapers/ → pipelines/ (dlt) → data/raw/reg={id}/source={fuente}/*.parquet
                              ↓
                         dbt models → data/curated/reg={id}/{tabla}/data.parquet
                              ↓
                         DuckDB (vgc.duckdb) — OLAP queries
                         SQLite (app.sqlite)  — user state only
```

**Five data sources:** Limitless VGC, Pikalytics Champions, Pokémon Showdown replays, Smogon Chaos stats, RK9/Pokedata tournament results.

### Regulation system

`RegulationConfig` (`src/app/core/schema.py`) is the single source of truth for every regulation. It's a frozen Pydantic v2 model with a SHA256 checksum. JSON files live in `regulations/*.json`.

`get_active_regulation()` (`src/app/core/regulation_active.py`) resolves which regulation is active via three prioritized scenarios (exact date match → transition window → most recent fallback). It's injected into `st.session_state["selected_reg_id"]` by `init_session()` at app startup.

**Critical rule:** `regulation_id` must never be hardcoded in Python source. All modules receive it as an explicit parameter. The only allowed literals are `regulations/*.json` and `tests/fixtures/`.

### Streamlit layer

`streamlit_app.py` → `init_session()` → `render_regulation_selector()` → pages.

Each page reads `regulation_id = st.session_state["selected_reg_id"]` and passes it down explicitly. `init_session()` must be the first call in every page.

### Module boundaries

| Layer | Directory | Streamlit imports? | DB access? |
|---|---|---|---|
| Domain schemas | `core/` | No | No |
| DB connections + path helpers | `utils/` | Yes (cache decorators) | Yes |
| Reusable UI widgets | `components/` | Yes | No (data via params) |
| ML, metrics, GA | `modules/`, `metrics/` | No | No |
| Scrapers | `data/scrapers/` | No | No |
| Pages | `pages/` | Yes | Via utils/ only |

### Two databases, two purposes

- **DuckDB** (`data/vgc.duckdb`): OLAP. All Parquet queries go through DuckDB. Always filter with `WHERE` before fetching — never load full Parquet to memory. Get connection via `get_duckdb()`.
- **SQLite** (`data/app.sqlite`): OLTP. User state only (saved teams, roster, notes). Get connection via `get_sqlite()`.

Never mix queries between the two engines in the same operation.

### Genetic Algorithm (Team Builder)

`modules/ga.py` defines the canonical chromosome (`Chromosome` + `SlotGene`, 6 slots × 8 genes). The GA system is split across 6 files: `ga.py` (chromosome), `ga_fitness.py` (4-objective NSGA-II fitness), `ga_nsga2.py` (DEAP integration), `ga_blending.py`, `ga_repair.py`, `ga_warmstart.py`.

### Advanced metrics

`metrics/` contains the analytics layer: `mlwr.py` (MLWR/METI), `spdo.py` (SPDO), `lre.py` (LRE), `advanced_metrics.py` (MEIT, TAI, Shapley, cross-reg normalization). All metric functions are decorated with `@st.cache_data(ttl=3600)` and receive `reg_id` as explicit first parameter.

## Key conventions

**Python files:** Always start with `from __future__ import annotations`. Use `logging.getLogger(__name__)` — never `print()` in `src/`.

**Pydantic v2:** Use `model_validate()` / `model_dump()`. Never mix v1/v2 syntax. Use `ConfigDict(frozen=True)` for immutable models.

**Streamlit cache:** `@st.cache_resource` for DB connections and ML models. `@st.cache_data` for DataFrames and serializable objects. `@st.cache_data(ttl=3600)` for live meta queries. Never store ML models in `st.session_state`.

**Hive-partitioned paths:** Raw data lives at `data/raw/reg={id}/source={fuente}/{YYYY-MM-DD}.parquet`. Use `get_parquet_path(reg_id, source, fecha)` from `utils/db.py` — never build paths by hand.

**Tests:** Every module in `src/` has a corresponding `tests/test_{modulo}.py`. Tests that involve `regulation_id` must be `@pytest.mark.parametrize`-d — never hardcode a `regulation_id` literal in a test body. No real HTTP calls in tests; use `respx` or `httpx.MockTransport`.

**Forbidden stack additions:** No FastAPI, Next.js, PostgreSQL, Airflow, torch, or tensorflow. No `alibi-detect` backends other than numpy.

## CI/CD workflows

Six GitHub Actions workflows in `.github/workflows/`:
- `daily-ingest.yml` — scrapes all sources at 06:00 UTC, commits with `[skip ci]`
- `dbt-build.yml` — transforms raw → curated via dbt
- `drift-check.yml` — detects meta drift using alibi-detect
- `retrain-wp-weekly.yml` — retrains the XGBoost win probability model
- `detect-new-reg.yml` — watches for new regulation announcements
- `validate-regulations.yml` — validates checksums in `regulations/*.json`

Pipeline runs use `requirements-pipeline.txt` (not `requirements.txt`) to avoid loading ML dependencies.

## Agenda de tareas pendientes

### Bloque 16 — Cross-reg + V2 Release

**T-138** (LLM) — Crear tests/test_metrics_v3.py
- Tests para normalize_metric_cross_reg
- Función pura, sin DuckDB, solo Series y DataFrames
- Cubrir: zscore media=0, zscore std=1, minmax [0,1],
  ValueError para método inválido, dict vacío,
  raw_value preservado, std=0 retorna zeros,
  soporta cualquier nombre de métrica

**T-139** (MANUAL) — Tag v2.0 + GitHub Release
- git tag -a v2.0 -m "V2 completo: 10 tabs Analytics,
  Champions Calc, SPDO, MLWR, LRE, METI/MEIT, TAI,
  SV Shapley, Cross-Reg, ~464 tests PASS"
- Crear GitHub Release con changelog

### Principio rector
Regulación-agnóstico: nunca hardcodear "M-A" en código.
Siempre recibir reg_id como parámetro explícito.

### Estado de tests
~464 PASS distribuidos en:
- tests/test_champions_calc.py (30)
- tests/test_mlwr.py (15)
- tests/test_spdo.py (26)
- tests/test_lre.py (14)
- tests/test_metrics_v2.py (24)
- tests/ otros módulos (~355 del Bloque 0-14)

### Indianapolis Regionals
Fecha: 29-31 mayo 2026
Acción post-evento: python scripts/parse_replays.py --reg M-A
Luego: python scripts/train_wp.py --reg M-A
