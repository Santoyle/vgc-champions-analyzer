# VGC Champions Analyzer

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Herramienta de análisis competitivo para Pokémon VGC Champions. Combina
estadísticas de uso, modelos ML y métricas avanzadas inspiradas en analítica
deportiva profesional. Diseñada para adaptarse automáticamente a cualquier
regulación activa.

---

## Stack

Python 3.12 · Streamlit · DuckDB · DEAP · XGBoost · alibi-detect · dlt

---

## Quickstart (Windows PowerShell)

```powershell
git clone https://github.com/tu-usuario/vgc-champions-analyzer.git
cd vgc-champions-analyzer
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Estructura

```
vgc-champions-analyzer/
├── src/
│   └── app/
│       ├── core/
│       ├── utils/
│       ├── components/
│       ├── modules/
│       ├── pages/
│       └── data/
│           ├── scrapers/
│           ├── pipelines/
│           ├── parsers/
│           └── sql/
├── data/
│   ├── raw/
│   ├── curated/
│   └── snapshots/
├── models/
│   ├── wp/
│   └── ga/
├── regulations/
├── tests/
└── scripts/
```

---

## Regulación activa

> **Regulación actual: Reg M-A** (8 abr – 17 jun 2026)
>
> El sistema detecta y aplica automáticamente la regulación vigente.
> Al cambiar de regulación, solo se actualiza `regulations/{nueva}.json`.

---

## Estado del proyecto

> Proyecto en construcción activa. MVP objetivo:
> **Indianapolis Regionals (29–31 mayo 2026).**
