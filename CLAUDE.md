# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Causal inference analysis of gas turbine CO and NOx emissions. Uses observational data to estimate causal effects (not just correlations) of operating conditions on emissions, leveraging the DoWhy/EconML/CausalML ecosystem.

## Commands

```bash
uv sync                          # install/update dependencies
uv run marimo edit notebooks/01_explore.py   # open a marimo notebook in browser
uv run python <script.py>        # run any Python script
```

## Tech Stack

- **Python 3.11** managed with **uv** (pyproject.toml, uv.lock)
- **Marimo** for interactive notebooks (notebooks are `.py` files, not `.ipynb`)
- **DoWhy** — causal graph specification and identification
- **EconML** — heterogeneous treatment effect estimators (DML, causal forests)
- **CausalML** — uplift modeling, meta-learners (S/T/X-learner)
- **pandas, numpy, scikit-learn** — data wrangling and ML baselines
- **matplotlib, seaborn, graphviz** — visualization

## Data

`data/gas_turbine_emissions.csv` — hourly readings from a gas turbine (2011-2015). Columns:

- **Ambient (exogenous):** AT (temperature °C), AP (pressure mbar), AH (humidity %)
- **Process:** AFDP (filter pressure), GTEP (exhaust pressure), TIT (turbine inlet temp — primary control), TAT (turbine after temp), TEY (energy yield MWh), CDP (compressor discharge pressure)
- **Emissions (outcomes):** CO (mg/m³), NOx (mg/m³)

## Domain Knowledge

Read `docs/domain_knowledge.md` before working on causal models. Key points:

- **CO and NOx have opposite responses to flame temperature** (the CO-NOx tradeoff). They are NOT causally related to each other — they share a common cause (flame temperature).
- **TIT is the primary controllable variable** (set via fuel flow rate). Its effect on emissions is confounded by ambient conditions (AT, AP, AH).
- **Causal graph structure:** Ambient conditions → air density → mass flow → process variables → emissions. AT/AP/AH are the key confounders that must be controlled for.

## Project Structure

```
notebooks/     # marimo notebooks (01_explore.py, ...)
data/          # CSV dataset
docs/          # domain knowledge documentation
```
