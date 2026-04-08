import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def intro():
    import marimo as mo

    mo.md(
        """
        # 04 — The "What If?" Machine

        We've built up the full causal pipeline across three notebooks:

        ```
        01: Explore the data
        02: Build a causal graph → identify what to control for (estimand)
        03: Estimate effects with DML + causal forests → refute and validate
        ```

        This notebook is the **payoff**. We take the fitted causal model and turn
        it into an interactive tool that answers interventional questions:

        > "Given today's weather, if I set TIT to X instead of Y,
        >  what happens to NOX and CO?"

        These are **causal** answers — not predictions from correlations. They
        tell you what would happen if you reached in and turned the dial,
        holding everything else at its natural value.

        ### Pearl's Ladder of Causation

        | Rung | Question | Example | We used |
        |------|----------|---------|---------|
        | 1. Association | "What do I see?" | "NOX is high when TIT is low" | Notebook 01 (EDA) |
        | 2. Intervention | "What if I do X?" | "If I *set* TIT to 1080, what happens to NOX?" | **This notebook** |
        | 3. Counterfactual | "What if I had done X instead?" | "This reading had TIT=1060. What *would* NOX have been at 1080?" | Notebook 03 |

        Rung 1 is what standard ML does. Rungs 2 and 3 require a causal model.
        This notebook lives on Rung 2 — the rung that matters for decision-making.
        """
    )
    return (mo,)


@app.cell
def load_and_fit(mo):
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from econml.dml import CausalForestDML
    from sklearn.ensemble import GradientBoostingRegressor

    df = pd.read_csv(Path(__file__).parent / "../data/gas_turbine_emissions.csv")

    mo.md(
        """
        ## Fitting the Causal Models

        We refit the causal forests from notebook 03. Same setup:

        - **Y** — outcome (NOX or CO)
        - **T** — treatment (TIT — the lever we control)
        - **W** — confounders to strip out (AT, AP, AH — backdoor variables)
        - **X** — effect modifiers (AT — because the effect varies by temperature)

        Two models: one for NOX, one for CO. Each takes about 30 seconds.
        """
    )

    T = df["TIT"].values
    X = df[["AT"]].values
    W = df[["AT", "AP", "AH"]].values

    # --- NOX model ---
    cf_nox = CausalForestDML(
        model_y=GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=42,
        ),
        model_t=GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=42,
        ),
        n_estimators=200,
        min_samples_leaf=20,
        random_state=42,
    )
    cf_nox.fit(df["NOX"].values, T, X=X, W=W)

    # --- CO model ---
    cf_co = CausalForestDML(
        model_y=GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=42,
        ),
        model_t=GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=42,
        ),
        n_estimators=200,
        min_samples_leaf=20,
        random_state=42,
    )
    cf_co.fit(df["CO"].values, T, X=X, W=W)
    return cf_co, cf_nox, df, np


@app.cell
def intervention_explainer(mo):
    mo.md("""
    ## What Does "Intervention" Mean Here?

    When you move the TIT slider below, you're asking:

    > "If an operator **set** TIT to this value — regardless of what the
    > control system would normally choose — what would happen to emissions?"

    This is different from a prediction. A prediction asks: "when we *observe*
    TIT at 1080, what NOX do we expect?" But that mixes in confounding — days
    where TIT happens to be 1080 might be systematically different (hotter,
    higher demand).

    An intervention strips that out. The causal forest has already removed
    the confounders (AT, AP, AH) during estimation, so the effect is clean.

    **The slider for AT isn't an intervention** — you can't control the weather.
    It's asking: "on a day *like this* (with this AT), what's the effect of
    changing TIT?" AT modifies the effect but isn't being intervened on.
    """)
    return


@app.cell
def controls(mo):
    at_slider = mo.ui.slider(
        start=-6, stop=37, step=1, value=15,
        label="Ambient Temperature (°C)",
    )
    tit_baseline = mo.ui.slider(
        start=1010, stop=1100, step=5, value=1060,
        label="TIT Baseline (°C) — current setting",
    )
    tit_intervention = mo.ui.slider(
        start=1010, stop=1100, step=5, value=1080,
        label="TIT Intervention (°C) — what you'd change it to",
    )

    mo.md(
        f"""
        ## Interactive Controls

        Set the conditions and see what happens. The **baseline** is where TIT
        is now. The **intervention** is where you'd move it.

        {at_slider}

        {tit_baseline}

        {tit_intervention}
        """
    )
    return at_slider, tit_baseline, tit_intervention


@app.cell
def intervention_result(
    at_slider,
    cf_co,
    cf_nox,
    mo,
    np,
    tit_baseline,
    tit_intervention,
):
    at_val = at_slider.value
    t0 = tit_baseline.value
    t1 = tit_intervention.value

    X_point = np.array([[at_val]])

    # Estimate the causal effect of changing TIT from t0 to t1
    delta_nox = cf_nox.effect(X=X_point, T0=t0, T1=t1)[0]
    delta_co = cf_co.effect(X=X_point, T0=t0, T1=t1)[0]

    # Confidence intervals
    nox_ci_lo, nox_ci_hi = cf_nox.effect_interval(X=X_point, T0=t0, T1=t1, alpha=0.05)
    co_ci_lo, co_ci_hi = cf_co.effect_interval(X=X_point, T0=t0, T1=t1, alpha=0.05)

    # Per-degree effect (for interpretation)
    tit_diff = t1 - t0
    if tit_diff != 0:
        per_deg_nox = delta_nox / tit_diff
        per_deg_co = delta_co / tit_diff
    else:
        per_deg_nox = 0
        per_deg_co = 0

    direction_nox = "increases" if delta_nox > 0 else "decreases"
    direction_co = "increases" if delta_co > 0 else "decreases"
    sign_nox = "+" if delta_nox > 0 else ""
    sign_co = "+" if delta_co > 0 else ""

    mo.md(
        f"""
        ## Result: TIT {t0}°C → {t1}°C at AT = {at_val}°C

        | Emission | Change | 95% CI | Per °C TIT |
        |----------|--------|--------|------------|
        | **NOX** | **{sign_nox}{delta_nox:.2f}** mg/m³ | [{nox_ci_lo[0]:.2f}, {nox_ci_hi[0]:.2f}] | {per_deg_nox:+.4f} mg/m³/°C |
        | **CO** | **{sign_co}{delta_co:.2f}** mg/m³ | [{co_ci_lo[0]:.2f}, {co_ci_hi[0]:.2f}] | {per_deg_co:+.4f} mg/m³/°C |

        **NOX** {direction_nox} by **{abs(delta_nox):.2f}** mg/m³.
        **CO** {direction_co} by **{abs(delta_co):.2f}** mg/m³.

        {"**Free lunch!** Both emissions improve." if delta_nox <= 0 and delta_co <= 0 and tit_diff != 0 else ""}{"**Tradeoff:** CO improves but NOX gets worse." if delta_nox > 0 and delta_co < 0 else ""}{"**No change** — baseline equals intervention." if tit_diff == 0 else ""}
        """
    )
    return at_val, t0


@app.cell
def sweep_plot(at_val, cf_co, cf_nox, mo, np, t0):
    def _():
        import matplotlib.pyplot as plt

        # Sweep TIT from baseline to see the full effect curve
        tit_range = np.linspace(1010, 1100, 50)
        X_point = np.full((50, 1), at_val)

        effects_nox = cf_nox.effect(X=X_point, T0=t0, T1=tit_range)
        effects_co = cf_co.effect(X=X_point, T0=t0, T1=tit_range)
        nox_lo, nox_hi = cf_nox.effect_interval(X=X_point, T0=t0, T1=tit_range, alpha=0.05)
        co_lo, co_hi = cf_co.effect_interval(X=X_point, T0=t0, T1=tit_range, alpha=0.05)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # NOX sweep
        axes[0].plot(tit_range, effects_nox, "b-", lw=2)
        axes[0].fill_between(tit_range, nox_lo.ravel(), nox_hi.ravel(),
                             alpha=0.2, color="blue")
        axes[0].axhline(y=0, color="black", lw=1, ls="--")
        axes[0].axvline(x=t0, color="red", ls=":", alpha=0.7, label=f"Baseline TIT={t0}°C")
        axes[0].set_xlabel("Intervention TIT (°C)")
        axes[0].set_ylabel("NOX change (mg/m³)")
        axes[0].set_title(f"NOX Change vs TIT Intervention\n(AT = {at_val}°C, baseline = {t0}°C)")
        axes[0].legend()

        # CO sweep
        axes[1].plot(tit_range, effects_co, "g-", lw=2)
        axes[1].fill_between(tit_range, co_lo.ravel(), co_hi.ravel(),
                             alpha=0.2, color="green")
        axes[1].axhline(y=0, color="black", lw=1, ls="--")
        axes[1].axvline(x=t0, color="red", ls=":", alpha=0.7, label=f"Baseline TIT={t0}°C")
        axes[1].set_xlabel("Intervention TIT (°C)")
        axes[1].set_ylabel("CO change (mg/m³)")
        axes[1].set_title(f"CO Change vs TIT Intervention\n(AT = {at_val}°C, baseline = {t0}°C)")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

        return mo.md(
            f"""
            ### Reading These Curves

            The x-axis is "what if TIT were set to this value?" The y-axis is the
            predicted change in emissions relative to the baseline ({t0}°C).

            - **Where the curve crosses zero** = no change from baseline
            - **Slope** = how sensitive the emission is to TIT at this operating point
            - **Blue/green bands** = 95% confidence intervals — wider means less certain

            Move the AT slider above and watch how the NOX curve changes shape.
            On cold days it slopes downward (higher TIT helps). On hot days it
            slopes upward (higher TIT hurts).

            CO always slopes downward — higher TIT always means more complete combustion.
            """
        )

    _()
    return


@app.cell
def tradeoff_explorer_header(mo):
    mo.md("""
    ## The CO-NOX Tradeoff Map

    This is the operational question: **at what ambient temperature does
    increasing TIT stop being free and start costing you NOX?**

    Below is a heatmap. The x-axis is ambient temperature, the y-axis is
    the TIT change (from a 1060°C baseline). Color shows the NOX effect.
    The white contour line marks the boundary where NOX effect = 0.

    Everything below/left of that line is "free" — you improve both CO and NOX.
    Everything above/right is the tradeoff zone.
    """)
    return


@app.cell
def tradeoff_heatmap(cf_nox, mo, np):
    def _():
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        at_range = np.linspace(-5, 37, 80)
        tit_deltas = np.linspace(-40, 40, 80)
        baseline = 1060

        # Build the grid: for each (AT, TIT_delta), get the NOX effect
        nox_grid = np.zeros((len(tit_deltas), len(at_range)))

        for i, delta in enumerate(tit_deltas):
            X_row = at_range.reshape(-1, 1)
            effects = cf_nox.effect(
                X=X_row, T0=baseline, T1=baseline + delta,
            )
            nox_grid[i, :] = effects.ravel()

        fig, ax = plt.subplots(figsize=(12, 7))

        # Center the colormap at zero
        vmax = max(abs(nox_grid.min()), abs(nox_grid.max()))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im = ax.pcolormesh(
            at_range, tit_deltas, nox_grid,
            cmap="RdBu_r", norm=norm, shading="auto",
        )
        plt.colorbar(im, ax=ax, label="NOX change (mg/m³)")

        # Zero contour — the boundary
        ax.contour(at_range, tit_deltas, nox_grid, levels=[0],
                   colors="white", linewidths=2, linestyles="--")

        ax.set_xlabel("Ambient Temperature (°C)", fontsize=12)
        ax.set_ylabel(f"TIT Change from {baseline}°C", fontsize=12)
        ax.set_title("NOX Effect Map: Where Is It Safe to Increase TIT?", fontsize=13)
        ax.axhline(y=0, color="gray", lw=0.5, ls="-")

        # Annotate regions
        ax.text(5, 25, "FREE LUNCH\n↑TIT reduces NOX", color="white",
                fontsize=11, ha="center", fontweight="bold",
                bbox=dict(boxstyle="round", fc="#2166ac", alpha=0.8))
        ax.text(30, 25, "TRADEOFF\n↑TIT increases NOX", color="white",
                fontsize=11, ha="center", fontweight="bold",
                bbox=dict(boxstyle="round", fc="#b2182b", alpha=0.8))

        plt.tight_layout()
        plt.show()

        return mo.md(
            """
            ### How to Read This Map

            **Blue regions:** Increasing TIT *reduces* NOX — safe to run hotter.

            **Red regions:** Increasing TIT *increases* NOX — the tradeoff is real.

            **White dashed line:** The boundary. Left of it, more TIT is free.
            Right of it, you pay in NOX.

            **For an operator or control system:** Check today's AT, look up where
            you are on the map, and decide how aggressively to push TIT.

            This is the kind of decision support that a pure ML model can't give you —
            it would tell you what NOX to *expect* at a given TIT, but not what would
            *happen* if you changed TIT. The causal model answers the intervention
            question directly.
            """
        )

    _()
    return


@app.cell
def dataset_context(df, mo):
    def _():
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Where does our data actually live?
        axes[0].scatter(df["AT"], df["TIT"], alpha=0.05, s=3, color="#888")
        axes[0].set_xlabel("Ambient Temperature (°C)")
        axes[0].set_ylabel("TIT (°C)")
        axes[0].set_title("Where Our Data Lives\n(predictions outside this region are extrapolation)")

        # Histogram of AT
        axes[1].hist(df["AT"], bins=50, edgecolor="black", alpha=0.7, color="#4a9eff")
        axes[1].set_xlabel("Ambient Temperature (°C)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Distribution of AT\n(more data = more reliable estimates)")

        plt.tight_layout()
        plt.show()

        at_q25, at_q75 = df["AT"].quantile(0.25), df["AT"].quantile(0.75)
        tit_q25, tit_q75 = df["TIT"].quantile(0.25), df["TIT"].quantile(0.75)

        return mo.md(
            f"""
            ### A Word of Caution: Where Can We Trust This?

            The causal model is only as good as the data it was trained on.

            - Most data falls in AT = [{at_q25:.0f}, {at_q75:.0f}]°C — estimates here are tightest
            - TIT is mostly between [{tit_q25:.0f}, {tit_q75:.0f}]°C — large TIT changes extrapolate
            - Extreme corners of the heatmap (very cold + big TIT increase) have few data points

            **Rule of thumb:** Trust the estimates where the data is dense.
            Treat the edges as directionally useful but less precise.

            This is a general principle in causal inference: **identification tells you
            what you *can* estimate, but the data tells you where you can estimate it
            *well*.**
            """
        )

    _()
    return


@app.cell
def full_pipeline_summary(mo):
    mo.md("""
    ## The Complete Pipeline — What We Built

    ```
    ┌──────────────────────────────────────────────────────────────────┐
    │  01: EDA                                                        │
    │  "What does the data look like?"                                │
    │  → Found the CO-NOX tradeoff, saw AT dominates NOX             │
    └─────────────────────────┬────────────────────────────────────────┘
                              ↓
    ┌──────────────────────────────────────────────────────────────────┐
    │  02: Causal Graph + Identification                              │
    │  "What causes what? What should we control for?"                │
    │  → Built DAG from physics, found Simpson's Paradox              │
    │  → DoWhy: control for AT, AP (backdoor criterion)               │
    │  → Linear estimation failed (wrong sign — estimation problem)   │
    └─────────────────────────┬────────────────────────────────────────┘
                              ↓
    ┌──────────────────────────────────────────────────────────────────┐
    │  03: DML + Causal Forests                                       │
    │  "What IS the effect, properly estimated?"                      │
    │  → DML strips out confounders with flexible ML                  │
    │  → Causal forest finds heterogeneous effects                    │
    │  → Key finding: TIT effect flips sign by ambient temp           │
    │  → Refutation + sensitivity analysis: finding is robust         │
    └─────────────────────────┬────────────────────────────────────────┘
                              ↓
    ┌──────────────────────────────────────────────────────────────────┐
    │  04: What-If Machine (this notebook)                            │
    │  "What happens if I change TIT?"                                │
    │  → Interactive intervention queries                             │
    │  → CO-NOX tradeoff map                                         │
    │  → Operational decision support                                 │
    └──────────────────────────────────────────────────────────────────┘
    ```

    ### What Causal Inference Gave Us That ML Couldn't

    1. **Correct direction:** ML would say "higher TIT → lower NOX" (the confounded
       correlation). The causal model says "it depends on temperature."

    2. **Intervention answers:** ML predicts what you'll *observe*. The causal model
       predicts what will *happen if you act*.

    3. **Transparent assumptions:** We know exactly what we assumed (the DAG), what
       we controlled for (AT, AP), and what could go wrong (unmeasured confounders).
       A black-box model hides all of this.

    4. **Actionable boundaries:** The tradeoff map gives a concrete AT threshold
       for switching operating strategies. No amount of correlational analysis
       produces this.

    ### What We Assumed (and Can't Prove)

    - The causal graph is correct (no missing edges, no wrong directions)
    - No unmeasured confounders (sensitivity analysis says this is plausible)
    - The data is representative of future operating conditions
    - The causal forest's functional form is adequate

    Every causal claim comes with assumptions. The graph makes them explicit.
    That's the point — not certainty, but transparency about what could be wrong.
    """)
    return


if __name__ == "__main__":
    app.run()
