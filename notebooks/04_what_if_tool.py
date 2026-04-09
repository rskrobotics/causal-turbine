import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def intro():
    import marimo as mo

    mo.md(
        """
        # 04 — Interactive "What If?" Tool

        This is the payoff of causal inference: a tool that answers **causal questions**.

        In notebooks 01-03 we:
        1. Explored the data (correlations, distributions)
        2. Built a causal graph from domain knowledge
        3. Used the backdoor criterion to identify what to control for
        4. Estimated heterogeneous treatment effects with CausalForestDML

        Now we have a fitted causal model. This notebook demonstrates **two levels**
        of causal reasoning from Pearl's "Ladder of Causation":

        | Rung | Name | Question | What It Uses |
        |------|------|----------|--------------|
        | **2** | Intervention | "If I **set** TIT to X under conditions Y, what happens?" | Covariate settings only |
        | **3** | Counterfactual | "For **this specific event** where I observed outcome Z, what **would have** happened if TIT were different?" | Covariates **+ observed outcome** |

        **The key insight:** Rung 3 is strictly more informative. By conditioning on the
        observed outcome, we capture unit-specific factors (unmeasured confounders, noise)
        that Rung 2 averages over. This lets us give **personalized** counterfactual answers.
        """
    )
    return (mo,)


@app.cell
def load_and_fit():
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from econml.dml import CausalForestDML
    from sklearn.ensemble import GradientBoostingRegressor

    # Load data
    df = pd.read_csv(Path(__file__).parent / "../data/gas_turbine_emissions.csv")

    # Fit causal models (same as notebook 03)
    Y_nox = df["NOX"].values
    Y_co = df["CO"].values
    T = df["TIT"].values
    X = df[["AT"]].values
    W = df[["AT", "AP", "AH"]].values

    # NOX model
    cf_nox = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
        model_t=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
        n_estimators=200,
        min_samples_leaf=20,
        random_state=42,
    )
    cf_nox.fit(Y_nox, T, X=X, W=W)

    # CO model
    cf_co = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
        model_t=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
        n_estimators=200,
        min_samples_leaf=20,
        random_state=42,
    )
    cf_co.fit(Y_co, T, X=X, W=W)

    # Get data ranges for UI bounds
    data_ranges = {
        "AT": (df["AT"].min(), df["AT"].max()),
        "TIT": (df["TIT"].min(), df["TIT"].max()),
        "NOX": (df["NOX"].min(), df["NOX"].max()),
        "CO": (df["CO"].min(), df["CO"].max()),
    }

    return cf_co, cf_nox, data_ranges, df, np


@app.cell
def rung2_header(mo):
    mo.md("""
    ---

    # Rung 2: Interventional Queries

    *"If I set TIT to X under conditions Y, what happens to emissions?"*

    This answers questions about **generic interventions**. You specify the ambient
    conditions (AT) and ask what happens if you change TIT. The model returns the
    **average effect** for units with those covariates.

    **What this ignores:** Unit-specific factors. Two turbine-hours with identical
    AT might have different baseline emissions due to unmeasured factors (fuel batch,
    maintenance state, grid load). Rung 2 averages over this heterogeneity.
    """)
    return


@app.cell
def ui_controls(data_ranges, mo):
    mo.md("## Set the Intervention Conditions")

    at_slider = mo.ui.slider(
        start=int(data_ranges["AT"][0]),
        stop=int(data_ranges["AT"][1]),
        value=15,
        label="Ambient Temperature (AT)",
        show_value=True,
    )

    tit_baseline = mo.ui.slider(
        start=int(data_ranges["TIT"][0]),
        stop=int(data_ranges["TIT"][1]),
        value=1060,
        label="Baseline TIT (current)",
        show_value=True,
    )

    tit_target = mo.ui.slider(
        start=int(data_ranges["TIT"][0]),
        stop=int(data_ranges["TIT"][1]),
        value=1080,
        label="Target TIT (intervention)",
        show_value=True,
    )

    return at_slider, tit_baseline, tit_target


@app.cell
def display_controls(at_slider, mo, tit_baseline, tit_target):
    controls = mo.vstack([
        mo.md("### Operating Conditions"),
        mo.hstack([at_slider], justify="start"),
        mo.md("### TIT Intervention"),
        mo.hstack([tit_baseline, tit_target], justify="start"),
        mo.md(f"**Intervention:** TIT {tit_baseline.value}°C → {tit_target.value}°C "
              f"(Δ = {tit_target.value - tit_baseline.value:+d}°C)"),
    ])
    return (controls,)


@app.cell
def compute_effects(at_slider, cf_co, cf_nox, np, tit_baseline, tit_target):
    # Compute causal effects at the selected conditions
    X_query = np.array([[at_slider.value]])
    r2_T0 = tit_baseline.value
    r2_T1 = tit_target.value

    # Point estimates
    r2_delta_nox = cf_nox.effect(X=X_query, T0=r2_T0, T1=r2_T1)[0]
    r2_delta_co = cf_co.effect(X=X_query, T0=r2_T0, T1=r2_T1)[0]

    # Confidence intervals
    r2_nox_ci = cf_nox.effect_interval(X=X_query, T0=r2_T0, T1=r2_T1, alpha=0.05)
    r2_co_ci = cf_co.effect_interval(X=X_query, T0=r2_T0, T1=r2_T1, alpha=0.05)

    return r2_T0, r2_T1, r2_co_ci, r2_delta_co, r2_delta_nox, r2_nox_ci


@app.cell
def display_results(r2_T0, r2_T1, at_slider, r2_co_ci, r2_delta_co, r2_delta_nox, mo, r2_nox_ci):
    def format_effect(val, ci_lo, ci_hi, unit="mg/m³"):
        color = "#ff6b6b" if val > 0 else "#4a9eff"
        direction = "increases" if val > 0 else "decreases"
        return f'<span style="color: {color}; font-weight: bold">{val:+.2f} {unit}</span> ({direction}), 95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]'

    nox_result = format_effect(r2_delta_nox, r2_nox_ci[0][0], r2_nox_ci[1][0])
    co_result = format_effect(r2_delta_co, r2_co_ci[0][0], r2_co_ci[1][0])

    results = mo.md(
        f"""
        ## Causal Effects

        **If you change TIT from {r2_T0}°C to {r2_T1}°C on a day with AT = {at_slider.value}°C:**

        | Emission | Predicted Change | Interpretation |
        |----------|-----------------|----------------|
        | **NOX** | {nox_result} | {"More thermal NOX from hotter flame" if r2_delta_nox > 0 else "Less NOX — cooler effective flame"} |
        | **CO** | {co_result} | {"Less complete combustion" if r2_delta_co > 0 else "More complete combustion"} |

        ---

        **Remember:** These are *causal* estimates, not predictions. They answer:
        "What would happen if we *intervened* to set TIT to {r2_T1}°C, holding all else equal?"

        This accounts for confounding by ambient conditions (AT, AP, AH) using the
        backdoor adjustment from notebook 02.
        """
    )
    return (results,)


@app.cell
def rung3_header(mo):
    mo.md("""
    ---

    # Rung 3: Counterfactual Queries

    *"For this specific event where I observed Y, what would have happened if TIT were different?"*

    This is the highest rung of causal reasoning. Instead of asking about generic units
    with certain covariates, we ask about **a specific observed event**.

    **The key formula:**

    ```
    Y_counterfactual = Y_observed + τ(X) × (T_new - T_observed)
    ```

    Where τ(X) is the causal effect estimate from our model. This works because:

    1. **Y_observed captures everything about this unit** — measured covariates, unmeasured
       factors, idiosyncratic noise. It's our "abduction" step.

    2. **The causal effect τ(X) tells us how Y changes with T** — this is what we estimated
       with the backdoor adjustment.

    3. **We apply the effect to the observed baseline** — not to some population average.

    **Why this matters:** If a particular turbine-hour had unusually high NOX (maybe due to
    a fuel batch issue), the counterfactual preserves that. Rung 2 would average it away.
    """)
    return


@app.cell
def rung3_event_selector(df, mo):
    # Create a sample of interesting events to choose from
    # We'll pick events that span the AT range and have varying emission levels
    sample_indices = [0, 1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000]
    sample_indices = [i for i in sample_indices if i < len(df)]

    # Build options dict: label -> index
    options = {f"Row {i}: AT={df.iloc[i]['AT']:.1f}°C, TIT={df.iloc[i]['TIT']:.0f}°C, NOX={df.iloc[i]['NOX']:.1f}, CO={df.iloc[i]['CO']:.2f}": i
               for i in sample_indices}

    # Default to third option (index 5000) - need the label string, not the index
    default_idx = sample_indices[2] if len(sample_indices) > 2 else sample_indices[0]
    default_label = [k for k, v in options.items() if v == default_idx][0]

    event_dropdown = mo.ui.dropdown(
        options=options,
        value=default_label,
        label="Select a specific observed event",
    )

    tit_cf = mo.ui.slider(
        start=1000,
        stop=1100,
        value=1080,
        label="Counterfactual TIT (what if TIT had been...)",
        show_value=True,
    )

    return event_dropdown, tit_cf


@app.cell
def rung3_display_selector(event_dropdown, mo, tit_cf):
    mo.vstack([
        mo.md("## Select an Observed Event"),
        event_dropdown,
        mo.md("## Set Counterfactual Treatment"),
        tit_cf,
    ])
    return


@app.cell
def rung3_compute(cf_co, cf_nox, df, event_dropdown, np, tit_cf):
    # Get the selected event
    r3_idx = event_dropdown.value
    r3_event = df.iloc[r3_idx]

    # Observed values
    r3_T_obs = r3_event["TIT"]
    r3_Y_nox_obs = r3_event["NOX"]
    r3_Y_co_obs = r3_event["CO"]
    r3_X_event = np.array([[r3_event["AT"]]])

    # Counterfactual treatment
    r3_T_cf = tit_cf.value

    # Compute causal effects (this is τ(X) × (T_cf - T_obs))
    r3_delta_nox = cf_nox.effect(X=r3_X_event, T0=r3_T_obs, T1=r3_T_cf)[0]
    r3_delta_co = cf_co.effect(X=r3_X_event, T0=r3_T_obs, T1=r3_T_cf)[0]

    # Counterfactual outcomes: Y_cf = Y_obs + effect
    r3_Y_nox_cf = r3_Y_nox_obs + r3_delta_nox
    r3_Y_co_cf = r3_Y_co_obs + r3_delta_co

    # Also compute Rung 2 answer for comparison (what we'd predict without observing Y)
    # For Rung 2, we'd use population baseline, but let's show the effect only
    r3_nox_ci = cf_nox.effect_interval(X=r3_X_event, T0=r3_T_obs, T1=r3_T_cf, alpha=0.05)
    r3_co_ci = cf_co.effect_interval(X=r3_X_event, T0=r3_T_obs, T1=r3_T_cf, alpha=0.05)

    return (r3_T_cf, r3_T_obs, r3_X_event, r3_Y_co_cf, r3_Y_co_obs, r3_Y_nox_cf, r3_Y_nox_obs,
            r3_co_ci, r3_delta_co, r3_delta_nox, r3_event, r3_idx, r3_nox_ci)


@app.cell
def rung3_results(r3_T_cf, r3_T_obs, r3_Y_co_cf, r3_Y_co_obs, r3_Y_nox_cf, r3_Y_nox_obs,
                  r3_co_ci, r3_delta_co, r3_delta_nox, r3_event, r3_idx, mo, r3_nox_ci):
    def format_change(delta, ci_lo, ci_hi):
        color = "#ff6b6b" if delta > 0 else "#4a9eff"
        arrow = "↑" if delta > 0 else "↓"
        return f'<span style="color: {color}; font-weight: bold">{delta:+.2f} {arrow}</span> (95% CI: [{ci_lo:.2f}, {ci_hi:.2f}])'

    nox_change = format_change(r3_delta_nox, r3_nox_ci[0][0], r3_nox_ci[1][0])
    co_change = format_change(r3_delta_co, r3_co_ci[0][0], r3_co_ci[1][0])

    mo.md(f"""
    ## Counterfactual Results for Event #{r3_idx}

    ### Observed Event (What Actually Happened)

    | Variable | Value |
    |----------|-------|
    | Ambient Temp (AT) | {r3_event['AT']:.1f}°C |
    | Ambient Pressure (AP) | {r3_event['AP']:.1f} mbar |
    | Ambient Humidity (AH) | {r3_event['AH']:.1f}% |
    | **TIT (actual)** | **{r3_T_obs:.0f}°C** |
    | **NOX (observed)** | **{r3_Y_nox_obs:.2f} mg/m³** |
    | **CO (observed)** | **{r3_Y_co_obs:.2f} mg/m³** |

    ### Counterfactual: What If TIT Had Been {r3_T_cf}°C?

    | Emission | Observed | Counterfactual | Change |
    |----------|----------|----------------|--------|
    | **NOX** | {r3_Y_nox_obs:.2f} mg/m³ | **{r3_Y_nox_cf:.2f} mg/m³** | {nox_change} |
    | **CO** | {r3_Y_co_obs:.2f} mg/m³ | **{r3_Y_co_cf:.2f} mg/m³** | {co_change} |

    ---

    **This is a Rung 3 answer.** We're not asking "what happens on average for AT={r3_event['AT']:.1f}°C?"
    We're asking "what would have happened **for this specific turbine-hour**, given that
    we actually observed these emission levels?"

    The counterfactual preserves whatever made this unit special — if it was a high-NOX
    hour for unobserved reasons, the counterfactual reflects that.
    """)
    return


@app.cell
def rung3_comparison(df, r3_event, mo):
    # Show how Rung 3 differs from Rung 2 by comparing to other events at similar AT
    at_val = r3_event["AT"]
    similar_events = df[(df["AT"] > at_val - 1) & (df["AT"] < at_val + 1)]

    if len(similar_events) > 1:
        nox_range = f"{similar_events['NOX'].min():.1f} - {similar_events['NOX'].max():.1f}"
        co_range = f"{similar_events['CO'].min():.2f} - {similar_events['CO'].max():.2f}"
        n_similar = len(similar_events)
    else:
        nox_range = "N/A"
        co_range = "N/A"
        n_similar = 0

    mo.md(f"""
    ### Why Rung 3 Matters: Heterogeneity at Similar Conditions

    There are **{n_similar} events** in the dataset with AT within ±1°C of this event.
    Their emission ranges:

    - NOX: {nox_range} mg/m³
    - CO: {co_range} mg/m³

    **Rung 2** would give the same answer for all of them (same effect estimate).

    **Rung 3** gives different counterfactuals because each event has a different
    observed baseline — we're personalizing to the specific event.

    This is analogous to personalized medicine: two patients with the same measured
    covariates might respond differently to treatment. If we observe their baseline
    outcome, we can give a more informed counterfactual.
    """)
    return


@app.cell
def effect_curve_section(mo):
    mo.md("""
    ---

    ## How the Effect Varies by Temperature

    The chart below shows the causal effect of a +20°C TIT increase across all
    ambient temperatures. Your current selection is marked.

    Key insight from notebook 03: **the effect flips sign around 15-17°C**.
    On cold days, higher TIT reduces NOX. On hot days, it increases NOX.
    """)
    return


@app.cell
def effect_curve_plot(at_slider, cf_co, cf_nox, np, tit_baseline, tit_target):
    def _():
        import matplotlib.pyplot as plt

        # Compute effect curves across AT range
        at_grid = np.linspace(-5, 37, 100).reshape(-1, 1)
        T0 = tit_baseline.value
        T1 = tit_target.value

        effects_nox = cf_nox.effect(X=at_grid, T0=T0, T1=T1)
        effects_co = cf_co.effect(X=at_grid, T0=T0, T1=T1)

        nox_ci_lo, nox_ci_hi = cf_nox.effect_interval(X=at_grid, T0=T0, T1=T1, alpha=0.05)
        co_ci_lo, co_ci_hi = cf_co.effect_interval(X=at_grid, T0=T0, T1=T1, alpha=0.05)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # NOX panel
        axes[0].plot(at_grid, effects_nox, "b-", lw=2, label="NOX effect")
        axes[0].fill_between(at_grid.ravel(), nox_ci_lo.ravel(), nox_ci_hi.ravel(),
                             alpha=0.2, color="blue")
        axes[0].axhline(y=0, color="black", lw=1, ls="--")
        axes[0].axvline(x=at_slider.value, color="red", lw=2, ls="-", alpha=0.7)
        axes[0].scatter([at_slider.value],
                        [cf_nox.effect(X=np.array([[at_slider.value]]), T0=T0, T1=T1)[0]],
                        color="red", s=100, zorder=5, label=f"Your selection (AT={at_slider.value}°C)")
        axes[0].set_xlabel("Ambient Temperature (°C)")
        axes[0].set_ylabel(f"NOX change (mg/m³)")
        axes[0].set_title(f"NOX Effect: TIT {T0}→{T1}°C")
        axes[0].legend(loc="upper left")

        # CO panel
        axes[1].plot(at_grid, effects_co, "g-", lw=2, label="CO effect")
        axes[1].fill_between(at_grid.ravel(), co_ci_lo.ravel(), co_ci_hi.ravel(),
                             alpha=0.2, color="green")
        axes[1].axhline(y=0, color="black", lw=1, ls="--")
        axes[1].axvline(x=at_slider.value, color="red", lw=2, ls="-", alpha=0.7)
        axes[1].scatter([at_slider.value],
                        [cf_co.effect(X=np.array([[at_slider.value]]), T0=T0, T1=T1)[0]],
                        color="red", s=100, zorder=5, label=f"Your selection (AT={at_slider.value}°C)")
        axes[1].set_xlabel("Ambient Temperature (°C)")
        axes[1].set_ylabel(f"CO change (mg/m³)")
        axes[1].set_title(f"CO Effect: TIT {T0}→{T1}°C")
        axes[1].legend(loc="upper right")

        plt.tight_layout()
        plt.show()

    _()
    return


@app.cell
def tradeoff_section(mo):
    mo.md("""
    ## The CO-NOX Tradeoff Zone

    One of the key findings: **on cold days, there is no tradeoff**. Higher TIT
    reduces both CO and NOX. On hot days, you face the classic tradeoff —
    reducing CO means increasing NOX.

    The chart below maps this out across AT and TIT combinations.
    """)
    return


@app.cell
def tradeoff_heatmap(cf_co, cf_nox, np):
    def _():
        import matplotlib.pyplot as plt

        # Create a grid of (AT, TIT change) combinations
        at_values = np.linspace(-5, 35, 20)
        tit_changes = np.linspace(-30, 30, 20)  # change from baseline of 1070

        nox_grid = np.zeros((len(tit_changes), len(at_values)))
        co_grid = np.zeros((len(tit_changes), len(at_values)))

        baseline_tit = 1070

        for i, delta_tit in enumerate(tit_changes):
            for j, at in enumerate(at_values):
                X_q = np.array([[at]])
                nox_grid[i, j] = cf_nox.effect(X=X_q, T0=baseline_tit, T1=baseline_tit + delta_tit)[0]
                co_grid[i, j] = cf_co.effect(X=X_q, T0=baseline_tit, T1=baseline_tit + delta_tit)[0]

        # Classify each cell: win-win, tradeoff, lose-lose
        # win-win: NOX down, CO down (both negative)
        # tradeoff: one up, one down
        # lose-lose: both up (both positive)
        classification = np.zeros_like(nox_grid)
        classification[(nox_grid < 0) & (co_grid < 0)] = 1   # win-win (green)
        classification[(nox_grid > 0) & (co_grid < 0)] = 2   # NOX up, CO down (yellow)
        classification[(nox_grid < 0) & (co_grid > 0)] = 3   # NOX down, CO up (orange)
        classification[(nox_grid > 0) & (co_grid > 0)] = 4   # lose-lose (red)

        fig, axes = plt.subplots(1, 3, figsize=(17, 5))

        # NOX heatmap
        im1 = axes[0].imshow(nox_grid, aspect="auto", origin="lower",
                             extent=[at_values[0], at_values[-1], tit_changes[0], tit_changes[-1]],
                             cmap="RdBu_r", vmin=-5, vmax=5)
        axes[0].set_xlabel("Ambient Temperature (°C)")
        axes[0].set_ylabel("TIT change from 1070°C")
        axes[0].set_title("NOX Change (mg/m³)")
        axes[0].axhline(y=0, color="black", lw=1, ls="--", alpha=0.5)
        plt.colorbar(im1, ax=axes[0])

        # CO heatmap
        im2 = axes[1].imshow(co_grid, aspect="auto", origin="lower",
                             extent=[at_values[0], at_values[-1], tit_changes[0], tit_changes[-1]],
                             cmap="RdBu_r", vmin=-2, vmax=2)
        axes[1].set_xlabel("Ambient Temperature (°C)")
        axes[1].set_ylabel("TIT change from 1070°C")
        axes[1].set_title("CO Change (mg/m³)")
        axes[1].axhline(y=0, color="black", lw=1, ls="--", alpha=0.5)
        plt.colorbar(im2, ax=axes[1])

        # Classification map
        from matplotlib.colors import ListedColormap
        cmap_class = ListedColormap(["white", "#4CAF50", "#FFC107", "#FF9800", "#F44336"])
        im3 = axes[2].imshow(classification, aspect="auto", origin="lower",
                             extent=[at_values[0], at_values[-1], tit_changes[0], tit_changes[-1]],
                             cmap=cmap_class, vmin=0, vmax=4)
        axes[2].set_xlabel("Ambient Temperature (°C)")
        axes[2].set_ylabel("TIT change from 1070°C")
        axes[2].set_title("Operating Zones")
        axes[2].axhline(y=0, color="black", lw=1, ls="--", alpha=0.5)

        # Legend for classification
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#4CAF50", label="Win-Win (both ↓)"),
            Patch(facecolor="#FFC107", label="Tradeoff: NOX↑, CO↓"),
            Patch(facecolor="#FF9800", label="Tradeoff: NOX↓, CO↑"),
            Patch(facecolor="#F44336", label="Lose-Lose (both ↑)"),
        ]
        axes[2].legend(handles=legend_elements, loc="lower right", fontsize=8)

        plt.tight_layout()
        plt.show()

    _()
    return


@app.cell
def interpretation_guide(mo):
    mo.md("""
    ## How to Read These Results

    ### The Key Insight: Context Matters

    A single "effect of TIT on emissions" doesn't exist. The effect depends on
    ambient conditions. This is **heterogeneous treatment effects** in action.

    | Condition | Strategy | Why |
    |-----------|----------|-----|
    | **Cold day (AT < 10°C)** | Run hotter (↑ TIT) | Free lunch — both emissions decrease |
    | **Mild day (AT 10-20°C)** | Depends | Near the crossover — small effects either way |
    | **Hot day (AT > 25°C)** | Accept tradeoff | Higher TIT → lower CO but higher NOX |

    ### Rung 2 vs Rung 3: When to Use Which

    | Scenario | Use | Why |
    |----------|-----|-----|
    | Planning future operations | **Rung 2** | No specific event to condition on yet |
    | Evaluating a past decision | **Rung 3** | We observed the actual outcome |
    | Policy analysis / average effects | **Rung 2** | Interested in population-level effects |

    ### Limitations

    1. **Extrapolation:** The model was trained on TIT 1000-1100°C. Queries
       far outside this range are less reliable.

    2. **Unmeasured confounders:** We assumed AT, AP, AH are the only confounders.
       If grid demand or fuel composition also matter, estimates may be biased.

    3. **Local linearity:** The causal forest assumes the effect is locally linear
       in TIT. For large TIT changes, the true nonlinear effect may differ.

    4. **Observational data:** We can't prove causality — only make causal assumptions
       explicit and test them with refutation (notebook 03).

    5. **Rung 3 assumes stable unit effects:** The counterfactual formula assumes
       the causal effect τ(X) applies uniformly. If the unit-specific deviation
       is itself treatment-dependent, this may not hold.
    """)
    return


@app.cell
def practical_recommendations(mo):
    mo.md("""
    ## Practical Takeaways

    If this were a real operational tool, here's what an operator might do:

    1. **Monitor ambient temperature** — it determines which regime you're in

    2. **On cold days (AT < 10°C):**
       - Don't worry about the NOX-CO tradeoff
       - Running hotter improves *both* emissions
       - Focus on efficiency and power output

    3. **On hot days (AT > 25°C):**
       - The classic tradeoff applies
       - Balance NOX and CO based on regulatory limits
       - Consider demand-response if NOX limits are binding

    4. **Near the crossover (~15°C):**
       - Effects are small — other factors may dominate
       - Monitor both emissions and adjust TIT based on what matters more

    ---

    **This is what causal inference enables:** not just predicting emissions,
    but understanding *what you can do about them* under different conditions.
    """)
    return


@app.cell
def summary(mo):
    mo.md("""
    ## Series Summary

    We've built a complete causal inference pipeline:

    | Notebook | What We Did | Key Tool |
    |----------|-------------|----------|
    | **01** | Explored data, saw correlations | pandas, matplotlib |
    | **02** | Built causal graph, found backdoor paths | DoWhy, networkx |
    | **03** | Estimated heterogeneous effects | EconML CausalForestDML |
    | **04** | Interactive what-if tool with Rung 2 & 3 queries | marimo UI |

    **Key lessons:**

    1. **Correlation ≠ causation** — we need causal graphs and adjustment sets

    2. **Average effects hide heterogeneity** — the TIT-NOX relationship flips
       sign depending on temperature

    3. **Rung 2 ≠ Rung 3** — interventional queries (Rung 2) ask about generic
       effects; counterfactual queries (Rung 3) ask about specific events and
       preserve unit-specific information

    4. **Rung 3 enables personalization** — by conditioning on observed outcomes,
       we can give answers tailored to the specific state of the system

    The ladder of causation isn't just philosophy — it has practical implications
    for how precisely we can answer "what if?" questions.
    """)
    return


if __name__ == "__main__":
    app.run()
