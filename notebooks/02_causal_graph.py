import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def intro():
    import marimo as mo

    mo.md(
        """
        # 02 — The Causal Graph

        This is where causal inference diverges from traditional ML.

        In ML, you throw features into a model and let it find patterns. In causal inference,
        you start by **drawing a graph** that encodes your assumptions about what causes what.
        This graph determines everything: what to control for, what estimators are valid,
        and what questions you can even answer.

        The graph comes from **domain knowledge** — physics, engineering, common sense —
        not from the data. The data can't tell you the causal structure (many different
        causal graphs produce the same correlations). You have to bring the structure,
        then the data estimates the strength of the effects.

        See `docs/domain_knowledge.md` for the full reasoning behind each edge.
        """
    )
    return (mo,)


@app.cell
def load_data():
    import pandas as pd
    from pathlib import Path

    df = pd.read_csv(Path(__file__).parent / "../data/gas_turbine_emissions.csv")
    return (df,)


@app.cell
def define_graph(mo):
    import networkx as nx

    # Build the causal DAG from domain knowledge.
    # Each edge is: (cause, effect).
    # If you can't justify an edge with physics/engineering, don't add it.
    causal_graph = nx.DiGraph()

    # --- Ambient conditions affect process variables ---
    # Hot/thin air → less mass flow → lower compressor discharge pressure
    causal_graph.add_edge("AT", "CDP")
    causal_graph.add_edge("AP", "CDP")
    causal_graph.add_edge("AH", "CDP")

    # Ambient conditions affect energy yield (less dense air → less power)
    causal_graph.add_edge("AT", "TEY")
    causal_graph.add_edge("AP", "TEY")

    # Humidity increases air filter pressure drop
    causal_graph.add_edge("AH", "AFDP")

    # --- Ambient conditions affect TIT (through operator/control response) ---
    # Hot day → higher demand → operator raises TIT to compensate
    causal_graph.add_edge("AT", "TIT")
    causal_graph.add_edge("AP", "TIT")

    # --- TIT is the primary control — it drives process variables ---
    causal_graph.add_edge("TIT", "TAT")   # hotter inlet → hotter exhaust
    causal_graph.add_edge("TIT", "TEY")   # more energy in → more power out
    causal_graph.add_edge("TIT", "CDP")   # fuel flow affects pressure balance

    # --- AFDP and GTEP affect the thermodynamic cycle ---
    causal_graph.add_edge("AFDP", "CDP")  # restricted intake → changed pressure
    causal_graph.add_edge("GTEP", "TAT")  # backpressure affects exhaust temp
    causal_graph.add_edge("CDP", "TAT")   # compression ratio → expansion → TAT

    # --- Emissions: the outcomes ---
    # TIT → flame temperature → emissions (the key causal pathways)
    causal_graph.add_edge("TIT", "NOX")   # hotter flame → exponentially more NOx
    causal_graph.add_edge("TIT", "CO")    # hotter flame → more complete combustion → less CO

    # Ambient conditions have DIRECT effects on emissions too (not just through TIT)
    # This is critical — AT/AH affect combustion chemistry independently
    causal_graph.add_edge("AT", "NOX")    # air temp affects combustion chemistry
    causal_graph.add_edge("AT", "CO")
    causal_graph.add_edge("AH", "NOX")    # humidity cools flame → less NOx (direct chemical effect)
    causal_graph.add_edge("AH", "CO")
    causal_graph.add_edge("AP", "NOX")    # pressure affects reactant concentration
    causal_graph.add_edge("AP", "CO")

    # AFDP affects emissions through changed airflow
    causal_graph.add_edge("AFDP", "CO")
    causal_graph.add_edge("AFDP", "NOX")

    mo.md(
        """
        ## Building the DAG

        Every edge above is a claim: "X causes Y." These come from thermodynamics
        and combustion chemistry, not from running regressions.

        Key structural features:
        - **AT, AP, AH** are exogenous (nothing in the system causes them — they're weather)
        - **TIT** is the primary control variable (treatment we want to study)
        - **AT, AP, AH** are confounders: they cause both TIT and emissions
        - **TAT, TEY, CDP** are downstream of TIT — mediators/consequences, not confounders
        - **CO and NOX** are joint effects of shared causes — no edge between them
        """
    )
    return (causal_graph,)


@app.cell
def visualize_graph(causal_graph):
    def _():
        import matplotlib.pyplot as plt

        # Color nodes by their role in the causal structure
        node_colors = {
            # Exogenous (weather) — can't be intervened on
            "AT": "#4a9eff", "AP": "#4a9eff", "AH": "#4a9eff",
            # Quasi-exogenous
            "AFDP": "#88bb88", "GTEP": "#88bb88",
            # Treatment (what we want to intervene on)
            "TIT": "#ff6b6b",
            # Mechanical consequences (mediators/descendants of treatment)
            "TAT": "#ffaa44", "TEY": "#ffaa44", "CDP": "#ffaa44",
            # Outcomes
            "CO": "#cc66ff", "NOX": "#cc66ff",
        }

        # Manual layout: left-to-right causal flow
        pos = {
            "AT":   (0, 2),   "AP":  (0, 1),   "AH":   (0, 0),
            "AFDP": (1, -0.5), "GTEP": (1, -1.5),
            "TIT":  (2, 1.5),
            "CDP":  (2, 0),   "TAT": (3, 0),   "TEY":  (3, 1.5),
            "CO":   (4, 2),   "NOX": (4, 0.5),
        }

        fig, ax = plt.subplots(figsize=(14, 7))

        import networkx as nx
        colors = [node_colors[n] for n in causal_graph.nodes()]
        nx.draw(
            causal_graph, pos, ax=ax, with_labels=True,
            node_color=colors, node_size=2000, font_size=11,
            font_weight="bold", edge_color="#888888",
            arrowsize=20, arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
        )

        # Legend
        from matplotlib.patches import Patch
        legend_items = [
            Patch(color="#4a9eff", label="Exogenous (weather)"),
            Patch(color="#88bb88", label="Quasi-exogenous"),
            Patch(color="#ff6b6b", label="Treatment (TIT)"),
            Patch(color="#ffaa44", label="Mediators/consequences"),
            Patch(color="#cc66ff", label="Outcomes (emissions)"),
        ]
        ax.legend(handles=legend_items, loc="lower left", fontsize=10)
        ax.set_title("Causal DAG — Gas Turbine Emissions", fontsize=14)
        plt.tight_layout()
        plt.show()

    _()
    return


@app.cell
def the_puzzle(df, mo):
    mo.md(f"""
    ## The Puzzle: What does TIT do to NOX?

    We built a graph that says TIT → NOX (hotter flame → more NOx, via Zeldovich
    kinetics). Physics says this should be a **positive** effect.

    But before we use any causal tools, let's just look at the raw data and see
    what a naive analysis tells us. This is what you'd get from standard ML.

    **Raw correlation between TIT and NOX: {df['TIT'].corr(df['NOX']):.3f}**

    It's *negative*. Higher TIT is associated with *lower* NOX.
    That contradicts thermodynamics. What's going on?
    """)
    return


@app.cell
def simpsons_paradox(df, mo):
    def _():
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Panel 1: raw scatter
        axes[0].scatter(df["TIT"], df["NOX"], alpha=0.1, s=3, color="#888")
        z = np.polyfit(df["TIT"], df["NOX"], 1)
        x_line = np.linspace(df["TIT"].min(), df["TIT"].max(), 100)
        axes[0].plot(x_line, np.polyval(z, x_line), "r-", lw=2, label=f"slope = {z[0]:.3f}")
        axes[0].set_xlabel("TIT (°C)")
        axes[0].set_ylabel("NOX (mg/m³)")
        axes[0].set_title("Naive: TIT vs NOX")
        axes[0].legend()

        # Panel 2: colored by AT — reveals the confounder
        sc = axes[1].scatter(df["TIT"], df["NOX"], c=df["AT"], cmap="coolwarm", alpha=0.2, s=3)
        axes[1].set_xlabel("TIT (°C)")
        axes[1].set_ylabel("NOX (mg/m³)")
        axes[1].set_title("Same plot, colored by Ambient Temp (AT)")
        plt.colorbar(sc, ax=axes[1], label="AT (°C)")

        # Panel 3: split by cold vs hot days
        q25, q75 = df["AT"].quantile(0.25), df["AT"].quantile(0.75)
        cold = df[df["AT"] < q25]
        hot = df[df["AT"] > q75]

        axes[2].scatter(cold["TIT"], cold["NOX"], alpha=0.15, s=3,
                        label=f"Cold days (AT < {q25:.0f}°C)", color="#4a9eff")
        axes[2].scatter(hot["TIT"], hot["NOX"], alpha=0.15, s=3,
                        label=f"Hot days (AT > {q75:.0f}°C)", color="#ff6b6b")

        # Trend lines within each group
        for subset, color in [(cold, "#4a9eff"), (hot, "#ff6b6b")]:
            z_sub = np.polyfit(subset["TIT"], subset["NOX"], 1)
            x_sub = np.linspace(subset["TIT"].min(), subset["TIT"].max(), 100)
            axes[2].plot(x_sub, np.polyval(z_sub, x_sub), color=color, lw=2)

        axes[2].set_xlabel("TIT (°C)")
        axes[2].set_ylabel("NOX (mg/m³)")
        axes[2].set_title("Simpson's Paradox")
        axes[2].legend()

        plt.tight_layout()
        plt.show()

        return mo.md(
            """
            ### Simpson's Paradox in action

            **Left:** The overall trend is negative — higher TIT, lower NOX. This is wrong.

            **Middle:** Color by AT and the structure appears. Blue dots (cold days) sit
            at high NOX. Red dots (hot days) sit at low NOX. AT is driving NOX far more
            than TIT is.

            **Right:** Split the data by temperature. *Within* cold days and *within* hot
            days, the TIT→NOX relationship is weakly positive (as physics predicts). But
            pool them together and the trend reverses — because AT has such a strong
            negative effect on NOX that it overwhelms TIT.

            **This is confounding.** AT causes both TIT (hot days → operators raise TIT)
            and NOX (hot air → lower density → cooler flame → less NOX). If you don't
            account for AT, you get the wrong sign.
            """
        )

    _()
    return


@app.cell
def backdoor_paths(mo):
    mo.md("""
    ## Backdoor Paths: Why Confounding Happens

    Look at the graph. There are paths from TIT to NOX that go *backwards*
    through a shared cause:

    ```
    TIT  ←  AT  →  NOX     (AT causes both — "backdoor path")
    TIT  ←  AP  →  NOX     (AP causes both — another backdoor path)
    ```

    These paths carry correlation that is NOT causal. When AT goes up:
    - TIT goes up (operators compensate) — so TIT and NOX move together via AT
    - But NOX goes *down* (hot air → less dense → cooler flame)

    This creates a spurious **negative** association between TIT and NOX.

    ### The Backdoor Criterion

    The fix: **block every backdoor path** by controlling for the variables on them.

    If we hold AT constant (compare readings at the *same* ambient temperature
    but *different* TIT), the backdoor path is blocked. Same for AP.

    That's it. The "backdoor criterion" is just: find all the backdoor paths,
    pick variables to block them, and make sure you don't accidentally block
    the real causal path (don't control for mediators like TAT or TEY).
    """)
    return


@app.cell
def why_not_mediators(mo):
    mo.md("""
    ## What NOT to Control For

    It's tempting to throw every variable into the regression. But controlling
    for the wrong variables is just as dangerous as not controlling at all.

    **Mediators (TAT, TEY, CDP):** These are *downstream* of TIT.

    ```
    TIT → TAT → (part of the effect on emissions)
    ```

    If you hold TAT constant, you're blocking part of TIT's real causal effect.
    You'd be asking: "what's the effect of TIT, if TIT had no effect on TAT?"
    — which is a different (and usually less useful) question.

    **Colliders:** If two causes point into the same node, conditioning on that
    node *opens* a spurious path instead of closing one. Our graph doesn't have
    an obvious collider problem, but it's a common trap.

    **Rule of thumb:** Control for variables that are *causes of both* treatment
    and outcome (confounders). Don't control for anything that's a *consequence*
    of treatment.
    """)
    return


@app.cell
def dowhy_identification(causal_graph, df, mo):
    from dowhy import CausalModel

    # Hand the graph to DoWhy and ask: "what should I control for?"
    model_nox = CausalModel(
        data=df,
        treatment="TIT",
        outcome="NOX",
        graph=causal_graph,
    )

    # DoWhy tries multiple identification strategies
    estimand_nox = model_nox.identify_effect(method_name="default")

    mo.md(
        f"""
        ## DoWhy: Automated Identification

        We give DoWhy our graph and ask: "how can I estimate TIT → NOX?"

        It tries **four strategies** and reports which ones are feasible:

        | Strategy | Idea | Result |
        |----------|------|--------|
        | **Backdoor** | Control for common causes of TIT and NOX | Found: control for AT, AP |
        | **IV** (instrumental variable) | Find a variable that affects TIT but not NOX directly | Not found (AT, AP all directly affect NOX too) |
        | **Frontdoor** | Go through a mediator that confounders don't affect | Not found |
        | **General adjustment** | Mathematical generalization | Same as backdoor |

        The backdoor strategy works. DoWhy says: **control for AT and AP**.

        The mathematical expression `d/d[TIT] (E[NOX|AT,AP])` means:
        "the change in expected NOX per unit change in TIT, holding AT and AP constant."

        ### The Unconfoundedness Assumption

        DoWhy also prints a warning: this only works if there are no *unmeasured*
        confounders beyond AT and AP. If some hidden variable (e.g., grid demand)
        causes both TIT and NOX and we didn't include it, our estimate is biased.
        That's a real concern — we'll come back to it with refutation tests.
        """
    )
    return estimand_nox, model_nox


@app.cell
def raw_estimand(estimand_nox, mo):
    mo.md(f"""
    ## DoWhy's Raw Identification Report

    This is what `identify_effect()` returns — the graph's answer to
    "can we estimate TIT → NOX, and how?"

    ```
    {str(estimand_nox)}
    ```
    """)
    return


@app.cell
def naive_regression(df, mo):
    from sklearn.linear_model import LinearRegression

    # Step 1: the WRONG way — ignore confounders
    naive = LinearRegression().fit(df[["TIT"]], df["NOX"])
    naive_coef = naive.coef_[0]

    # Step 2: the RIGHT way — control for backdoor variables
    adjusted = LinearRegression().fit(df[["TIT", "AT", "AP"]], df["NOX"])
    adj_tit = adjusted.coef_[0]
    adj_at = adjusted.coef_[1]

    mo.md(
        f"""
        ## Naive vs. Adjusted: Seeing the Bias

        | Approach | TIT coefficient | Interpretation |
        |----------|----------------|----------------|
        | **Naive** (NOX ~ TIT) | {naive_coef:+.4f} | Wrong sign! Confounding. |
        | **Adjusted** (NOX ~ TIT + AT + AP) | {adj_tit:+.4f} | Still negative... |

        Both give a negative coefficient. Controlling for AT and AP *reduces* the
        bias (the coefficient moves toward zero), but it's still negative.

        For comparison, the AT coefficient is **{adj_at:+.4f}** — ten times larger
        than TIT's. Ambient temperature dominates this system.

        **So was our physics wrong?** Not necessarily. There are two issues:
        """
    )
    return


@app.cell
def why_linear_fails(df, mo):
    def _():
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: NOX vs TIT is nonlinear — most of the action is at the edges
        axes[0].scatter(df["TIT"], df["NOX"], alpha=0.08, s=3, color="#888")
        axes[0].set_xlabel("TIT (°C)")
        axes[0].set_ylabel("NOX (mg/m³)")
        axes[0].set_title("TIT vs NOX — is a line appropriate?")
        axes[0].axvline(x=1060, color="red", ls="--", alpha=0.5, label="most data here")
        axes[0].axvline(x=1100, color="red", ls="--", alpha=0.5)
        # Mark the dense region
        mask = (df["TIT"] > 1060) & (df["TIT"] < 1100)
        axes[0].annotate(
            f"{mask.mean():.0%} of data\nin this band",
            xy=(1080, 100), fontsize=11, ha="center",
            bbox=dict(boxstyle="round", fc="lightyellow"),
        )

        # Panel 2: NOX vs AT — much clearer, more spread
        axes[1].scatter(df["AT"], df["NOX"], alpha=0.08, s=3, color="#888")
        axes[1].set_xlabel("AT (°C)")
        axes[1].set_ylabel("NOX (mg/m³)")
        axes[1].set_title("AT vs NOX — much wider range, clearer signal")

        plt.tight_layout()
        plt.show()

        return mo.md(
            """
            ### Why the linear estimate is misleading

            **Problem 1 — Narrow range:** TIT spans only 1000-1100°C. Most data is
            clustered between 1060-1100°C. The Zeldovich mechanism produces exponential
            NOX growth, but over a 40°C range, the curve looks nearly flat. A linear fit
            over a flat region picks up noise and confounding instead of the real signal.

            **Problem 2 — Nonlinearity:** Even if we had a wider range, a straight line
            is the wrong functional form for an exponential relationship. The true effect
            of TIT on NOX accelerates at higher temperatures.

            **Problem 3 — AT dominates:** AT spans -6°C to 37°C (a 43°C range) and
            has a clear, strong negative relationship with NOX. In a linear regression,
            AT eats all the explanatory power, leaving TIT with whatever residual
            variation is left — which is mostly noise.

            **Conclusion:** The linear regression is a bad tool for this job. We need
            a method that can handle nonlinearity. That's what notebook 03 is for
            (Double Machine Learning).
            """
        )

    _()
    return


@app.cell
def estimate_with_dowhy(estimand_nox, mo, model_nox):
    # Go ahead and run DoWhy's linear estimate — but now we understand its limits
    estimate_nox = model_nox.estimate_effect(
        estimand_nox,
        method_name="backdoor.linear_regression",
        test_significance=True,
    )

    mo.md(
        f"""
        ## DoWhy's Linear Estimate (for reference)

        DoWhy's backdoor linear regression gives: **{estimate_nox.value:.3f}** mg/m³ per °C.

        We now understand why this is negative:
        1. The backdoor criterion correctly identifies AT, AP as confounders
        2. But the linear functional form is wrong for this relationship
        3. TIT has a narrow range where the true (exponential) effect is weak
        4. AT's strong effect leaves little variation for TIT to explain

        The identification step (backdoor criterion) was correct — it told us
        *what* to control for. The estimation step (linear regression) was the
        problem — it assumed the wrong shape.

        **The fix isn't more confounders. It's a more flexible estimator.**
        """
    )
    return


@app.cell
def co_estimate(causal_graph, df, mo):
    def _():
        from dowhy import CausalModel

        model_co = CausalModel(
            data=df, treatment="TIT", outcome="CO", graph=causal_graph,
        )
        estimand_co = model_co.identify_effect(method_name="default")
        estimate_co = model_co.estimate_effect(
            estimand_co,
            method_name="backdoor.linear_regression",
            test_significance=True,
        )

        return mo.md(
            f"""
            ## Quick Check: TIT → CO

            DoWhy's linear estimate for CO: **{estimate_co.value:.3f}** mg/m³ per °C.

            Negative — as expected. Higher TIT → more complete combustion → less CO.
            This one has the right sign because the CO-TIT relationship is more
            monotonic and closer to linear over this range. But the magnitude is still
            suspect for the same reasons (narrow range, nonlinearity).
            """
        )

    _()
    return


@app.cell
def next_steps(mo):
    mo.md("""
    ## What We've Learned

    1. **The causal graph** encodes domain knowledge about what causes what
    2. **Backdoor paths** are spurious pathways through common causes (AT, AP)
    3. **The backdoor criterion** says: block these paths by controlling for confounders
    4. **Don't control for mediators** (TAT, TEY, CDP) — they're part of the causal effect
    5. **Simpson's paradox** is real: the raw TIT-NOX correlation has the wrong sign
    6. **Linear regression fails** here because TIT's effect is nonlinear and its range is narrow

    ## What's Next (Notebook 03)

    The identification was right. The estimation was wrong. We need methods that:

    - Handle **nonlinear** treatment effects (not just a straight line)
    - Can detect **heterogeneous effects** (TIT's effect might differ on hot vs cold days)
    - Still respect the causal graph (control for the right variables)

    That's exactly what **Double Machine Learning (DML)** does: use flexible ML
    models (random forests, gradient boosting) to control for confounders, while
    still producing valid causal estimates. Coming up next.
    """)
    return


if __name__ == "__main__":
    app.run()
