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
def identify_and_estimate_nox(causal_graph, df, mo):
    def _():
        from dowhy import CausalModel

        # Create the causal model: we want the effect of TIT on NOX
        model_nox = CausalModel(
            data=df,
            treatment="TIT",
            outcome="NOX",
            graph=causal_graph,
        )

        # Identify: DoWhy applies the backdoor criterion automatically
        estimand_nox = model_nox.identify_effect(method_name="default")

        # Estimate with linear regression (backdoor adjustment)
        estimate_nox = model_nox.estimate_effect(
            estimand_nox,
            method_name="backdoor.linear_regression",
            test_significance=True,
        )

        return mo.md(
            f"""
            ## Identification & Estimation: TIT → NOX

            DoWhy applied the backdoor criterion to our graph and found:

            ```
            {estimand_nox}
            ```

            The **backdoor variables** are the ones DoWhy says we must control for.
            These are exactly the confounders — variables that cause both TIT and NOX.

            Notice: TAT, TEY, CDP are **not** in the adjustment set, even though they
            correlate with NOX. They're downstream of TIT (mediators), and controlling
            for them would block the causal pathway we're trying to measure.

            ---

            ### Linear Regression Estimate

            {estimate_nox}

            Controlling for the backdoor variables, each 1°C increase
            in TIT causes approximately **{estimate_nox.value:.3f} mg/m³ change in NOX**.

            But this assumes a constant, linear effect. The domain knowledge says NOX
            increases **exponentially** with temperature (Zeldovich kinetics).
            A linear model is a rough first approximation — we'll use more flexible
            estimators (EconML's DML, causal forests) in the next notebook.
            """
        )

    _()
    return


@app.cell
def naive_vs_adjusted(df, mo):
    def _():
        import numpy as np
        from sklearn.linear_model import LinearRegression

        # Naive: just regress NOX on TIT, ignoring confounders
        X_naive = df[["TIT"]].values
        y = df["NOX"].values
        naive = LinearRegression().fit(X_naive, y)
        naive_coef = naive.coef_[0]

        # Adjusted: include the backdoor variables
        X_adjusted = df[["TIT", "AT", "AP", "AH", "AFDP"]].values
        adjusted = LinearRegression().fit(X_adjusted, y)
        adjusted_coef = adjusted.coef_[0]  # TIT coefficient

        return mo.md(
            f"""
            ## Naive vs. Adjusted: Seeing Confounding Bias

            | Approach | TIT coefficient (effect on NOX per °C) |
            |----------|---------------------------------------|
            | **Naive** (no controls) | {naive_coef:.4f} |
            | **Adjusted** (backdoor controls) | {adjusted_coef:.4f} |

            The difference between these two numbers **is confounding bias**.

            The naive estimate mixes up TIT's real effect with the effect of ambient
            conditions that happen to correlate with TIT. The adjusted estimate isolates
            TIT's causal contribution by holding confounders constant.

            This is the core of causal inference: same data, same outcome, but a
            fundamentally different answer depending on whether you account for
            the causal structure.
            """
        )

    _()
    return


@app.cell
def identify_and_estimate_co(causal_graph, df, mo):
    def _():
        from dowhy import CausalModel

        model_co = CausalModel(
            data=df,
            treatment="TIT",
            outcome="CO",
            graph=causal_graph,
        )

        estimand_co = model_co.identify_effect(method_name="default")

        estimate_co_linear = model_co.estimate_effect(
            estimand_co,
            method_name="backdoor.linear_regression",
            test_significance=True,
        )

        return mo.md(
            f"""
            ## Identification & Estimation: TIT → CO

            ```
            {estimand_co}
            ```

            ### Linear Regression Estimate

            {estimate_co_linear}

            Negative coefficient — as expected. Higher TIT → more complete
            combustion → less CO. This is the other side of the tradeoff.
            """
        )

    _()
    return


@app.cell
def next_steps(mo):
    mo.md("""
    ## What we've done and what's next

    **Done:**
    1. Drew a causal DAG encoding domain knowledge (physics, not statistics)
    2. DoWhy applied the backdoor criterion to find the right adjustment set
    3. Estimated the Average Treatment Effect (ATE) of TIT on NOX and CO
    4. Saw how confounding bias changes the estimate

    **Limitations of what we just did:**
    - Linear regression assumes a constant, linear effect — but NOX vs TIT is nonlinear
    - We estimated one number (the ATE) — but the effect likely varies by conditions
      (e.g., TIT might matter more on hot days than cold days)
    - We haven't tested how sensitive our estimates are to DAG misspecification

    **Next (notebook 03):**
    - Use EconML's Double Machine Learning (DML) for flexible, nonlinear estimation
    - Estimate heterogeneous treatment effects with causal forests
      ("the effect of TIT on NOX is *different* depending on ambient conditions")
    - Refutation tests: how robust are our results if our DAG is slightly wrong?
    """)
    return


if __name__ == "__main__":
    app.run()
