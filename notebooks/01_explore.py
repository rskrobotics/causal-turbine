import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def intro():
    import marimo as mo

    mo.md(
        """
        # 01 — Exploratory Data Analysis
        ## Gas Turbine CO & NOx Emissions

        Before any causal modeling, we need to understand the data:
        - Distributions of each variable
        - Correlations (which are NOT causal — but useful to see)
        - The CO-NOx tradeoff
        - How ambient conditions relate to process variables
        """
    )
    return (mo,)


@app.cell
def load_data():
    import pandas as pd
    import numpy as np

    from pathlib import Path
    df = pd.read_csv(Path(__file__).parent / "../data/gas_turbine_emissions.csv")
    print(f"Shape: {df.shape}")
    print(f"Years: {df['year'].unique()}")
    df.describe().round(2)
    return (df,)


@app.cell
def distributions(df, mo):
    def _():
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        cols = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP", "CO", "NOX"]

        for i, col in enumerate(cols):
            ax = axes[i // 4, i % 4]
            ax.hist(df[col], bins=50, edgecolor="black", alpha=0.7)
            ax.set_title(col)
            ax.set_ylabel("count")

        axes[2, 3].axis("off")
        fig.suptitle("Variable Distributions", fontsize=14)
        plt.tight_layout()
        plt.show()
        return mo.md("**Look for:** skewed distributions (CO is likely right-skewed), multimodality, outliers.")


    _()
    return


@app.cell
def correlation_matrix(df, mo):
    def _():
        import matplotlib.pyplot as plt
        import seaborn as sns

        cols = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP", "CO", "NOX"]
        corr = df[cols].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
        ax.set_title("Correlation Matrix (correlation ≠ causation!)")
        plt.tight_layout()
        plt.show()
        return mo.md(
            """
            **Remember:** These correlations mix up causal effects, confounding, and collider bias.
            A strong correlation between AT and NOx does NOT mean AT causes NOx —
            it could be that hot days → higher load demand → higher TIT → higher NOx.
            Disentangling this is exactly what causal inference does.
            """
        )


    _()
    return


@app.cell
def co_nox_tradeoff(df, mo):
    def _():
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # CO vs NOx colored by TIT
        sc = axes[0].scatter(df["CO"], df["NOX"], c=df["TIT"], cmap="hot", alpha=0.3, s=5)
        axes[0].set_xlabel("CO (mg/m³)")
        axes[0].set_ylabel("NOx (mg/m³)")
        axes[0].set_title("CO vs NOx (colored by TIT)")
        plt.colorbar(sc, ax=axes[0], label="TIT (°C)")

        # NOx vs TIT colored by AT
        sc2 = axes[1].scatter(df["TIT"], df["NOX"], c=df["AT"], cmap="coolwarm", alpha=0.3, s=5)
        axes[1].set_xlabel("TIT (°C)")
        axes[1].set_ylabel("NOx (mg/m³)")
        axes[1].set_title("NOx vs TIT (colored by AT)")
        plt.colorbar(sc2, ax=axes[1], label="AT (°C)")

        # CO vs TIT colored by AT
        sc3 = axes[2].scatter(df["TIT"], df["CO"], c=df["AT"], cmap="coolwarm", alpha=0.3, s=5)
        axes[2].set_xlabel("TIT (°C)")
        axes[2].set_ylabel("CO (mg/m³)")
        axes[2].set_title("CO vs TIT (colored by AT)")
        plt.colorbar(sc3, ax=axes[2], label="AT (°C)")

        plt.tight_layout()
        plt.show()

        return mo.md(
            """
            **Key things to observe:**
            - CO and NOx should show a negative correlation (the tradeoff)
            - NOx should increase with TIT (exponential-ish)
            - CO should decrease with TIT (more complete combustion)
            - The color gradients reveal confounding: same TIT gives different emissions at different ambient temps
            """
        )

    _()
    return


@app.cell
def ambient_effects(df, mo):
    def _():
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        pairs = [
            ("AT", "NOX"), ("AT", "CO"), ("AT", "TEY"),
            ("AH", "NOX"), ("AH", "CO"), ("AP", "CDP"),
        ]

        for ax, (x, y) in zip(axes.flat, pairs):
            ax.scatter(df[x], df[y], alpha=0.1, s=3)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(f"{y} vs {x}")

        plt.suptitle("Ambient Conditions vs Process/Emission Variables", fontsize=14)
        plt.tight_layout()
        plt.show()

        return mo.md(
            """
            **This is where causal thinking starts:**
            If AT correlates with NOx, is it because:
            - (a) AT directly affects combustion chemistry? (causal)
            - (b) Hot days → higher grid demand → higher TIT → higher NOx? (confounded)
            - (c) Both?

            Standard ML doesn't care — it just predicts. Causal ML must distinguish these pathways.
            """
        )

    _()
    return


@app.cell
def yearly_patterns(df, mo):
    def _():
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        for ax, col in zip(axes, ["NOX", "CO", "TEY"]):
            df.boxplot(column=col, by="year", ax=ax)
            ax.set_title(col)
            ax.set_xlabel("Year")

        plt.suptitle("Year-over-Year Patterns", fontsize=14)
        plt.tight_layout()
        plt.show()

        return mo.md(
            """
            **Check for drift:** If emissions change year-over-year, it could be due to
            equipment aging (filter degradation, turbine wear) or changing operating conditions.
            This matters for causal identification — time trends are potential confounders.
            """
        )

    _()
    return


if __name__ == "__main__":
    app.run()
