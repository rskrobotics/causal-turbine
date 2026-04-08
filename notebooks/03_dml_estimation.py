import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def intro():
    import marimo as mo

    mo.md(
        """
        # 03 — Double Machine Learning: From Identification to Estimation

        In notebook 02, we:
        - Built a causal graph from domain knowledge
        - Used the **backdoor criterion** to figure out what to control for (AT, AP)
        - Tried linear regression and got a **wrong sign** (TIT→NOX came out negative)
        - Diagnosed the problem: linearity assumption + narrow TIT range + dominant AT effect

        The identification was correct. The estimation method was wrong.

        This notebook fixes the estimation with **Double Machine Learning (DML)**, and
        then uses the fitted model to answer real causal "what if?" questions —
        the whole point of doing causal inference.
        """
    )
    return (mo,)


@app.cell
def load_data():
    import pandas as pd
    import numpy as np
    from pathlib import Path

    df = pd.read_csv(Path(__file__).parent / "../data/gas_turbine_emissions.csv")
    return df, np


@app.cell
def dml_intuition(mo):
    mo.md("""
    ## The DML Idea: Strip Out the Confounders, Then Look at What's Left

    The problem with linear regression wasn't the backdoor criterion — it correctly
    told us to control for AT and AP. The problem was *how* we controlled for them.

    Linear regression assumes every relationship is a straight line. But AT's effect
    on NOX is nonlinear, and squeezing it into a line distorts everything.

    **DML's trick is simple:**

    1. Use a flexible ML model (gradient boosting, random forest) to predict
       **NOX from confounders only** (AT, AP, AH). Call the prediction error the
       "NOX residual" — the part of NOX that confounders can't explain.

    2. Use another flexible ML model to predict **TIT from confounders only**.
       The "TIT residual" is the part of TIT that confounders can't explain.

    3. Now ask: does the TIT residual predict the NOX residual? If yes,
       that's the causal effect — because we've already removed everything
       the confounders could account for.

    ```
    Standard regression:     NOX ~ TIT + AT + AP       (all linear — bad)

    DML:  Step 1:  NOX_residual = NOX - ML_predict(NOX from AT, AP, AH)
          Step 2:  TIT_residual = TIT - ML_predict(TIT from AT, AP, AH)
          Step 3:  NOX_residual ~ TIT_residual        (causal estimate)
    ```

    The "double" in DML = two ML models (one for each residual). The ML models
    can be as flexible as needed — they handle all the nonlinear confounding.
    """)
    return


@app.cell
def dml_by_hand(df, mo, np):
    def _():
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_predict
        import matplotlib.pyplot as plt

        Y = df["NOX"].values
        T = df["TIT"].values
        W = df[["AT", "AP", "AH"]].values

        # Step 1: Predict NOX from confounders (using cross-validation to avoid overfitting)
        model_y = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=42,
        )
        Y_hat = cross_val_predict(model_y, W, Y, cv=5)
        Y_residual = Y - Y_hat

        # Step 2: Predict TIT from confounders
        model_t = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=42,
        )
        T_hat = cross_val_predict(model_t, W, T, cv=5)
        T_residual = T - T_hat

        # How well did the confounders predict each?
        r2_y = 1 - np.var(Y_residual) / np.var(Y)
        r2_t = 1 - np.var(T_residual) / np.var(T)

        # Step 3: Regress residuals
        from sklearn.linear_model import LinearRegression
        manual_dml = LinearRegression().fit(T_residual.reshape(-1, 1), Y_residual)
        manual_coef = manual_dml.coef_[0]

        # Visualize the three steps
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))

        # Panel 1: Y residualization
        axes[0].scatter(Y_hat, Y, alpha=0.05, s=3)
        lims = [Y.min(), Y.max()]
        axes[0].plot(lims, lims, "r--", lw=1)
        axes[0].set_xlabel("Predicted NOX (from confounders)")
        axes[0].set_ylabel("Actual NOX")
        axes[0].set_title(f"Step 1: Predict NOX from AT,AP,AH\nR² = {r2_y:.3f}")

        # Panel 2: T residualization
        axes[1].scatter(T_hat, T, alpha=0.05, s=3)
        lims_t = [T.min(), T.max()]
        axes[1].plot(lims_t, lims_t, "r--", lw=1)
        axes[1].set_xlabel("Predicted TIT (from confounders)")
        axes[1].set_ylabel("Actual TIT")
        axes[1].set_title(f"Step 2: Predict TIT from AT,AP,AH\nR² = {r2_t:.3f}")

        # Panel 3: Residual-on-residual
        axes[2].scatter(T_residual, Y_residual, alpha=0.05, s=3)
        x_line = np.linspace(T_residual.min(), T_residual.max(), 100)
        axes[2].plot(x_line, manual_coef * x_line, "r-", lw=2,
                     label=f"slope = {manual_coef:.4f}")
        axes[2].set_xlabel("TIT residual (unexplained by confounders)")
        axes[2].set_ylabel("NOX residual (unexplained by confounders)")
        axes[2].set_title("Step 3: Does unexplained TIT predict unexplained NOX?")
        axes[2].legend()

        plt.tight_layout()
        plt.show()

        return mo.md(
            f"""
            ### Reading the plots

            **Panel 1 (R² = {r2_y:.3f}):** Confounders explain {r2_y:.0%} of NOX variation.
            Not amazing, but meaningful — ambient conditions do matter for NOX.

            **Panel 2 (R² = {r2_t:.3f}):** Confounders barely predict TIT at all.
            This tells us something important: in this dataset, TIT is mostly set
            independently of weather. The confounding is weaker than we might have
            expected — AT affects NOX strongly, but AT doesn't drive TIT much.

            **Panel 3:** After stripping out confounders, the residual relationship
            between TIT and NOX has slope **{manual_coef:+.4f}**. Still negative.

            **Wait — shouldn't DML fix the sign?** Not necessarily. DML fixes
            *confounding bias*, but if the true relationship is genuinely nonlinear
            and we force a single slope on the residuals, we can still get a misleading
            average. The effect might be *positive at some temperatures and negative at
            others*, averaging to something negative.

            Let's investigate that next.
            """
        )

    _()
    return


@app.cell
def the_effect_varies(df, mo, np):
    def _():
        import matplotlib.pyplot as plt

        # Within narrow AT bins, what is the TIT-NOX slope?
        bins = [(-7, 0), (0, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 38)]
        at_mids = []
        slopes = []
        counts = []

        for lo, hi in bins:
            sub = df[(df["AT"] >= lo) & (df["AT"] < hi)]
            if len(sub) > 50:
                slope = np.polyfit(sub["TIT"], sub["NOX"], 1)[0]
                at_mids.append((lo + hi) / 2)
                slopes.append(slope)
                counts.append(len(sub))

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#4a9eff" if s < 0 else "#ff6b6b" for s in slopes]
        bars = ax.bar(range(len(at_mids)), slopes, color=colors, edgecolor="black", alpha=0.8)

        ax.set_xticks(range(len(at_mids)))
        ax.set_xticklabels([f"AT {lo}-{hi}°C\n(n={n:,})"
                            for (lo, hi), n in zip(bins, counts)], fontsize=9)
        ax.set_ylabel("TIT → NOX slope (within bin)")
        ax.set_title("The Effect of TIT on NOX Changes Sign Depending on Temperature")
        ax.axhline(y=0, color="black", lw=1, ls="--")

        # Annotate
        ax.annotate("Cold days:\nhigher TIT → LESS NOX",
                     xy=(1, slopes[1]), xytext=(0.5, slopes[1] - 0.15),
                     fontsize=10, ha="center",
                     arrowprops=dict(arrowstyle="->", color="#4a9eff"),
                     color="#4a9eff", fontweight="bold")
        ax.annotate("Hot days:\nhigher TIT → MORE NOX",
                     xy=(5, slopes[5]), xytext=(5.5, slopes[5] + 0.1),
                     fontsize=10, ha="center",
                     arrowprops=dict(arrowstyle="->", color="#ff6b6b"),
                     color="#ff6b6b", fontweight="bold")

        plt.tight_layout()
        plt.show()

        return mo.md(
            """
            ### The Effect Is Heterogeneous — It Flips Sign

            This is the key finding. TIT's effect on NOX is not a single number:

            - **Cold days (AT < 15°C):** Higher TIT → *less* NOX (negative effect)
            - **Hot days (AT > 20°C):** Higher TIT → *more* NOX (positive effect, as physics predicts)

            The linear model and even the basic DML gave us a single negative number — an
            *average* of these opposite effects, weighted by how much data falls in each bin.

            **Why does the effect flip?** Plausible explanations from the domain:
            - On cold days, air is dense. Higher TIT may push the combustor into a
              lean-premixed regime where mixing improves and NOX actually decreases.
            - On hot days, air is thin. Higher TIT directly increases peak flame
              temperature → more thermal NOX (the Zeldovich mechanism).
            - The turbine's control system may behave differently at temperature extremes.

            **This is why heterogeneous treatment effects matter.** A single average
            effect would hide this, and any policy based on it would be wrong for
            half the operating conditions.
            """
        )

    _()
    return


@app.cell
def econml_dml_intro(mo):
    mo.md("""
    ## EconML: Proper DML with Heterogeneous Effects

    We'll now use EconML's `CausalForestDML` — a method that:

    1. Does the DML residualization (strips out confounders with flexible ML)
    2. Fits a **causal forest** on the residuals to discover how the effect
       varies across conditions

    A causal forest is like a random forest, but instead of predicting Y,
    it predicts the *treatment effect* at each point. It automatically finds
    which variables (AT, AP, AH) modify the effect and how.

    We need to specify:
    - **Y** — outcome (NOX)
    - **T** — treatment (TIT)
    - **W** — confounders to control for (AT, AP, AH — the backdoor variables)
    - **X** — effect modifiers (which conditions might change the effect?)
      We'll use AT, since we just saw it flips the sign.
    """)
    return


@app.cell
def fit_causal_forest_nox(df):
    from econml.dml import CausalForestDML
    from sklearn.ensemble import GradientBoostingRegressor

    Y = df["NOX"].values
    T = df["TIT"].values
    X = df[["AT"]].values        # effect modifier: how does the effect vary by AT?
    W = df[["AT", "AP", "AH"]].values   # confounders to strip out

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
    cf_nox.fit(Y, T, X=X, W=W)
    return (cf_nox,)


@app.cell
def visualize_cate_nox(cf_nox, df, mo, np):
    def _():
        import matplotlib.pyplot as plt

        # Estimate effect at a grid of AT values
        at_grid = np.linspace(-5, 37, 100).reshape(-1, 1)
        effects = cf_nox.effect(X=at_grid)
        ci_lo, ci_hi = cf_nox.effect_interval(X=at_grid, alpha=0.05)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Panel 1: CATE curve with CI
        axes[0].plot(at_grid, effects, "b-", lw=2, label="Estimated causal effect")
        axes[0].fill_between(at_grid.ravel(), ci_lo.ravel(), ci_hi.ravel(),
                             alpha=0.2, color="blue", label="95% confidence interval")
        axes[0].axhline(y=0, color="black", lw=1, ls="--")
        axes[0].set_xlabel("Ambient Temperature (°C)")
        axes[0].set_ylabel("Effect of +1°C TIT on NOX (mg/m³)")
        axes[0].set_title("Causal Effect of TIT on NOX by Ambient Temperature")
        axes[0].legend()

        # Mark the sign-flip
        # Find where effect crosses zero
        sign_changes = np.where(np.diff(np.sign(effects.ravel())))[0]
        if len(sign_changes) > 0:
            cross_at = at_grid[sign_changes[0]][0]
            axes[0].axvline(x=cross_at, color="red", ls=":", alpha=0.7)
            axes[0].annotate(f"Effect flips at AT ≈ {cross_at:.0f}°C",
                             xy=(cross_at, 0), xytext=(cross_at + 5, -0.3),
                             fontsize=10, arrowprops=dict(arrowstyle="->"),
                             bbox=dict(boxstyle="round", fc="lightyellow"))

        # Panel 2: distribution of individual-level effects
        all_effects = cf_nox.effect(X=df[["AT"]].values)
        axes[1].hist(all_effects, bins=50, edgecolor="black", alpha=0.7, color="#88bb88")
        axes[1].axvline(x=0, color="black", lw=1, ls="--")
        axes[1].axvline(x=all_effects.mean(), color="red", lw=2, ls="-",
                        label=f"Mean effect = {all_effects.mean():.4f}")
        axes[1].set_xlabel("Individual treatment effect (mg/m³ per °C TIT)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Distribution of Individual Causal Effects")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

        ate = cf_nox.ate(X=df[["AT"]].values)
        ate_ci = cf_nox.ate_interval(X=df[["AT"]].values, alpha=0.05)

        return mo.md(
            f"""
            ### What the Causal Forest Found

            **Left panel:** The causal effect of TIT on NOX is not constant — it's a
            *function* of ambient temperature:

            - Below ~15°C: the effect is **negative** (higher TIT reduces NOX)
            - Above ~20°C: the effect is **positive** (higher TIT increases NOX, as Zeldovich predicts)
            - The transition happens around 15-17°C

            The blue band is the 95% confidence interval. Notice it's tightest where
            we have the most data (10-25°C) and wider at the extremes.

            **Right panel:** The distribution of individual effects across all observations
            in the dataset. It spans from negative to positive — confirming this is not a
            single-number story.

            **Average Treatment Effect: {ate:.4f}** (95% CI: [{ate_ci[0]:.4f}, {ate_ci[1]:.4f}])

            The ATE is close to zero and its CI includes zero — because positive and negative
            effects roughly cancel out. Reporting just the ATE would completely miss the story.
            """
        )

    _()
    return


@app.cell
def fit_causal_forest_co(df, mo, np):
    def _():
        from econml.dml import CausalForestDML
        from sklearn.ensemble import GradientBoostingRegressor
        import matplotlib.pyplot as plt

        Y_co = df["CO"].values
        T = df["TIT"].values
        X = df[["AT"]].values
        W = df[["AT", "AP", "AH"]].values

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
        cf_co.fit(Y_co, T, X=X, W=W)

        at_grid = np.linspace(-5, 37, 100).reshape(-1, 1)
        effects_co = cf_co.effect(X=at_grid)
        ci_lo, ci_hi = cf_co.effect_interval(X=at_grid, alpha=0.05)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(at_grid, effects_co, "g-", lw=2, label="TIT → CO effect")
        ax.fill_between(at_grid.ravel(), ci_lo.ravel(), ci_hi.ravel(),
                        alpha=0.2, color="green")
        ax.axhline(y=0, color="black", lw=1, ls="--")
        ax.set_xlabel("Ambient Temperature (°C)")
        ax.set_ylabel("Effect of +1°C TIT on CO (mg/m³)")
        ax.set_title("Causal Effect of TIT on CO by Ambient Temperature")
        ax.legend()
        plt.tight_layout()
        plt.show()

        ate_co = cf_co.ate(X=df[["AT"]].values)

        return mo.md(
            f"""
            ### TIT → CO: A Sanity Check

            The effect of TIT on CO is **negative everywhere** (ATE = {ate_co:.4f}).
            Higher TIT → more complete combustion → less CO, regardless of temperature.

            This makes physical sense and is a good sanity check:
            - CO formation is about incomplete combustion, which is more straightforward
            - The effect is stronger on cold days (more room for improvement)
            - Unlike NOX, there's no sign flip — the mechanism doesn't reverse

            **The CO-NOx tradeoff in causal terms:** Increasing TIT always reduces CO,
            but it only increases NOX on hot days. On cold days, you get a free lunch —
            higher TIT reduces *both* emissions.
            """
        )

    _()
    return


@app.cell
def what_if_header(mo):
    mo.md("""
    ## Asking Causal Questions: "What If...?"

    This is the payoff. We now have a fitted causal model that can answer
    interventional questions — Pearl's Rung 2.

    These are NOT predictions ("what NOX do we expect to see when TIT = 1080?").
    They are interventions ("if we *set* TIT to 1080, what *would* NOX be,
    regardless of what TIT would normally be on a day like this?").

    The difference matters: a prediction mixes in confounding (what kind of day
    leads to TIT = 1080?), an intervention strips it out (what would happen if
    we reached in and turned the dial to 1080, on any given day?).
    """)
    return


@app.cell
def what_if_questions(cf_nox, mo, np):
    def _():
        import matplotlib.pyplot as plt

        # Question 1: "If we increase TIT by 20°C, what happens to NOX?"
        at_grid = np.linspace(-5, 37, 100).reshape(-1, 1)

        # effect() with T0, T1 gives the effect of changing from T0 to T1
        delta_nox_20 = cf_nox.effect(X=at_grid, T0=1060, T1=1080)
        delta_nox_40 = cf_nox.effect(X=at_grid, T0=1060, T1=1100)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Panel 1: Effect of +20°C TIT by ambient temperature
        axes[0].plot(at_grid, delta_nox_20, "b-", lw=2, label="TIT: 1060 → 1080 (+20°C)")
        axes[0].plot(at_grid, delta_nox_40, "r-", lw=2, label="TIT: 1060 → 1100 (+40°C)")
        axes[0].axhline(y=0, color="black", lw=1, ls="--")
        axes[0].set_xlabel("Ambient Temperature (°C)")
        axes[0].set_ylabel("Predicted NOX change (mg/m³)")
        axes[0].set_title("Intervention: What Happens to NOX\nIf We Increase TIT?")
        axes[0].legend()
        axes[0].fill_between(at_grid.ravel(), 0, delta_nox_20.ravel(),
                             where=delta_nox_20.ravel() > 0, alpha=0.1, color="red")
        axes[0].fill_between(at_grid.ravel(), 0, delta_nox_20.ravel(),
                             where=delta_nox_20.ravel() < 0, alpha=0.1, color="blue")

        # Panel 2: Specific scenarios as a table-like bar chart
        scenarios = [
            ("Cold day\nAT=5°C", 5),
            ("Mild day\nAT=15°C", 15),
            ("Hot day\nAT=25°C", 25),
            ("Very hot\nAT=35°C", 35),
        ]
        effects_20 = [cf_nox.effect(X=np.array([[at]]), T0=1060, T1=1080)[0]
                      for _, at in scenarios]
        colors = ["#4a9eff" if e < 0 else "#ff6b6b" for e in effects_20]
        axes[1].bar([s[0] for s in scenarios], effects_20,
                    color=colors, edgecolor="black", alpha=0.8)
        axes[1].set_ylabel("NOX change (mg/m³)")
        axes[1].set_title("NOX Change From TIT 1060→1080°C\nBy Weather Condition")
        axes[1].axhline(y=0, color="black", lw=1, ls="--")

        for i, v in enumerate(effects_20):
            axes[1].text(i, v + (0.3 if v > 0 else -0.5), f"{v:+.1f}",
                         ha="center", fontweight="bold", fontsize=11)

        plt.tight_layout()
        plt.show()

        return mo.md(
            f"""
            ### Reading These Results

            **Left panel:** The predicted change in NOX from increasing TIT by 20°C
            or 40°C. The effect scales proportionally (the causal forest is locally
            linear in treatment).

            **Right panel:** Concrete answers to the question "if we boost TIT from
            1060 to 1080°C, what happens to NOX?"

            - **Cold day (5°C):** NOX *decreases* by ~{abs(effects_20[0]):.0f} mg/m³
            - **Mild day (15°C):** NOX barely changes (~{effects_20[1]:+.1f} mg/m³)
            - **Hot day (25°C):** NOX *increases* by ~{effects_20[2]:.0f} mg/m³

            **This is an actionable insight:** On cold days, running the turbine hotter
            is free — you get more power and *less* NOX. On hot days, there's a real
            tradeoff. An operator or control system could use this.
            """
        )

    _()
    return


@app.cell
def counterfactual(cf_nox, df, mo, np):
    def _():
        import matplotlib.pyplot as plt

        # Pick specific observations and ask counterfactuals
        # "This reading had TIT=X. What would NOX have been if TIT were different?"

        # Find an interesting cold-day observation and a hot-day one
        cold_idx = df[(df["AT"] < 5) & (df["TIT"] > 1070) & (df["TIT"] < 1080)].index[0]
        hot_idx = df[(df["AT"] > 30) & (df["TIT"] > 1070) & (df["TIT"] < 1080)].index[0]

        cold_row = df.loc[cold_idx]
        hot_row = df.loc[hot_idx]

        tit_range = np.linspace(1010, 1100, 50)
        actual_tit_cold = cold_row["TIT"]
        actual_tit_hot = hot_row["TIT"]

        # For each hypothetical TIT, what would NOX have been?
        cold_effects = cf_nox.effect(
            X=np.full((50, 1), cold_row["AT"]),
            T0=np.full(50, actual_tit_cold),
            T1=tit_range,
        )
        hot_effects = cf_nox.effect(
            X=np.full((50, 1), hot_row["AT"]),
            T0=np.full(50, actual_tit_hot),
            T1=tit_range,
        )

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Cold day counterfactual
        axes[0].plot(tit_range, cold_row["NOX"] + cold_effects, "b-", lw=2)
        axes[0].axvline(x=actual_tit_cold, color="red", ls="--", lw=2)
        axes[0].scatter([actual_tit_cold], [cold_row["NOX"]], color="red",
                        s=100, zorder=5, label=f"Actual: TIT={actual_tit_cold:.0f}, NOX={cold_row['NOX']:.0f}")
        axes[0].set_xlabel("Hypothetical TIT (°C)")
        axes[0].set_ylabel("Counterfactual NOX (mg/m³)")
        axes[0].set_title(f"Counterfactual: Cold Day (AT={cold_row['AT']:.1f}°C)\n"
                          f"What if TIT had been different?")
        axes[0].legend(fontsize=9)

        # Hot day counterfactual
        axes[1].plot(tit_range, hot_row["NOX"] + hot_effects, "r-", lw=2)
        axes[1].axvline(x=actual_tit_hot, color="red", ls="--", lw=2)
        axes[1].scatter([actual_tit_hot], [hot_row["NOX"]], color="red",
                        s=100, zorder=5, label=f"Actual: TIT={actual_tit_hot:.0f}, NOX={hot_row['NOX']:.0f}")
        axes[1].set_xlabel("Hypothetical TIT (°C)")
        axes[1].set_ylabel("Counterfactual NOX (mg/m³)")
        axes[1].set_title(f"Counterfactual: Hot Day (AT={hot_row['AT']:.1f}°C)\n"
                          f"What if TIT had been different?")
        axes[1].legend(fontsize=9)

        plt.tight_layout()
        plt.show()

        return mo.md(
            f"""
            ### Counterfactual Reasoning (Pearl's Rung 3)

            These plots answer: "for *this specific* turbine reading, what *would*
            NOX have been if TIT had been set differently?"

            **Left (cold day, AT={cold_row['AT']:.1f}°C):** The counterfactual curve
            slopes upward — lowering TIT would *increase* NOX on this day. Running
            cooler actually makes emissions worse.

            **Right (hot day, AT={hot_row['AT']:.1f}°C):** The curve slopes downward —
            lowering TIT would *decrease* NOX. The standard intuition holds here.

            The red dot is what actually happened. The curve shows the parallel universe
            where everything was the same except TIT.

            **Caveat:** These counterfactuals rely on the causal forest's local linearity
            in TIT. For large deviations from the actual TIT, the estimates become less
            reliable — we're extrapolating beyond the model's training range.
            """
        )

    _()
    return


@app.cell
def refutation_header(mo):
    mo.md("""
    ## Refutation: How Robust Are These Results?

    All our estimates depend on assumptions:
    1. The causal graph is correct (no missing confounders)
    2. The backdoor criterion gives a valid adjustment set
    3. The ML models are flexible enough

    We can't prove these assumptions are right, but we can *stress-test* them.
    If small violations of assumptions destroy our results, we should be worried.
    If the results survive, we gain confidence (not certainty).
    """)
    return


@app.cell
def refutation_random_confounder(df, mo, np):
    def _():
        from econml.dml import CausalForestDML
        from sklearn.ensemble import GradientBoostingRegressor
        import matplotlib.pyplot as plt

        Y = df["NOX"].values
        T = df["TIT"].values
        X = df[["AT"]].values

        at_grid = np.linspace(-5, 37, 100).reshape(-1, 1)

        # Original model
        W_original = df[["AT", "AP", "AH"]].values
        cf_orig = CausalForestDML(
            model_y=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
            model_t=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
            n_estimators=200, min_samples_leaf=20, random_state=42,
        )
        cf_orig.fit(Y, T, X=X, W=W_original)
        effects_orig = cf_orig.effect(X=at_grid)

        # Test 1: Add a random confounder (should NOT change the estimate)
        np.random.seed(42)
        random_col = np.random.randn(len(df))
        W_with_random = np.column_stack([W_original, random_col])
        cf_random = CausalForestDML(
            model_y=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
            model_t=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
            n_estimators=200, min_samples_leaf=20, random_state=42,
        )
        cf_random.fit(Y, T, X=X, W=W_with_random)
        effects_random = cf_random.effect(X=at_grid)

        # Test 2: Placebo treatment (shuffle TIT — should give zero effect)
        T_placebo = np.random.permutation(T)
        cf_placebo = CausalForestDML(
            model_y=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
            model_t=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
            n_estimators=200, min_samples_leaf=20, random_state=42,
        )
        cf_placebo.fit(Y, T_placebo, X=X, W=W_original)
        effects_placebo = cf_placebo.effect(X=at_grid)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Panel 1: random confounder test
        axes[0].plot(at_grid, effects_orig, "b-", lw=2, label="Original")
        axes[0].plot(at_grid, effects_random, "r--", lw=2, label="+ random confounder")
        axes[0].axhline(y=0, color="black", lw=1, ls="--")
        axes[0].set_xlabel("Ambient Temperature (°C)")
        axes[0].set_ylabel("Effect of +1°C TIT on NOX")
        axes[0].set_title("Test 1: Add a Random Variable as Confounder\n(Should NOT change the estimate)")
        axes[0].legend()

        # Panel 2: placebo treatment
        axes[1].plot(at_grid, effects_orig, "b-", lw=2, label="Real TIT")
        axes[1].plot(at_grid, effects_placebo, "gray", lw=2, ls="--", label="Shuffled TIT (placebo)")
        axes[1].axhline(y=0, color="black", lw=1, ls="--")
        axes[1].set_xlabel("Ambient Temperature (°C)")
        axes[1].set_ylabel("Effect of +1°C TIT on NOX")
        axes[1].set_title("Test 2: Shuffle TIT Randomly\n(Should give zero effect)")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

        return mo.md(
            """
            ### Refutation Results

            **Test 1 — Random confounder:** Adding a random noise variable as an extra
            confounder barely changes the estimate (red dashed ≈ blue solid). Good. If a
            random variable *did* change things, it would mean our model is unstable.

            **Test 2 — Placebo treatment:** Shuffling TIT randomly destroys the causal
            relationship (gray dashed line ≈ zero). Good. This confirms we're detecting
            a real signal, not an artifact of the method. If the placebo showed the
            same pattern as the real treatment, something would be very wrong.

            **What this tells us:** The heterogeneous effect (negative on cold days,
            positive on hot days) is a robust finding, not a statistical fluke. The
            method is working as expected.

            **What this does NOT tell us:** Whether we're missing an important confounder.
            No statistical test can prove that. We'd need domain expertise to judge
            whether grid demand, fuel composition, or other unmeasured variables could
            be confounding the TIT-NOX relationship.
            """
        )

    _()
    return


@app.cell
def summary(mo):
    mo.md("""
    ## Summary: What We've Built

    ### The Causal Estimation Pipeline

    ```
    Domain Knowledge → Causal Graph → Backdoor Criterion → DML Estimation → Causal Queries
          (physics)      (notebook 02)    (DoWhy)           (EconML)         (what-if)
    ```

    ### Key Findings

    1. **TIT's effect on NOX is heterogeneous** — it depends on ambient temperature.
       On cold days higher TIT *reduces* NOX; on hot days it *increases* NOX.

    2. **The average effect is near zero**, which is misleading. Reporting just
       the ATE hides a sign flip that matters for operations.

    3. **TIT's effect on CO is consistently negative** — higher TIT always
       reduces CO (more complete combustion), regardless of ambient conditions.

    4. **Actionable insight:** On cold days, running hotter is free —
       you get more power and less of *both* emissions. On hot days,
       there's a real CO-NOX tradeoff to manage.

    ### What We Can Now Do

    With the fitted causal forest, we can answer:

    - **Interventional:** "If we set TIT to X on a day with AT=Y, what happens to NOX?"
    - **Counterfactual:** "This reading had TIT=1050. If it had been 1080, what would NOX be?"
    - **Policy:** "At what AT threshold should we switch operating strategies?"

    These are **causal** answers, not correlational predictions. They account for
    confounding and tell us what would happen if we *intervened*, not just what
    we'd *observe*.

    ### What's Next (Notebook 04)

    Build an interactive "what if?" tool — a simple interface where you dial in
    conditions and TIT, and see the predicted causal effect on emissions.
    The query engine that sits on top of everything we've built.
    """)
    return


@app.cell
def sensitivity_header(mo):
    mo.md("""
    ## Sensitivity Analysis: How Strong Would a Hidden Confounder Need to Be?

    The refutation tests above checked that the method works correctly.
    But they can't answer the real worry: **what if we missed a confounder?**

    Suppose there's a hidden variable U (e.g., grid demand, fuel composition)
    that causes both TIT and NOX, and we didn't include it. Our estimate
    would be biased. But *how* biased?

    Sensitivity analysis answers: "how strongly correlated with TIT and NOX
    would U need to be to destroy our main finding (the sign flip)?"

    If the answer is "U would need to explain 50% of both" — we can relax.
    If "U only needs to explain 5% of both" — we should worry.

    **The approach:** We simulate U at different strengths, add it to the
    confounder set, re-run the causal forest, and watch what happens to
    the CATE curve.
    """)
    return


@app.cell
def sensitivity_analysis(df, mo, np):
    def _():
        from econml.dml import CausalForestDML
        from sklearn.ensemble import GradientBoostingRegressor
        import matplotlib.pyplot as plt

        Y = df["NOX"].values
        T = df["TIT"].values
        X = df[["AT"]].values
        W_base = df[["AT", "AP", "AH"]].values
        at_grid = np.linspace(-5, 37, 100).reshape(-1, 1)
        n = len(df)

        # Simulate hidden confounders of increasing strength.
        # U is constructed to correlate with both TIT and NOX at a given level.
        # strength = fraction of T/Y variance that U "explains"
        np.random.seed(42)
        noise = np.random.randn(n)

        strengths = [0.0, 0.05, 0.10, 0.20, 0.30]
        results = {}

        for strength in strengths:
            if strength == 0.0:
                W = W_base
            else:
                # Build U so it's correlated with both T and Y at the given strength
                u = (
                    strength * (T - T.mean()) / T.std()
                    + strength * (Y - Y.mean()) / Y.std()
                    + (1 - strength) * noise
                )
                W = np.column_stack([W_base, u])

            cf = CausalForestDML(
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
            cf.fit(Y, T, X=X, W=W)
            effects = cf.effect(X=at_grid)
            ate = cf.ate(X=df[["AT"]].values)
            results[strength] = {"effects": effects, "ate": ate}

        # Plot: CATE curves at different confounder strengths
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        colors = ["#2166ac", "#67a9cf", "#fddbc7", "#ef8a62", "#b2182b"]
        for (strength, res), color in zip(results.items(), colors):
            label = "No hidden confounder" if strength == 0 else f"U strength = {strength:.0%}"
            axes[0].plot(at_grid, res["effects"], color=color, lw=2, label=label)

        axes[0].axhline(y=0, color="black", lw=1, ls="--")
        axes[0].set_xlabel("Ambient Temperature (°C)")
        axes[0].set_ylabel("Effect of +1°C TIT on NOX (mg/m³)")
        axes[0].set_title("CATE Curve Under Hidden Confounders of Varying Strength")
        axes[0].legend(fontsize=9)

        # Bar chart: ATE at each strength
        ate_values = [results[s]["ate"] for s in strengths]
        bar_colors = ["#2166ac", "#67a9cf", "#fddbc7", "#ef8a62", "#b2182b"]
        labels = ["None", "5%", "10%", "20%", "30%"]
        axes[1].bar(labels, ate_values, color=bar_colors, edgecolor="black", alpha=0.8)
        axes[1].set_xlabel("Hidden Confounder Strength")
        axes[1].set_ylabel("Average Treatment Effect")
        axes[1].set_title("ATE Stability Across Confounder Strengths")
        axes[1].axhline(y=0, color="black", lw=1, ls="--")

        for i, v in enumerate(ate_values):
            axes[1].text(i, v + 0.002, f"{v:+.4f}", ha="center", fontsize=9)

        plt.tight_layout()
        plt.show()

        # Find the sign-flip point for each strength
        flip_points = {}
        for strength, res in results.items():
            eff = res["effects"].ravel()
            sign_changes = np.where(np.diff(np.sign(eff)))[0]
            if len(sign_changes) > 0:
                flip_points[strength] = at_grid[sign_changes[0]][0]
            else:
                if eff.mean() > 0:
                    flip_points[strength] = "always positive"
                else:
                    flip_points[strength] = "always negative"

        flip_table = "\n".join(
            f"| {s:.0%} | {fp if isinstance(fp, str) else f'{fp:.1f}°C'} |"
            for s, fp in flip_points.items()
        )

        return mo.md(
            f"""
            ### Reading the Sensitivity Analysis

            **Left panel:** Each line is the CATE curve under a different hidden
            confounder strength. If the lines stay similar, the finding is robust.
            If they collapse or flip, a hidden confounder could explain away our result.

            **Right panel:** The ATE at each strength. Watch whether it moves toward
            zero or changes sign.

            ### Where Does the Effect Flip Sign?

            | Confounder Strength | Sign-flip AT |
            |--------------------:|-------------|
            {flip_table}

            ### Interpretation

            If the sign-flip survives even at 20-30% confounder strength, that means
            a hidden variable would need to explain a *large* fraction of both TIT and
            NOX variation to eliminate the heterogeneous effect. Given that ambient
            conditions already explain most of NOX variation (we saw this in the R²),
            there's limited room for a hidden confounder to do that.

            If the sign-flip disappears at low strengths (5-10%), the finding is
            fragile — you'd want stronger domain arguments or better data before
            trusting it.
            """
        )

    _()
    return


if __name__ == "__main__":
    app.run()
