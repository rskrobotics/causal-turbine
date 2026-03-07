# Gas Turbine Emissions — Domain Knowledge

## How a Gas Turbine Works

A gas turbine runs the **Brayton cycle** — a continuous-flow process through three stages:

```
    AIR IN          FUEL IN
       ↓               ↓
  ┌─────────┐    ┌───────────┐    ┌─────────┐
  │COMPRESSOR│───→│ COMBUSTOR │───→│ TURBINE │───→ EXHAUST
  └─────────┘    └───────────┘    └─────────┘
       ↑                                │
       └────── mechanical shaft ────────┘
                                        │
                                   GENERATOR → electricity (TEY)
```

1. **Compression** — Ambient air is drawn through an inlet filter, then an axial compressor squeezes it to 10-15x ambient pressure. The compressor moves a fixed *volume* per revolution, so the *mass* of air depends on inlet air density.

2. **Combustion** — Compressed air enters the combustion chamber. Fuel (natural gas) is injected and burned, raising temperature to TIT (1000-1100°C in this dataset). This is roughly constant-pressure.

3. **Expansion** — Hot gas expands through the turbine, spinning it. ~60% of turbine power drives the compressor; the rest drives the generator. Gas exits at TAT (~510-550°C).

---

## The Variables

### Ambient Conditions (uncontrollable — set by weather)

| Variable | What it is | Range in dataset |
|----------|-----------|-----------------|
| **AT** | Ambient Temperature (°C) | -6 to 37°C |
| **AP** | Ambient Pressure (mbar) | 986 to 1037 mbar |
| **AH** | Ambient Humidity (%) | 24 to 100% |

**Why they matter:** Air density = P / (R × T). Hotter or lower-pressure air is less dense → less mass flow through the compressor → less power, different combustion. Every 10°C increase above 15°C costs 5-10% power output. Humidity displaces heavier N₂/O₂ with lighter H₂O, slightly reducing density — but more importantly, water vapor absorbs heat in the flame, lowering peak flame temperature.

### Process Variables

| Variable | What it is | Range | Controllable? |
|----------|-----------|-------|---------------|
| **AFDP** | Air Filter Differential Pressure (mbar) | 2-8 mbar | No — drifts up as filter gets dirty. Reset by filter replacement. |
| **CDP** | Compressor Discharge Pressure (bar) | 10-15 bar | No — consequence of mass flow + compressor design |
| **TIT** | Turbine Inlet Temperature (°C) | 1001-1101°C | **YES — primary control knob** (set via fuel flow rate) |
| **TAT** | Turbine After Temperature (°C) | 511-551°C | No — consequence of TIT and expansion ratio |
| **GTEP** | Gas Turbine Exhaust Pressure (mbar) | 18-41 mbar | No — set by downstream equipment (HRSG, stack) |
| **TEY** | Turbine Energy Yield (MWh) | 100-180 MWh | Semi — set by grid demand, achieved by adjusting TIT |

### Emissions (outcomes)

| Variable | What it is | Range |
|----------|-----------|-------|
| **CO** | Carbon monoxide (mg/m³) | 0-44 mg/m³ |
| **NOx** | Nitrogen oxides (mg/m³) | 26-120 mg/m³ |

---

## Emission Chemistry: The CO-NOx Tradeoff

This is the most important relationship in the dataset.

### NOx: Thermal Formation (Zeldovich Mechanism)

NOx forms when N₂ in the air reacts with O at extreme temperatures:

```
N₂ + O → NO + N     (rate-limiting, activation energy ~75 kcal/mol)
N + O₂ → NO + O     (fast)
```

- Below ~1527°C: essentially no thermal NOx
- Above ~1527°C: NOx increases **exponentially** with temperature (Arrhenius kinetics)
- More pressure → more reactant concentration → more NOx
- More time in the hot zone → more NOx

### CO: Incomplete Combustion

CO is an intermediate product. The final step CO + OH → CO₂ + H fails when:

- **Temperature too low** — below ~1230°C the reaction rate drops off
- **Residence time too short** — gas exits before CO fully oxidizes
- **Poor mixing** — locally fuel-rich pockets produce CO that can't fully burn out

### The Tradeoff

```
        CO emissions                    NOx emissions
        ↑                                         ↑
        │  ╲                                 ╱    │
        │    ╲                             ╱      │
        │      ╲                         ╱        │
        │        ╲     OPTIMAL         ╱          │
        │          ╲    WINDOW       ╱            │
        │            ╲─────────────╱              │
        └──────────────────────────────────────→
                    Flame Temperature
```

- **Hot flame** → complete combustion (low CO) but lots of NOx
- **Cool flame** → little NOx but incomplete combustion (high CO)
- **You cannot minimize both.** There is a narrow optimal window.

Modern Dry Low Emission (DLE) combustors try to stay in this window by running lean-premixed combustion, keeping flame temp below ~1538°C.

---

## Causal Structure: What Causes What

### Variable Classification

```
EXOGENOUS (weather):     AT, AP, AH
                              ↓
QUASI-EXOGENOUS:         AFDP (filter condition), GTEP (downstream equipment)
                              ↓
DEMAND (unobserved):     Grid demand → TEY (semi-exogenous)
                              ↓
CONTROL RESPONSE:        Fuel flow (unobserved) → TIT
                              ↓
MECHANICAL CONSEQUENCES: CDP, TAT
                              ↓
EMISSIONS:               CO, NOx
```

### Key Causal Pathways

1. **AT → air density → mass flow → CDP, TEY, combustion conditions → CO, NOx**
   Hotter day → less air mass → turbine compensates → different emissions.

2. **AP → air density → mass flow → CDP, TEY, combustion conditions → CO, NOx**
   Same mechanism as AT, through pressure.

3. **AH → flame temperature → NOx** (direct chemical effect, strong)
   Water vapor absorbs heat → cooler flame → less NOx. This is a direct effect on combustion chemistry, separate from the (weak) density effect.

4. **AH → AFDP** (humidity increases filter pressure drop)

5. **TIT → flame temperature → NOx (positive, exponential)**
   More fuel → hotter combustion → more NOx.

6. **TIT → flame temperature → CO (negative)**
   More fuel → hotter combustion → more complete → less CO.

7. **AFDP → effective inlet pressure → mass flow → everything downstream**
   Dirty filter → restricted airflow → different combustion → different emissions.

8. **TIT → TAT** (thermodynamic consequence)
9. **TIT → TEY** (more energy in → more power out)
10. **CDP, GTEP → expansion ratio → TAT, TEY**

### Key Confounders

- **AT confounds TIT↔emissions**: Hot days correlate with high grid demand → higher TIT set. Hot days also independently change combustion conditions through air density. Naive regression of emissions on TIT without controlling for AT gives biased estimates.

- **AP confounds CDP↔emissions**: AP directly scales CDP. Omitting AP makes CDP look like it has a different effect than it really does.

- **AH has dual pathways**: Affects both air density (weak) AND flame chemistry (strong). Must be conditioned on to isolate other effects.

- **AFDP confounds air flow and combustion**: Changes effective inlet pressure → changes everything downstream. Also correlated with time and humidity.

### Unobserved Variables (not in dataset)

- **Fuel flow rate** — the direct lever that sets TIT
- **Inlet Guide Vane (IGV) position** — modulates airflow at part load
- **Grid demand / load setpoint** — exogenous driver of TEY
- **Fuel composition** — natural gas composition varies

### Important: CO and NOx Are NOT Causally Related to Each Other

Their negative correlation exists because they share a common cause (flame temperature) that has opposite effects on each. They are joint effects, not cause and effect.

---

## Questions This Dataset Can Answer Causally

1. **"If we increase TIT by 20°C, what happens to NOx — controlling for ambient conditions?"**
   ATE of TIT on NOx. Must control for AT, AP, AH (confounders).

2. **"Does the effect of TIT on NOx vary by ambient temperature?"**
   Heterogeneous treatment effect. Causal forests / X-learner territory.

3. **"What would emissions have been on a hot day if conditions had been like a cold day?"**
   Counterfactual reasoning.

4. **"How much of TIT's effect on NOx goes through flame temperature vs. through changed pressure ratios?"**
   Mediation analysis.

5. **"If we could intervene on AFDP (replace filter), what's the causal effect on emissions?"**
   Actionable insight for maintenance scheduling.
