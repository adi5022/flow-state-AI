# CKD Stage Encoding

## Overview

CKD (Chronic Kidney Disease) Stage Encoding is the process of categorizing kidney function into discrete stages based on eGFR (estimated glomerular filtration rate). This categorization is used throughout the system to capture the nonlinear relationship between kidney function and disease progression.

The system uses a simplified 3-stage encoding:

- **Stage 0**: eGFR ≥ 60 ml/min/1.73m² (normal or mildly reduced)
- **Stage 3**: eGFR 30-59 ml/min/1.73m² (moderate to severe CKD)
- **Stage 4**: eGFR < 30 ml/min/1.73m² (severe CKD)

This encoding is used as a feature in the eGFR progression models and the mortality model to capture the threshold effects of kidney disease on health outcomes.

## Scientific / Theoretical Basis

### Why Categorize CKD?

Kidney function decline is not linear - patients with established CKD decline faster than healthy people, and the decline accelerates as kidney function worsens. By categorizing eGFR into stages, the models can capture these threshold effects.

The staging is based on the KDIGO (Kidney Disease: Improving Global Outcomes) guidelines, which are the international standard for CKD classification [1].

### KDIGO CKD Staging

The full KDIGO classification has 5 stages (G1-G5), but the system uses a simplified 3-stage encoding for the models:

```
KDIGO Stage | eGFR Range          | Simplified Encoding
-----------|---------------------|---------------------
G1-G2      | ≥ 90 or 60-89       | Stage 0
G3         | 30-59               | Stage 3
G4-G5      | 15-29 or < 15       | Stage 4
```

The simplified encoding groups G1-G2 together (normal to mildly reduced) and G4-G5 together (severe to end-stage), focusing on the key clinical thresholds.

### Why These Thresholds?

The thresholds are based on clinical evidence:

- **eGFR ≥ 60**: Below this level, cardiovascular risk increases significantly [2]
- **eGFR 30-59**: Moderate CKD - patients have accelerated decline and higher mortality [2]
- **eGFR < 30**: Severe CKD - rapid decline, high mortality, approaching end-stage renal disease [2]

### Nonlinear Decline Rates

Research shows that CKD stage affects decline rate:

- **Stage 0**: Decline follows normal age-related rates (Glassock rates)
- **Stage 3**: Decline is about 1.6x faster than normal [3]
- **Stage 4**: Decline is about 2.2x faster than normal [3]

This nonlinear relationship is why the encoding is important - a linear model using raw eGFR values would miss these threshold effects.

## Data Structures

### Encoding Function

The CKD stage is calculated from eGFR using a simple conditional:

```python
def ckd_stage(egfr):
    if egfr >= 60:
        return 0
    elif egfr >= 30:
        return 3
    else:
        return 4
```

### Stage Multipliers

For the synthetic eGFR cohort generation, stage-specific multipliers are applied:

```
CKD_MULTIPLIER = {
    0: 1.00,   # Normal rate
    3: 1.60,   # 1.6x faster decline
    4: 2.20    # 2.2x faster decline
}
```

## Step-by-Step Process

### Step 1: Calculate Current eGFR

Each year of the simulation, the system calculates the current eGFR based on the eGFR progression model:

```
egfr_t = max(0, egfr_t - egfr_decline)
```

### Step 2: Determine CKD Stage

The system categorizes the eGFR into a stage:

```
if egfr_t >= 60:
    ckd_stage = 0
elif egfr_t >= 30:
    ckd_stage = 3
else:
    ckd_stage = 4
```

**Example**: eGFR = 75 → ckd_stage = 0

**Example**: eGFR = 45 → ckd_stage = 3

**Example**: eGFR = 25 → ckd_stage = 4

### Step 3: Use CKD Stage as Feature

The CKD stage is used as a feature in the eGFR progression model:

```
egfr_feat = [age, sex, egfr, MAP, pack_years, sodium, egfr_baseline, ckd_stage]
residual = egfr_model.predict(egfr_feat)
```

The Gradient Boosting model learns that higher CKD stage values predict faster eGFR decline.

### Step 4: Apply Stage-Specific Effects (Synthetic Cohort)

In the synthetic cohort generation, the CKD stage multiplier is applied to the base decline rate:

```
base_rate = glassock_rate(age) × CKD_MULTIPLIER[ckd_stage]
```

**Example**: 60-year-old with eGFR = 45 (Stage 3):

```
glassock_rate(60) = 1.24 ml/min/1.73m²/year
CKD_MULTIPLIER[3] = 1.60
base_rate = 1.24 × 1.60 = 1.98 ml/min/1.73m²/year
```

This patient declines 1.6x faster than a healthy 60-year-old.

## Inputs and Outputs

### Inputs

The CKD Stage Encoding takes:

- **egfr**: Current kidney function in ml/min/1.73m²

### Outputs

The encoding produces:

- **ckd_stage**: Integer value (0, 3, or 4) representing the CKD category

## Design Decisions

### Why Skip Stages 1 and 2?

The full KDIGO classification includes stages 1 (eGFR ≥ 90) and 2 (eGFR 60-89). We group these together as Stage 0 because:

- The clinical distinction between G1 and G2 is less important for long-term outcomes
- The key threshold is eGFR < 60, where cardiovascular risk increases significantly
- Simplifying to 3 stages reduces model complexity without losing predictive power

### Why Use Integer Encoding Instead of One-Hot Encoding?

We use integer encoding (0, 3, 4) instead of one-hot encoding (binary columns for each stage) because:

- Gradient Boosting models can handle categorical integers effectively
- The encoding preserves the ordinal relationship (higher stage = worse kidney function)
- It's more memory-efficient than one-hot encoding

### Why Not Use Raw eGFR?

Using raw eGFR values instead of categorical stages would miss the threshold effects. The relationship between eGFR and decline is nonlinear - a patient with eGFR = 58 declines much faster than a patient with eGFR = 62, even though the difference is only 4 ml/min/1.73m². Categorical staging captures this discontinuity.

## Integration with Other Subsystems

The CKD Stage Encoding connects to:

- **eGFR Progression Model**: CKD stage is a feature in the Gradient Boosting residual model, capturing nonlinear acceleration
- **Mortality Model**: The mortality model uses categorical eGFR hazard ratios based on CKD stage (not a continuous eGFR term)
- **Synthetic eGFR Cohort Generation**: The synthetic cohort uses CKD stage multipliers to generate realistic decline patterns
- **Intervention System**: The discontinuation threshold for ACE inhibitors uses eGFR < 30 (Stage 4) as a safety criterion

The CKD stage encoding ensures that the system captures the clinically important threshold effects of kidney disease on health outcomes.

## References

[1] KDIGO 2012 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease. Kidney International Supplements, 3(1), 1-150.

[2] Go, A. S., et al. (2004). "Chronic kidney disease and the risks of death." New England Journal of Medicine, 351(13), 1296-1305.

[3] Go, A. S., et al. (2004). "Chronic kidney disease and the risks of death, cardiovascular events, and hospitalization." New England Journal of Medicine, 351(13), 1296-1305.
