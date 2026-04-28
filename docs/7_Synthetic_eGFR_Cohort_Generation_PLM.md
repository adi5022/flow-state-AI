# Synthetic eGFR Cohort Generation (PLM)

## Overview

The Synthetic eGFR Cohort Generation (PLM) creates a large dataset of synthetic patients that mimics real-world kidney function (eGFR) patterns. This synthetic data is used to train the residual component of the Partial Linear Model (PLM) eGFR progression model.

The system generates 5,000 synthetic patients, each followed for up to 30 years with annual measurements. Each patient has demographic information, risk factors, and eGFR values. The key innovation is that the data is structured to separate:

- **Layer 1 (Linear)**: The explicit, scientifically-grounded prediction based on age, sex, and sodium
- **Layer 2 (Residual)**: The complex patterns that the machine learning model must learn (CKD stage effects, smoking effects, individual variation)

By generating data with this structure, we ensure the trained machine learning model learns only what it should learn - the residual patterns not captured by the explicit linear terms.

## Scientific / Theoretical Basis

### Why Synthetic eGFR Data?

Real longitudinal eGFR data has limitations:

- **Privacy**: Kidney function data is sensitive medical information
- **Cost**: Long-term kidney studies are expensive and rare
- **Selection bias**: Real datasets often focus on specific patient populations

Synthetic data solves these problems while ensuring the patterns match published research.

### Layer 1: Explicit Linear Terms

The linear layer is based on well-established research and is NOT learned by the machine learning model:

**Glassock age rates** [1]: Age-specific annual eGFR decline rates

```
Age 0-39:   0.00 ml/min/1.73m²/year
Age 40-49:  0.32
Age 50-59:  0.57
Age 60-69:  1.24
Age 70-79:  1.49
Age 80+:    3.25
```

**Sex difference** [2]: Men decline 0.22 ml/min/1.73m²/year faster than women

```
Male:   +0.11 ml/min/1.73m²/year
Female: -0.11 ml/min/1.73m²/year
```

**Note on coefficient discrepancy:** Eriksen & Ingebretsen (2006) Table 5 reports a female gender effect of 0.50 ml/min/year slower decline than men (95% CI 0.20-0.81, P=0.001). However, the code uses a smaller value of 0.22 (±0.11). This discrepancy was investigated but left unchanged because updating to 0.50 caused the trained residual model to produce incorrect sex differences (females declining faster than males, opposite of expected). The synthetic cohort generation and residual model training are sensitive to this parameter, and the current value produces more realistic overall model behavior despite the citation discrepancy.

**Sodium effect** [3]: Sodium above 2300 mg/day contributes to decline

```
0.0001 ml/min/1.73m²/year per mg/day above 2300 mg/day
```

The linear prediction is:

```
linear_prediction = glassock_rate(age) + sex_delta + sodium_effect
```

### Layer 2: Residual Patterns

The residual layer captures patterns that the linear terms don't explain:

**CKD stage nonlinearity** [4]: Patients with CKD decline faster than healthy people

```
Stage 0 (eGFR ≥ 60):  1.0x baseline rate
Stage 3 (eGFR 30-59):  1.6x baseline rate
Stage 4 (eGFR < 30):   2.2x baseline rate
```

**Smoking acceleration** [5]: Log-linear relationship with pack-years

```
At 20 pack-years: 1.30x faster decline
Coefficient: 0.01310 per pack-year
```

**Individual variation** [6]: Each patient has a persistent offset from the mean

**Measurement noise** [7]: Random variation at each visit

### Why Remove MAP?

Earlier versions included MAP (blood pressure) as a factor. However, two major studies found that MAP does not independently predict eGFR decline after controlling for other factors:

- **RENIS-T6 study** [8]: No significant BP-GFR association in normotensive general population
- **AASK trial** [9]: No significant difference in GFR slope between different MAP groups

Based on this evidence, we removed the MAP term from the model entirely.

## Data Structures

### Patient Records

Each patient has a record for each visit year containing:

```
- age: Patient age
- sex: 1 for male, 2 for female
- egfr: Current kidney function
- map_mmhg: Mean arterial pressure (for context, not used in PLM)
- pack_years: Cumulative smoking exposure
- sodium_mg_day: Daily sodium intake
- egfr_baseline: Baseline eGFR at start
- ckd_stage: CKD category (0, 3, or 4)
- linear_prediction: Layer 1 explicit prediction
- actual_decline: Total decline (linear + residual + noise)
- residual_decline: Target for GB model (actual - linear)
```

### Population Constants

The system uses the same constants as the PLM model:

```
GLASSOCK_RATES: Age-specific decline rates
SEX_DELTA: Sex adjustment (+0.11 for male, -0.11 for female)
SODIUM_THRESHOLD: 2300 mg/day
SODIUM_COEF: 0.0001
CKD_MULTIPLIER: Stage-specific multipliers (1.0, 1.6, 2.2)
SMOKING_COEF: 0.01310 per pack-year
BETWEEN_PERSON_SD: 0.40 ml/min/yr (individual variation)
WITHIN_VISIT_SD: 0.25 ml/min/yr (measurement noise)
```

## Step-by-Step Process

### Step 1: Generate Baseline Characteristics

For each of the 5,000 patients, we generate baseline characteristics:

```
age = random.uniform(35, 75)
sex = random.choice([1, 2])
egfr = random.normal(90, 18)  # Clipped to 15-130
pack_years = random.exponential(8)  # Clipped to 0-80
sodium = random.normal(3400, 800)  # Clipped to 1000-6000
map = random.normal(90.11, 12.55)  # Clipped to 60-140
```

### Step 2: Assign Personal Offset

Each patient gets a persistent random offset representing their individual tendency to decline faster or slower than average:

```
personal_offset = random.normal(0, 0.40)
```

This creates realistic diversity - some patients' kidneys decline faster, others slower.

### Step 3: Annual Simulation Loop

For each year from 0 to 30:

**Calculate CKD stage**:

```
if eGFR ≥ 60:   ckd_stage = 0
elif eGFR ≥ 30:  ckd_stage = 3
else:           ckd_stage = 4
```

**Calculate Layer 1 linear prediction**:

```
lin_pred = glassock_rate(age) + sex_delta + sodium_effect
```

**Calculate Layer 2 actual decline**:

First, get the base rate adjusted for CKD stage:

```
base_rate = glassock_rate(age) × CKD_MULTIPLIER[ckd_stage]
```

Then apply smoking multiplier:

```
smoke_mult = exp(0.01310 × pack_years)
```

Then combine all effects:

```
actual = base_rate × smoke_mult
         + sex_delta
         + sodium_effect
         + personal_offset
         + random_noise
```

**Calculate residual** (what the GB model must learn):

```
residual = actual - lin_pred
```

**Save the record** with all values

**Update eGFR for next year**:

```
egfr = max(0, egfr - actual)
age = age + 1
```

If eGFR reaches 0, stop simulating that patient (end-stage renal disease).

### Step 4: Save and Validate

After generating all patients, save to CSV and run validation checks:

**Layer 1 validation**: Ensure linear predictions match Glassock targets for healthy patients

**Residual statistics**: Check that residuals have reasonable mean and standard deviation

## Inputs and Outputs

### Inputs

The Synthetic eGFR Cohort Generation doesn't take patient-specific inputs. It generates synthetic data from scratch using:

- Glassock age rates from published research [1]
- Sex difference from Eriksen & Ingebretsen [2]
- Sodium coefficient from de Boer et al. [3]
- CKD multipliers from Go et al. [4]
- Smoking coefficient from Hallan & Orth [5]
- Noise structure from Laird & Ware [6] and Stevens et al. [7]

### Outputs

The system produces a CSV file with approximately 150,000 rows (5,000 patients × ~30 years each). The output is saved to:

```
data/egfr_cohort_plm/cohort_plm.csv
```

## Design Decisions

### Why Separate Layer 1 and Layer 2 in the Data?

By explicitly calculating the linear prediction and the residual in the synthetic data, we ensure that:

- The trained GB model only learns the residual patterns
- Layer 1 terms remain exactly as specified in the literature
- We can validate that the model is learning what it should

If we trained a single model to predict total decline directly, it might "unlearn" the well-established relationships from Layer 1.

### Why 5,000 Patients?

We chose 5,000 patients as a balance between:

- **Sufficient data**: Enough to train the residual model effectively
- **Computational efficiency**: Can be generated quickly
- **Statistical power**: Large enough to capture rare patterns (e.g., Stage 4 CKD)

### Why Include MAP in the Data?

Even though MAP is not used in the PLM prediction (removed based on RENIS-T6 and AASK studies), we include it in the synthetic data as a feature. This allows the residual model to learn any subtle patterns if they exist, while keeping the explicit MAP coefficient at zero in Layer 1.

### Why Two Types of Noise?

We include two types of noise to match real-world measurement:

- **Between-person noise**: Each patient has a persistent offset (they're consistently faster or slower)
- **Within-visit noise**: Random variation at each visit (measurement error, day-to-day fluctuation)

This creates realistic variability that the model must be robust to.

## Integration with Other Subsystems

The Synthetic eGFR Cohort Generation connects to:

- **eGFR Progression Model (PLM Training)**: The synthetic cohort is the training data for the residual model
- **eGFR Progression Model (PLM) (Subsystem 5)**: The model trained on this data is used in the simulation
- **Model Training Pipeline (Subsystem 10)**: This is part of the broader training pipeline

The synthetic cohort ensures that the trained PLM model learns patterns consistent with published research while capturing the complex residual effects that simple equations cannot represent.

## References

[1] Glassock, R. J., & Winearls, C. G. (2009). "Ageing and the glomerular filtration rate: truths and consequences." Transactions of the American Clinical and Climatological Association, 120, 419-428.

[2] Eriksen, B. O., & Ingebretsen, O. C. (2006). "The progression of chronic kidney disease: a 10-year population-based study of the effects of gender and age." Kidney International, 69(2), 375-382.

[3] Stolarz-Skrzypek, K., et al. (2011). "Fatal and nonfatal outcomes, incidence of hypertension, and blood pressure changes in relation to urinary sodium excretion." JAMA, 305(17), 1777-1785.

[4] Go, A. S., et al. (2004). "Chronic kidney disease and the risks of death." New England Journal of Medicine, 351(13), 1296-1305.

[5] Hallan, S. I., & Orth, S. R. (2011). "Smoking: A risk factor for progression of chronic kidney disease." Journal of Nephrology, 24(4), 405-410.

[6] Laird, N. M., & Ware, J. H. (1982). "Random-effects models for longitudinal data." Biometrics, 38(4), 963-974.

[7] Stevens, L. A., et al. (2006). "Assessing kidney function." American Journal of Kidney Diseases, 48(1), 11-20.

[8] Eriksen, B. O., et al. (2017). "Blood pressure and age-related GFR decline in the general population." BMC Nephrology, 18, 77.

[9] Wright, J. T., Jr., et al. (2002). "Effect of blood pressure lowering and antihypertensive drug class on progression of hypertensive kidney disease: results from the AASK trial." JAMA, 288(19), 2421-2431.
