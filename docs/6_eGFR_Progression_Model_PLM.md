# eGFR Progression Model (PLM Training)

## Overview

The PLM (Partial Linear Model) training process creates the Gradient Boosting residual model that is used in the eGFR progression simulation. This is a two-layer model where:

- **Layer 1** is an explicit linear prediction based on published research (Glassock age rates, sex differences, sodium effects)
- **Layer 2** is a machine learning model that predicts the residual - the difference between actual decline and the linear prediction

The training script `train_egfr_plm.py` trains the Layer 2 residual model. The Layer 1 linear terms are hardcoded based on published research and are not trained.

This hybrid approach gives us the best of both worlds: the linear layer is scientifically grounded and interpretable, while the residual layer captures complex patterns that simple equations cannot represent.

## Scientific / Theoretical Basis

### Why a Partial Linear Model?

Traditional machine learning models for medical prediction can be "black boxes" - they make predictions but it's hard to explain why. The PLM approach addresses this by separating the prediction into two parts:

1. **Explicit Layer 1**: Terms we can explain and cite directly from research papers
2. **Residual Layer 2**: A machine learning model that captures what Layer 1 doesn't explain

This makes the model more transparent. If someone asks "why did you predict this eGFR decline?", we can point to the specific research citations for Layer 1, and explain that Layer 2 captures additional complexity.

### Training Data Source

The residual model is trained on a synthetic eGFR cohort generated specifically for PLM training. This cohort is located at `data/egfr_cohort_plm/cohort_plm.csv`.

The synthetic cohort contains:

- Patient demographics (age, sex)
- Current eGFR
- MAP (Mean Arterial Pressure)
- Pack-years of smoking
- Daily sodium intake
- Baseline eGFR (starting value)
- CKD stage (0, 3, or 4)
- The actual eGFR decline (target)
- The Layer 1 linear prediction
- The residual (actual - linear prediction)

The residual model learns to predict this residual from the patient characteristics.

### Layer 1 Linear Terms (Not Trained)

Layer 1 is hardcoded based on published research and is the same in both the training script and the simulation:

**Glassock age rates** [1]:

```
Age 0-39:   0.00 ml/min/1.73m²/year
Age 40-49:  0.32
Age 50-59:  0.57
Age 60-69:  1.24
Age 70-79:  1.49
Age 80+:    3.25
```

**Sex difference** [2]:

```
Male:   +0.11 ml/min/1.73m²/year
Female: -0.11 ml/min/1.73m²/year
```

**Note on coefficient discrepancy:** Eriksen & Ingebretsen (2006) Table 5 reports a female gender effect of 0.50 ml/min/year slower decline than men (95% CI 0.20-0.81, P=0.001). However, the code uses a smaller value of 0.22 (±0.11). This discrepancy was investigated but left unchanged because updating to 0.50 caused the trained residual model to produce incorrect sex differences (females declining faster than males, opposite of expected). The synthetic cohort generation and residual model training are sensitive to this parameter, and the current value produces more realistic overall model behavior despite the citation discrepancy.

**Sodium effect** [3]:

```
For each mg/day above 2300 mg/day: +0.0001 ml/min/1.73m²/year
```

The linear prediction is:

```
linear_prediction = glassock_rate(age) + sex_delta + sodium_effect
```

### Layer 2 Residual Model (Trained)

The residual model is a Gradient Boosting regressor that predicts:

```
residual = actual_decline - linear_prediction
```

The residual model uses these features:

```
FEATURES = [
    "age",             # Patient age
    "sex",             # 1 for male, 2 for female
    "egfr",            # Current eGFR
    "map_mmhg",        # Mean arterial pressure
    "pack_years",      # Cumulative smoking exposure
    "sodium_mg_day",   # Daily sodium intake
    "egfr_baseline",   # Baseline eGFR at start
    "ckd_stage"        # 0, 3, or 4
]
```

Note that even though age, sex, and sodium are in the residual model features, they should have LOW importance because Layer 1 already handles them explicitly. The residual model should focus on CKD stage, smoking, and other factors not captured by Layer 1.

## Data Structures

### Training Data

The training data is a CSV file with one row per patient visit. Key columns include:

- **age**: Patient age
- **sex**: 1 for male, 2 for female
- **egfr**: Current eGFR
- **map_mmhg**: Mean arterial pressure
- **pack_years**: Cumulative smoking exposure
- **sodium_mg_day**: Daily sodium intake
- **egfr_baseline**: Baseline eGFR
- **ckd_stage**: CKD category (0, 3, or 4)
- **residual_decline**: Target variable (actual decline - linear prediction)

### Trained Model

The output is a Gradient Boosting model saved as:

- **progression_egfr_plm.pkl**: The trained residual model

This model is loaded in main.py and used in the simulation loop.

## Step-by-Step Process

### Step 1: Load Training Data

The script loads the synthetic eGFR cohort:

```python
df = pd.read_csv("data/egfr_cohort_plm/cohort_plm.csv")
```

### Step 2: Prepare Features and Target

The script extracts the features and the target (residual decline):

```python
X = df[FEATURES].values
y = df["residual_decline"].values
```

### Step 3: Train Gradient Boosting Model

The script trains a Gradient Boosting regressor with these hyperparameters:

```python
model = GradientBoostingRegressor(
    n_estimators=100,      # Number of trees
    max_depth=4,           # Depth of each tree
    learning_rate=0.10,    # How much each tree contributes
    subsample=0.80,        # Use 80% of data for each tree
    min_samples_leaf=20,   # Minimum samples per leaf
    random_state=42        # For reproducibility
)
```

The model is trained with 5-fold cross-validation to ensure it generalizes to new data.

### Step 4: Validate on Benchmarks

The script validates the full PLM (Layer 1 + Layer 2) against published research benchmarks:

**Benchmark 1 - Glassock age rates**: For healthy patients at different ages, the PLM should match the published Glassock rates

**Benchmark 2 - Sex difference**: Men should decline about 0.22 ml/min/1.73m²/year faster than women

**Benchmark 3 - CKD stage acceleration**: Stage 3 patients should decline faster than normal patients, Stage 4 faster than Stage 3

**Benchmark 4 - Smoking effect**: Smokers with 20 pack-years should decline about 30% faster than non-smokers

### Step 5: Check Feature Importance

The script reports which features are most important in the residual model. As expected, CKD stage and smoking should have high importance, while age/sex/sodium should have low importance (since Layer 1 already handles them).

### Step 6: Save Model

The trained model is saved to `models/progression_egfr_plm.pkl`.

## Inputs and Outputs

### Inputs

The training script takes:

- **Synthetic eGFR cohort**: CSV file at `data/egfr_cohort_plm/cohort_plm.csv`
- **Layer 1 functions**: Hardcoded Glassock rates, sex delta, and sodium coefficient

### Outputs

The script produces:

- **Trained residual model**: Saved as `models/progression_egfr_plm.pkl`
- **Validation metrics**: CV R², in-sample R², MAE, RMSE
- **Benchmark results**: Comparison against published research values
- **Feature importance**: Which features drive the residual predictions

## Design Decisions

### Why Train Only the Residual?

By training only the residual (actual - linear prediction), we ensure that:

- Layer 1 terms are always exactly as specified in the literature
- The machine learning model only learns what the linear terms don't capture
- The model remains scientifically grounded

If we trained a single model to predict the total decline directly, it might "unlearn" the well-established relationships from Layer 1.

### Why Gradient Boosting for the Residual?

Gradient Boosting is well-suited for this task because:

- It can capture nonlinear interactions (e.g., CKD stage × smoking)
- It handles mixed data types well
- It's robust to outliers
- It performs well on tabular medical data

### Why Remove MAP from Explicit Terms?

Earlier versions included MAP in the linear layer. However, two major studies found that MAP does not independently predict eGFR decline after controlling for other factors [4, 5]. Based on this evidence, we removed the MAP term from Layer 1. MAP is still in the residual model features, but its importance should be low.

### Why Separate Training from Simulation?

The training script (`train_egfr_plm.py`) is separate from the simulation (`main.py`) because:

- Training only needs to happen once (or when the cohort is updated)
- Simulation happens every time the digital twin runs
- This keeps the simulation code cleaner and faster

## Integration with Other Subsystems

The PLM training process connects to:

- **eGFR Progression Model (PLM) (Subsystem 5)**: The model trained here is used in the simulation to predict eGFR decline
- **Synthetic eGFR Cohort Generation (Subsystem 7)**: The training data comes from the synthetic eGFR cohort
- **Model Training Pipeline (Subsystem 10)**: This is part of the broader model training pipeline

The trained model (`progression_egfr_plm.pkl`) is loaded in main.py and used in the simulation loop alongside the Layer 1 linear functions to make full PLM predictions.

## References

[1] Glassock, R. J., & Winearls, C. G. (2009). "Ageing and the glomerular filtration rate: truths and consequences." Transactions of the American Clinical and Climatological Association, 120, 419-428.

[2] Eriksen, B. O., & Ingebretsen, O. C. (2006). "The progression of chronic kidney disease: a 10-year population-based study of the effects of gender and age." Kidney International, 69(2), 375-382.

[3] Stolarz-Skrzypek, K., et al. (2011). "Fatal and nonfatal outcomes, incidence of hypertension, and blood pressure changes in relation to urinary sodium excretion." JAMA, 305(17), 1777-1785.

[4] Eriksen, B. O., et al. (2017). "Blood pressure and age-related GFR decline in the general population." BMC Nephrology, 18, 77.

[5] Wright, J. T., Jr., et al. (2002). "Effect of blood pressure lowering and antihypertensive drug class on progression of hypertensive kidney disease: results from the AASK trial." JAMA, 288(19), 2421-2431.
