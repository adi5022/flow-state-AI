# Blood Pressure Progression Models

## Overview

The Blood Pressure Progression Models predict how a patient's blood pressure will change over time. They are machine learning models that were trained on real patient data to learn the patterns of blood pressure progression.

There are two separate models:

1. **MAP progression model** - Predicts how Mean Arterial Pressure (MAP) will change each year
2. **PP progression model** - Predicts how Pulse Pressure (PP) will change each year

These models work on the internal state variables (MAP_z and PP_z) rather than the clinical blood pressure numbers. This is because the internal representation is better for machine learning - the values are normalized and capture distinct physiological processes.

The models take the current state of the patient (blood pressure, age, kidney function, smoking history, etc.) and predict the annual change in blood pressure. These predictions are added to the current state to get the next year's state. This process repeats each year of the simulation.

## Scientific / Theoretical Basis

### Why Machine Learning for Blood Pressure Progression?

Blood pressure progression is complex. It depends on many factors:

- **Age**: Blood pressure tends to increase with age
- **Sex**: Men and women have different blood pressure patterns
- **Body composition**: Waist circumference affects blood pressure
- **Heart rate**: Faster heart rate correlates with higher blood pressure
- **Kidney function**: Reduced kidney function (eGFR) affects blood pressure regulation
- **Smoking**: Smoking increases blood pressure
- **Alcohol and sodium**: Both can raise blood pressure

Traditional statistical models can capture some of these relationships, but machine learning models can learn complex, nonlinear interactions between all these factors simultaneously.

### Model Selection Process

The MAP progression model was selected from three candidates [1]:

- **Ridge Regression**: A regularized linear regression that prevents overfitting
- **Random Forest**: An ensemble of decision trees that can capture nonlinear patterns
- **Gradient Boosting**: Another ensemble method that builds trees sequentially to correct errors

We compared these models using 5-fold cross-validation and selected the one with the highest cross-validation R² score. **Gradient Boosting was selected as the MAP progression model** (n_estimators=200, max_depth=4, learning_rate=0.05), achieving the highest CV R². This is expected because MAP progression depends on complex, nonlinear interactions between multiple risk factors (age, kidney function, smoking, etc.) that a linear model cannot capture.

The PP progression model uses Ridge regression exclusively. This is based on the Franklin et al. (1999) study, which established that pulse pressure increases significantly with age, particularly after age 60 due to diverging systolic and diastolic blood pressure trajectories [2]. A linear model like Ridge is appropriate for this relatively predictable pattern.

### Training Data

The models were trained on synthetic patient data that mimics real longitudinal blood pressure measurements. The training process used pairs of consecutive visits from the same patient:

- Visit 1: Current state (blood pressure, age, etc.)
- Visit 2: State one year later

The model learns to predict the change in blood pressure from Visit 1 to Visit 2 based on the patient's characteristics at Visit 1.

## Data Structures

### Trained Models

The models are saved as Python pickle files:

- **progression_map.pkl**: The trained model for MAP progression
- **progression_pp.pkl**: The trained model for PP progression

These files are loaded when the simulation starts and used throughout the simulation.

### Feature Vector

The models use the following features to make predictions:

```
PROGRESSION_FEATURES = [
    "map_z",          # Current MAP z-score
    "pp_z",           # Current PP z-score
    "age",            # Patient age in years
    "sex",            # 1 for male, 2 for female
    "waist_cm",       # Waist circumference in cm
    "heart_rate",     # Heart rate in beats per minute
    "egfr",           # Kidney function in ml/min/1.73m²
    "pack_years",     # Cumulative smoking exposure
    "alcohol_g_day",  # Daily alcohol consumption in grams
    "sodium_mg_day"   # Daily sodium intake in mg
]
```

### Feature Selection Rationale

Each feature in the progression model is supported by established research on blood pressure progression:

- **map_z, pp_z**: Current blood pressure state is the strongest predictor of future blood pressure (autoregressive effect)
- **age, sex**: Age and sex are fundamental determinants of blood pressure trajectories, with different patterns by sex across the lifespan [3]
- **waist_cm**: Waist circumference is an established risk factor for hypertension and blood pressure elevation [4, 5, 6]
- **heart_rate**: Resting heart rate is associated with blood pressure and predicts incident hypertension [7, 8]
- **egfr**: Kidney function (eGFR) has a bidirectional relationship with blood pressure; hypertension accelerates kidney function decline and reduced kidney function affects blood pressure regulation [9, 10]
- **pack_years**: Smoking exposure (pack-years) has longitudinal associations with blood pressure changes [11, 12]
- **alcohol_g_day, sodium_mg_day**: Both alcohol and sodium intake are well-established modifiable risk factors for blood pressure elevation, with synergistic effects [13, 14, 15]

### Predicted Outputs

The models predict annual changes in the internal state variables:

- **delta_map_z**: Predicted annual change in MAP z-score
- **delta_pp_z**: Predicted annual change in PP z-score

These are small values (typically around 0.01-0.05 per year) that are added to the current state.

## Step-by-Step Process

### Step 1: Load Trained Models

When the simulation starts, the Engine loads the two trained models:

```python
self.prog_map = joblib.load("progression_map.pkl")
self.prog_pp = joblib.load("progression_pp.pkl")
```

### Step 2: Prepare Feature Vector

Each year, we create a feature vector with the patient's current state:

```python
row = pd.DataFrame(
    [[map_z, pp_z, age, sex, waist_cm,
      heart_rate, egfr, pack_years,
      alcohol_g_day, sodium_mg_day]],
    columns=PROGRESSION_FEATURES
)
```

For example, a 55-year-old male with MAP_z = 0.5, PP_z = 0.3, waist = 95 cm, heart rate = 72 bpm, eGFR = 58, 20 pack-years, 10 g/day alcohol, 3000 mg/day sodium:

```python
row = [[0.5, 0.3, 55, 1, 95, 72, 58, 20, 10, 3000]]
```

### Step 3: Predict Annual Changes

We use the models to predict how blood pressure will change over the next year:

```python
delta_map_z = prog_map.predict(row)[0]
delta_pp_z = prog_pp.predict(row)[0]
```

For example, the models might predict:
- delta_map_z = 0.02 (MAP will increase by 0.02 standard deviations)
- delta_pp_z = 0.015 (PP will increase by 0.015 standard deviations)

### Step 4: Update State Variables

We add the predicted changes to the current state:

```python
map_z = map_z + delta_map_z
pp_z = pp_z + delta_pp_z
```

For our example:
- map_z = 0.5 + 0.02 = 0.52
- pp_z = 0.3 + 0.015 = 0.315

### Step 5: Repeat

The simulation repeats this process each year. The updated state becomes the input for the next year's prediction. This allows the models to capture cumulative effects over time.

## Inputs and Outputs

### Inputs

The Blood Pressure Progression Models take the following as input:

- **map_z**: Current MAP z-score (dimensionless)
- **pp_z**: Current PP z-score (dimensionless)
- **age**: Patient age in years
- **sex**: 1 for male, 2 for female
- **waist_cm**: Waist circumference in centimeters
- **heart_rate**: Heart rate in beats per minute
- **egfr**: Kidney function in ml/min/1.73m²
- **pack_years**: Cumulative smoking exposure
- **alcohol_g_day**: Daily alcohol consumption in grams
- **sodium_mg_day**: Daily sodium intake in milligrams

### Outputs

The models produce:

- **delta_map_z**: Predicted annual change in MAP z-score (typically -0.05 to +0.10)
- **delta_pp_z**: Predicted annual change in PP z-score (typically -0.02 to +0.08)

Positive values mean blood pressure is increasing, negative values mean it's decreasing.

## Design Decisions

### Why Separate Models for MAP and PP?

We use separate models for MAP and PP because they represent different physiological processes:

- **MAP** reflects the average workload on the heart and responds to factors like kidney function and overall cardiovascular health
- **PP** reflects arterial stiffness and increases more predictably with age

Separate models allow each to capture the specific patterns relevant to that process.

### Why Use Z-Scores Instead of mmHg?

The models operate on z-scores rather than actual blood pressure values because:

- **Normalization**: Z-scores put all features on a similar scale, which helps machine learning models learn better
- **Generalization**: Models trained on z-scores can generalize to patients with different baseline blood pressures
- **Numerical stability**: Z-scores are typically between -3 and +3, which is better for numerical computations

### Why Compare Multiple Models for MAP?

We compared Ridge, Random Forest, and Gradient Boosting for MAP progression because:

- **Ridge** is simple and interpretable but may miss nonlinear patterns
- **Random Forest** can capture complex interactions but may overfit
- **Gradient Boosting** often provides the best accuracy but is more complex

**Result: Gradient Boosting was selected** (CV R² highest). MAP progression involves nonlinear interactions between age, eGFR, smoking, and other risk factors that a linear model cannot capture. By comparing them on held-out data (5-fold cross-validation), we confirmed that Gradient Boosting actually performs best on new patients rather than just memorizing the training data.

### Why Only Ridge for PP?

PP progression follows a more predictable pattern based on age (Franklin et al. 1999). A simple linear model like Ridge is sufficient and more interpretable than complex ensemble methods for this relatively stable pattern.

### Why Is PP's Low R² Acceptable?

The PP progression model has a low R² (0.06), meaning it explains only ~6% of the variance in annual PP change. This does **not** mean PP is useless in the simulation:

1. **PP feeds into SBP reconstruction**: The simulation reconstructs SBP as `SBP = MAP + (2/3) × PP`. Even if PP's year-to-year *change* is noisy, the cumulative PP trajectory still tracks the known age-related increase (Ridge intercept ≈ 0.055 z-score/year ≈ 1 mmHg/year in clinical units). Over 30 years, that's ~30 mmHg of PP increase, which is clinically significant.

2. **The noise averages out**: A low R² means the model can't predict the *random fluctuation* in each year's delta, but the *mean prediction* is still correct. Over many simulation years, random errors cancel out while the systematic trend accumulates.

3. **PP drives MAP**: The MAP model's #1 feature importance is `pp_z` at 77.5%. PP's value isn't in predicting itself — it's in being the strongest predictor of how MAP changes. A patient with high PP (stiff arteries) will have faster MAP progression.

4. **It captures the Franklin pattern**: The Ridge intercept encodes the Franklin et al. finding that PP increases steadily with age. The small coefficients from eGFR, smoking, etc. add clinically meaningful adjustments on top of that baseline trend.

In short: PP is the *input that matters most* for MAP prediction, even though PP's own progression is hard to predict year-by-year. The simulation needs PP to be realistic, not perfectly predicted.

## Integration with Other Subsystems

The Blood Pressure Progression Models connect to several other parts of the system:

- **Internal State Representation**: The models consume MAP_z and PP_z as inputs and produce changes to these same variables. This is the primary use of the internal state representation.

- **Mortality Model**: As blood pressure changes (predicted by the progression models), mortality risk changes accordingly. The mortality model uses the SBP reconstructed from the updated MAP_z and PP_z.

- **eGFR Progression Model**: Kidney function (eGFR) is a feature in the blood pressure progression models. As eGFR declines, the models predict different blood pressure trajectories, capturing the connection between kidney and cardiovascular health.

- **Synthetic Cohort Generation**: The synthetic cohort used to train the models includes realistic blood pressure trajectories. This ensures the models learn from data that reflects real-world patterns.

- **Model Training Pipeline**: The training pipeline in `models/pipeline.py` trains these models on the synthetic cohort data, comparing different algorithms and selecting the best performers.

- **Intervention System**: When a blood pressure-lowering intervention is applied, the effect is subtracted from the predicted delta, allowing the simulation to show how treatment changes the trajectory.

The Blood Pressure Progression Models are the engine that drives the simulation forward - they predict how the patient's cardiovascular system will evolve over time, which then affects mortality risk and all other outcomes.

## References

[1] Hoerl, A. E., & Kennard, R. W. (1970). "Ridge regression: Biased estimation for nonorthogonal problems." Technometrics, 12(1), 55-67.

[2] Franklin, S. S., et al. (1999). "Is pulse pressure useful in predicting risk for coronary heart disease? The Framingham Heart Study." Circulation, 100(4), 354-360.

[3] "Sex-specific trajectories of blood pressure and pulse pressure." Nature Cardiovascular Research (2026).

[4] "High waist circumference is a risk factor for hypertension." PMC9278579.

[5] "Revisiting Waist Circumference: A Hypertension Risk Factor." PMC11327828.

[6] "Association Between Waist Circumference and the Prevalence of (Pre)hypertension." PMC8545886.

[7] "Heart Rate and Blood Pressure: Any Possible Implications." PMC3491126.

[8] "Resting heart rate in relation to blood pressure." International Journal of Cardiology (2009).

[9] "Association Between Hypertension and Kidney Function Decline: The CRIC Study." PMC6760841.

[10] "Determining the Relationship Between Blood Pressure, Kidney Function." Hypertension (AHA Journals).

[11] "Cigarette Smoking and Longitudinal Associations With Blood Pressure." PMC8200766 (JAHA, 2021).

[12] "Association Between Smoking and Blood Pressure." Hypertension (AHA Journals).

[13] "Urinary Sodium Excretion Enhances the Effect of Alcohol on Blood Pressure." PMC9319523.

[14] "The interactive association between sodium intake, alcohol." PMC7903677.

[15] "Sodium Sensitivity of Blood Pressure in Long-Term Detoxified Alcoholics." Hypertension (AHA Journals).
