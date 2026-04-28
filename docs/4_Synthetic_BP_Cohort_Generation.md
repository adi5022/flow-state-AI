# Synthetic BP Cohort Generation

## Overview

The Synthetic BP Cohort Generation subsystem creates a large dataset of synthetic patients that mimics real-world blood pressure patterns. This synthetic data is used to train the blood pressure progression models.

The system generates 5,000 synthetic patients, each followed for 30 years with annual measurements. Each patient has demographic information (age, sex, waist circumference, heart rate), risk factors (smoking, alcohol, sodium intake), kidney function (eGFR), and blood pressure measurements (SBP, DBP).

The key insight is that we don't use real patient data (which would have privacy issues). Instead, we generate synthetic data that follows the same statistical patterns as real populations, based on published research studies. This gives us a large, diverse dataset for training machine learning models while protecting patient privacy.

The generation process has four phases:

1. **Phase 1**: Generate demographic characteristics
2. **Phase 2**: Generate baseline blood pressure
3. **Phase 3**: Simulate blood pressure progression over 30 years
4. **Phase 4**: Save and validate the dataset

## Scientific / Theoretical Basis

### Why Synthetic Data?

Real patient data has several limitations:

- **Privacy**: Medical records contain sensitive information that cannot be shared freely
- **Cost**: Collecting large longitudinal datasets is expensive
- **Bias**: Real datasets may have selection bias or missing data

Synthetic data solves these problems by generating realistic-looking data from scratch using known population statistics. The key is ensuring the synthetic data matches real-world patterns.

### Population Distributions

We use data from NHANES 2017-2018, a large US health survey, to ensure our synthetic blood pressure values match the real US population [1]. The survey found:

```
SBP: mean = 125.3 mmHg, standard deviation = 20.8 mmHg
DBP: mean = 74.1 mmHg,  standard deviation = 11.6 mmHg
```

We sample blood pressure values from these distributions, then adjust for age using established rates from the Framingham study [2].

### Blood Pressure Progression

Blood pressure naturally increases with age. The Framingham Heart Study established the typical rates of increase [2]:

- **SBP**: Increases about 0.7 mmHg per year from age 30-65
- **DBP**: Increases about 0.1 mmHg per year from age 35-65

After age 65, SBP continues to rise but DBP plateaus or even falls [3]. This pattern reflects arterial stiffening with age.

### Individual Variation

Not everyone's blood pressure progresses at the same rate. We use a "mixed effects" approach [4] where each patient has:

- A personal progression rate (their tendency for blood pressure to increase)
- Random variation at each visit (measurement noise)

This creates realistic diversity in the cohort - some patients' blood pressure rises quickly, others slowly or not at all.

### Risk Factor Effects

We make blood pressure progression depend on observable risk factors:

- **Waist circumference**: Larger waist = faster progression
- **Kidney function (eGFR)**: Lower eGFR = faster progression
- **Smoking**: More pack-years = faster progression
- **Heart rate**: Higher heart rate = faster progression

This creates the "signal" that machine learning models can learn - the relationship between risk factors and blood pressure changes.

## Data Structures

### Patient Records

Each patient has a record for each visit year (0 to 30) containing:

```
- patient_id: Unique identifier
- visit_year: Year of visit (0 = baseline)
- age: Patient age
- sex: 1 for male, 2 for female
- waist_cm: Waist circumference
- heart_rate: Heart rate in bpm
- egfr: Kidney function
- pack_years: Cumulative smoking exposure
- alcohol_g_day: Daily alcohol consumption
- sodium_mg_day: Daily sodium intake
- personal_sbp_rate: Individual SBP progression rate
- personal_dbp_rate: Individual DBP progression rate
- map_z: MAP z-score
- pp_z: PP z-score
- sbp: Systolic blood pressure
- dbp: Diastolic blood pressure
- pp: Pulse pressure
- survival: Cumulative survival probability
- p_death_annual: Annual death probability
```

### Population Constants

The system uses the same population constants as the Internal State Representation:

```
MAP_MEAN = 90.11 mmHg
MAP_STD = 12.55 mmHg
PP_MEAN = 53.74 mmHg
PP_STD = 18.75 mmHg
```

## Step-by-Step Process

### Step 1: Generate Demographics

We create 5,000 synthetic patients with realistic demographic characteristics:

**Age**: Uniform distribution from 35 to 75 years

```
age = random.uniform(35, 75)
```

**Sex**: 45% male, 45% female (matching US population)

```
sex = random.choice([1, 2], p=[0.45, 0.55])
```

**Waist circumference**: Different distributions for men and women

```
Men: mean = 99 cm, SD = 14 cm
Women: mean = 90 cm, SD = 15 cm
```

**Heart rate**: Mean = 70 bpm, SD = 10 bpm

**eGFR**: Mean = 90 ml/min/1.73m², SD = 20

**Smoking status**: 60% never smokers, 25% former smokers, 15% current smokers

**Alcohol intake**: 40% none, 45% moderate (5-20 g/day), 15% heavy (20-60 g/day)

**Sodium intake**: Mean = 3500 mg/day, SD = 900 mg/day

### Step 2: Generate Baseline Blood Pressure

We sample blood pressure from the NHANES distribution at a reference age of 55:

```
sbp_base = random.normal(125.3, 20.8)
dbp_base = random.normal(74.1, 11.6)
```

Then we adjust for the patient's actual age:

```
age_diff = patient_age - 55
sbp = sbp_base + 0.7 × age_diff
dbp = dbp_base + 0.1 × age_diff
```

For example, a 45-year-old (10 years younger than 55):

```
age_diff = 45 - 55 = -10
sbp = 125.3 + 0.7 × (-10) = 125.3 - 7 = 118.3 mmHg
dbp = 74.1 + 0.1 × (-10) = 74.1 - 1 = 73.1 mmHg
```

This makes sense - younger people typically have lower blood pressure.

We then convert to the internal state representation:

```
MAP = (SBP + 2 × DBP) / 3
PP = SBP - DBP
map_z = (MAP - MAP_MEAN) / MAP_STD
pp_z = (PP - PP_MEAN) / PP_STD
```

### Step 3: Calculate Personal Progression Rates

Each patient gets a personal blood pressure progression rate based on their risk factors:

```
personal_sbp_rate = 0.7
                    + 0.020 × max(0, waist - threshold)
                    + 0.015 × max(0, 90 - eGFR)
                    + 0.012 × pack_years
                    + 0.008 × max(0, heart_rate - 70)
                    + N(0, 0.5)  # individual variation noise
```

The base rate is 0.7 mmHg/year (from Framingham [2]). The risk factor coefficients (0.020, 0.015, 0.012, 0.008) are engineering choices calibrated to produce realistic progression rates — they are not directly from a single published source, but are consistent with the direction and relative magnitude of effects reported in the literature [5, 6, 7, 8].

- If waist is above the threshold (94 cm for men, 80 cm for women), add 0.020 per cm over threshold
- If eGFR is below 90, add 0.015 per point below 90
- Add 0.012 per pack-year of smoking
- If heart rate is above 70, add 0.008 per bpm over 70

**Example**: A 55-year-old male with waist = 105 cm, eGFR = 75, 20 pack-years, heart rate = 78:

```
personal_sbp_rate = 0.7
                    + 0.020 × (105 - 94) = 0.22
                    + 0.015 × (90 - 75) = 0.225
                    + 0.012 × 20 = 0.24
                    + 0.008 × (78 - 70) = 0.064
                    + random_noise
                    = 1.449 + noise
```

This patient's blood pressure will rise faster than average due to multiple risk factors.

### Step 4: Annual Simulation Loop

For each year from 0 to 30:

**Calculate internal state**:

```
MAP = (SBP + 2 × DBP) / 3
PP = SBP - DBP
map_z = (MAP - MAP_MEAN) / MAP_STD
pp_z = (PP - PP_MEAN) / PP_STD
```

**Calculate mortality** (using the same formula as the Mortality Model):

```
log_h = log(baseline_mortality) + blood_pressure_risk + kidney_risk + smoking_risk
p_death = 1 - exp(-exp(log_h))
survival = survival × (1 - p_death)
```

**Save the record** with all current values

**Check termination**: If survival falls below 1%, stop simulating this patient (they've reached end-of-life)

**Update blood pressure for next year**:

```
SBP = SBP + personal_sbp_rate + N(0, 0.5)  # visit noise
DBP = DBP + personal_dbp_rate + N(0, 0.3)  # visit noise
```

After age 65, DBP stops increasing (rate = 0) based on Franklin et al. [3].

**Update other variables**:

```
eGFR = eGFR - decline_rate(age)  # Kidney function declines with age
waist = waist + 0.5  # Waist increases slowly until age 60
heart_rate = heart_rate - 0.4  # Heart rate decreases slowly after age 50
pack_years = pack_years + 1  # For current smokers only
age = age + 1
```

### Step 5: Save and Validate

After simulating all patients, we save the data to a CSV file and run sanity checks:

```
Mean SBP should be 122-128 mmHg (target ~125)
Mean DBP should be 72-78 mmHg (target ~74)
SD SBP should be ~20 mmHg
SD DBP should be ~11 mmHg
MAP_z SD should be ~1.0
PP_z SD should be ~1.0
```

We also check the distribution of blood pressure categories:

```
Normal (<120 mmHg): ~30%
Elevated (120-129 mmHg): ~25%
Stage 1 (130-139 mmHg): ~20%
Stage 2 (≥140 mmHg): ~25%
```

These targets come from the Ostchega et al. study of NHANES data [1].

## Inputs and Outputs

### Inputs

The Synthetic BP Cohort Generation doesn't take patient-specific inputs. Instead, it generates synthetic data from scratch using:

- Population statistics from NHANES 2017-2018 [1]
- Blood pressure progression rates from Framingham [2, 3]
- Demographic distributions from various studies
- Risk factor relationships from epidemiological literature

### Outputs

The system produces a CSV file with approximately 150,000 rows (5,000 patients × 30 years on average, though some patients die earlier). Each row represents one visit for one patient.

The output is saved to `data/synthetic_cohort/cohort.csv`.

## Design Decisions

### Why 5,000 Patients?

We chose 5,000 patients as a balance between:

- **Sufficient data**: Enough to train machine learning models effectively
- **Computational efficiency**: Can be generated quickly on a standard computer
- **Statistical power**: Large enough to capture rare patterns

### Why 30 Years?

A 30-year follow-up period is long enough to:

- Show meaningful blood pressure progression
- Capture the effects of aging on blood pressure
- Include enough patients who reach end-of-life for mortality modeling

### Why Use Personal Progression Rates?

Instead of giving everyone the same blood pressure progression rate, we give each patient a personal rate based on their risk factors. This creates realistic diversity and gives the machine learning models a clear signal to learn - the relationship between risk factors and blood pressure changes.

### Why Add Random Noise?

Real blood pressure measurements have variability due to:

- Measurement error (different devices, different techniques)
- Natural day-to-day fluctuation
- Stress, caffeine, recent activity

We add random noise to each visit to mimic this real-world variability. This makes the trained models more robust to measurement noise.

### Why Censor Patients Who Die?

In real longitudinal studies, patients who die are "censored" - they stop contributing data after death. We mimic this by stopping the simulation when survival falls below 1%. This ensures the training data reflects what would be observed in a real study.

## Integration with Other Subsystems

The Synthetic BP Cohort Generation connects to several other parts of the system:

- **Model Training Pipeline**: The synthetic cohort is the primary training data for the blood pressure progression models. The pipeline loads this CSV file and trains the models on the consecutive visit pairs.

- **Blood Pressure Progression Models**: These models are trained on the synthetic cohort data. The patterns learned from the synthetic data are what allow the models to make predictions in the simulation.

- **Mortality Model**: The synthetic cohort uses the same mortality calculation as the simulation engine, ensuring consistency between training and simulation.

- **Internal State Representation**: The synthetic cohort uses the same transformation (SBP/DBP to MAP/PP to z-scores) as the simulation, ensuring the trained models work correctly with the simulation's internal representation.

The Synthetic BP Cohort Generation is the foundation of the entire machine learning pipeline - without realistic synthetic data, the progression models would have nothing to learn from.

## References

[1] Ostchega, Y., et al. (2020). "Hypertension prevalence and control among adults: United States, 2015-2018." NCHS Data Brief, No. 364.

[2] Franklin, S. S., et al. (1999). "Is pulse pressure useful in predicting risk for coronary heart disease? The Framingham Heart Study." Circulation, 100(4), 354-360.

[3] Franklin, S. S., et al. (1997). "Hemodynamic patterns of age-related changes in blood pressure." Circulation, 96(1), 308-315.

[4] Laird, N. M., & Ware, J. H. (1982). "Random-effects models for longitudinal data." Biometrics, 38(4), 963-974.

[5] Janssen, I., et al. (2004). "Waist circumference and not body mass index explains obesity-related health risk." American Journal of Clinical Nutrition, 79(3), 379-384.

[6] Go, A. S., et al. (2004). "Chronic kidney disease and the risks of death." New England Journal of Medicine, 351(13), 1296-1305.

[7] Lubin, J. H., et al. (2016). "Risk of Cardiovascular Disease from Cumulative Cigarette Use and the Impact of Smoking Intensity." Epidemiology, 27(3), 395-404.

[8] Palatini, P., et al. (2011). "Heart rate and hypertension." Journal of Hypertension, 29(7), 1303.
