# Mortality Model

## Overview

The Mortality Model calculates the probability that a simulated patient will die in each year of the simulation. This serves two purposes: it determines when the simulation should end (when survival drops below 1%), and it provides a measure of cardiovascular risk that helps evaluate the impact of treatments.

The model combines four different sources of mortality risk:

1. **Baseline mortality** from US life tables - how likely people of a given age and sex are to die in the general population
2. **Blood pressure risk** from the Lewington study - how much elevated blood pressure increases mortality risk
3. **Kidney function risk** from the Go et al. study - how reduced kidney function increases cardiovascular risk
4. **Smoking risk** from the Doll study - how smoking exposure affects mortality

These four components are combined using the D'Agostino framework, a well-established method in cardiovascular risk prediction that allows different risk factors to be combined in a mathematically sound way.

The mortality calculation happens each year before the patient's physiological state (blood pressure, kidney function, etc.) is updated. This means the mortality risk for year 5 depends on the patient's state at the start of year 5, which reflects all the changes that happened in years 1-4.

The simulation stops when cumulative survival falls below 1%. This is more realistic than running for a fixed number of years because high-risk patients naturally have shorter simulated lifespans, while low-risk patients can live longer. This produces survival curves that look like what we see in real populations.

The Mortality Model is the bridge between the physiological simulation and clinical outcomes. While other models describe how blood pressure and kidney function change over time, the mortality model translates those changes into the ultimate endpoint: death. This allows us to measure the benefit of treatments by comparing survival curves with and without treatment.

## Scientific / Theoretical Basis

### The D'Agostino Framework

The D'Agostino framework provides a mathematically sound way to combine different risk factors [1]. Instead of directly adding probabilities, we work in "log-hazard space":

```
total log-hazard = log(baseline mortality) + blood pressure contribution + kidney contribution + smoking contribution
```

The total hazard is then:

```
hazard = exp(total log-hazard)
```

And the annual probability of death is:

```
p_death = 1 - exp(-hazard)
```

This approach ensures risk factors compound correctly without being double-counted or undercounted.

### Baseline Mortality from Life Tables

The baseline mortality comes from US national life tables published by Arias and Xu [2]. These tables tell us the probability that a person of a given age and sex will die within one year from all causes. The tables are based on death certificates and population estimates, making them the gold standard for US mortality data.

The simulation uses life table values at ages 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, and 85. For ages between these values, we use linear interpolation to get a smooth transition.

### Blood Pressure Risk (Lewington Study)

The Lewington meta-analysis pooled data from 61 studies with about one million adults [3]. It found that each 20 mmHg increase in systolic blood pressure above 115 mmHg approximately doubles vascular mortality risk.

Importantly, the effect varies by age. Younger people have a stronger relative risk from blood pressure, while older people have a weaker relative risk (though their absolute risk is higher due to age). The study reported age-specific coefficients:

| Age Range | Coefficient |
|-----------|-------------|
| 40-49     | 0.706       |
| 50-59     | 0.693       |
| 60-69     | 0.615       |
| 70-79     | 0.513       |
| 80+       | 0.398       |

The blood pressure contribution is calculated as:

```
sbp_term = max(0, beta × (SBP - 115) / 20)
```

The `max(0, ...)` part means that if SBP is below 115 mmHg, the contribution is zero - we don't assume a protective effect for low blood pressure. This is because the Lewington study only validated the relationship in the range of 115-180 mmHg. The 115 mmHg floor represents the lower boundary of the study's observed SBP range, below which no mortality coefficient is applied as the relationship was not studied in that range.

### Kidney Function Risk (Go et al. Study)

The Go et al. study analyzed over 1 million adults and found that chronic kidney disease is an independent risk factor for cardiovascular death [4]. They reported hazard ratios by eGFR category:

| eGFR Range (ml/min/1.73m²) | Hazard Ratio | Log-Hazard |
|---------------------------|--------------|------------|
| ≥ 60                      | 1.00         | 0.000      |
| 45-59                     | 1.20         | 0.182      |
| 30-44                     | 1.80         | 0.588      |
| 15-29                     | 3.20         | 1.163      |
| < 15                      | 5.90         | 1.775      |

We only apply this risk when eGFR falls below 60 ml/min/1.73m², which is the clinical threshold for Stage 3 chronic kidney disease. The 60-89 range represents mildly reduced kidney function that doesn't carry the same independent cardiovascular risk.

### Smoking Risk (Doll Study)

The smoking contribution uses a coefficient of 0.166 per 10 pack-years (0.0166 per pack-year). A "pack-year" is the number of packs smoked per day multiplied by the number of years of smoking.

```
smoking_term = 0.166 × pack_years / 10
```

This linear approximation captures the general dose-response relationship between smoking and mortality. The coefficient value (0.0166 per pack-year) falls within the range of excess relative risk per pack-year reported by Lubin et al. (2016) for cardiovascular disease (0.011 to 0.046 per pack-year) [6].

## Data Structures

The Mortality Model uses lookup tables and computed values. There are no machine learning models here - this subsystem just calculates mortality risk based on established formulas.

### Life Table Dictionaries

**QX_MALE** and **QX_FEMALE** are dictionaries that map ages to baseline annual death probabilities. These values come from US national life tables.

**Male mortality rates by age:**

| Age | Death Probability |
|-----|------------------|
| 35  | 0.001932 |
| 40  | 0.002585 |
| 45  | 0.003681 |
| 50  | 0.005737 |
| 55  | 0.008729 |
| 60  | 0.013599 |
| 65  | 0.020636 |
| 70  | 0.032154 |
| 75  | 0.049400 |
| 80  | 0.076066 |
| 85  | 0.116485 |

**Female mortality rates by age:**

| Age | Death Probability |
|-----|------------------|
| 35  | 0.001186 |
| 40  | 0.001579 |
| 45  | 0.002350 |
| 50  | 0.003637 |
| 55  | 0.005571 |
| 60  | 0.007928 |
| 65  | 0.012261 |
| 70  | 0.019826 |
| 75  | 0.031721 |
| 80  | 0.052210 |
| 85  | 0.085374 |

Notice that male mortality is consistently higher than female mortality at all ages.

### Lewington Beta Coefficients

**LEWINGTON_BETA** is a dictionary that maps age decades to blood pressure risk coefficients:

```python
LEWINGTON_BETA = {
    40: 0.706,
    50: 0.693,
    60: 0.615,
    70: 0.513,
    80: 0.398
}
```

These coefficients decrease with age, meaning blood pressure has a weaker relative effect on mortality in older adults (though their absolute risk is still higher due to age).

### Interpolation Function

The `baseline_qx()` function performs linear interpolation to get mortality rates for ages that aren't exactly in the table. For example, if a patient is 47.5 years old, it interpolates between the values for ages 45 and 50.

### Computed Variables

The simulation computes these values each year:

- **sbp_term**: Blood pressure contribution to mortality risk
- **egfr_hr**: Kidney function contribution to mortality risk
- **log_h**: Total log-hazard (sum of all components)
- **p_death**: Annual probability of death
- **survival**: Cumulative survival probability (starts at 1.0)

## Step-by-Step Process

### Step 1: Get Baseline Mortality

First, we get the baseline mortality probability from the life tables based on the patient's age and sex. If the age isn't exactly in the table, we interpolate between the nearest values.

For example, a 47-year-old male would have mortality interpolated between age 45 (0.003681) and age 50 (0.005737).

### Step 2: Get Blood Pressure Coefficient

We determine which Lewington coefficient to use based on the patient's age decade:

```
age_dec = round_down_to_nearest_10(age)
beta = LEWINGTON_BETA[age_dec]
```

For example, age 47 would use coefficient 0.706 (the 40s coefficient). Age 83 would use coefficient 0.398 (the 80+ coefficient).

### Step 3: Calculate Blood Pressure Contribution

We calculate how much the patient's blood pressure adds to mortality risk:

```
sbp_term = max(0, beta × (SBP - 115) / 20)
```

The `max(0, ...)` ensures that if SBP is below 115 mmHg, the contribution is zero.

**Example:** A 45-year-old with SBP 160 mmHg:

```
sbp_term = 0.706 × (160 - 115) / 20 = 0.706 × 2.25 = 1.589
```

This corresponds to a hazard ratio of exp(1.589) ≈ 4.9, meaning about 4.9 times higher risk than someone with SBP 115 mmHg.

### Step 4: Calculate Kidney Function Contribution

We check the patient's eGFR and assign a risk based on which category they fall into:

```
if eGFR ≥ 60:   egfr_hr = 0.0
elif eGFR ≥ 45:  egfr_hr = log(1.2) = 0.182
elif eGFR ≥ 30:  egfr_hr = log(1.8) = 0.588
elif eGFR ≥ 15:  egfr_hr = log(3.2) = 1.163
else:            egfr_hr = log(5.9) = 1.775
```

**Example:** A patient with eGFR 50 would have egfr_hr = 0.182 (hazard ratio 1.2). A patient with eGFR 25 would have egfr_hr = 1.163 (hazard ratio 3.2).

### Step 5: Calculate Smoking Contribution

We calculate the smoking risk based on pack-years:

```
smoking_term = 0.166 × pack_years / 10
```

**Example:** A patient with 20 pack-years:

```
smoking_term = 0.166 × 20 / 10 = 0.332
```

### Step 6: Combine All Components

We add all the contributions together in log-hazard space:

```
log_h = log(baseline_mortality) + sbp_term + egfr_hr + smoking_term
```

### Step 7: Convert to Death Probability

We convert the log-hazard to an annual death probability:

```
hazard = exp(log_h)
p_death = 1 - exp(-hazard)
```

We also clip the result to be between 0 and 0.999 to avoid numerical issues.

### Step 8: Update Cumulative Survival

We update the cumulative survival probability:

```
survival = survival × (1 - p_death)
```

Survival starts at 1.0 (100%) and decreases each year.

### Step 9: Check Termination

If survival falls below 1%, we stop the simulation - the patient has effectively reached end-of-life.

## Inputs and Outputs

### Inputs

The Mortality Model takes the following as input:

- **Age**: The patient's current age in years (typically 35-85)
- **Sex**: 1 for male, 2 for female
- **SBP**: Systolic blood pressure in mmHg (typically 70-220)
- **eGFR**: Kidney function in ml/min/1.73m² (typically 15-160)
- **Pack-years**: Cumulative smoking exposure (typically 0-100)

### Outputs

The Mortality Model produces:

- **Annual death probability**: The probability the patient will die in the current year (displayed as a percentage)
- **Cumulative survival**: The probability the patient has survived through all years up to now (displayed as a percentage)

The simulation stops when cumulative survival falls below 1%.

## Design Decisions

### Why Use the D'Agostino Framework?

The D'Agostino framework is mathematically sound for combining risk factors. It ensures risk factors compound correctly without being double-counted. Alternative approaches (like simple multiplication) can produce unrealistic results when multiple risk factors are present.

### Why Floor the SBP Contribution at Zero?

We floor the blood pressure contribution at zero for SBP below 115 mmHg. Without this floor, the formula would predict a protective effect for low blood pressure, which would produce unrealistically high survival probabilities. The Lewington study only validated the relationship in the range of 115-180 mmHg, so we don't extrapolate below that range. The 115 mmHg floor represents the lower boundary of the study's observed SBP range, below which no mortality coefficient is applied as the relationship was not studied in that range.

### Why Use Categorical eGFR Values?

We use the exact categorical hazard ratios from the Go et al. study rather than a continuous function. This is more faithful to the source data. A continuous function would require assumptions about the relationship that aren't supported by the study. The categorical approach produces step changes at the thresholds, but this matches what the study actually found.

### Why US Life Tables?

We use US national life tables because the simulation is designed for use in the United States. This ensures baseline mortality reflects the actual population. If the simulation were deployed to other countries, the life tables would need to be recalibrated.

### Why Stop at 1% Survival?

We terminate the simulation when survival falls below 1%. This represents a point where the patient has effectively reached end-of-life. Continuing beyond this adds computational cost without meaningful clinical information. This threshold is consistent with clinical practice.

## Integration with Other Subsystems

The Mortality Model connects to several other parts of the system:

- **Internal State Representation**: Uses the reconstructed SBP value (converted from MAP_z and PP_z) for blood pressure risk calculation

- **Blood Pressure Progression Models**: As blood pressure changes over time (predicted by the progression models), mortality risk changes accordingly. This creates a feedback loop where higher blood pressure increases mortality, and if the patient survives, blood pressure continues to worsen

- **eGFR Progression Model**: As kidney function declines over time, mortality risk increases. This captures the connection between kidney and cardiovascular health

- **Synthetic Cohort Generation**: When creating training data, we use the same mortality calculation to censor patients who would have died, ensuring the training data reflects real-world patterns

- **Intervention System**: When a medication is applied, we modify the mortality calculation to reflect the treatment's benefit. This allows us to compare survival curves with and without treatment

- **Physiological Drift Models**: Age and other physiological factors affect blood pressure and kidney function, which in turn affect mortality

The Mortality Model is the integrator of the entire simulation - it combines all the physiological changes to produce the ultimate outcome: survival.

## References

[1] D'Agostino, R. B., et al. (2008). "General cardiovascular risk profile for use in primary care." Circulation, 117(6), 743-753.

[2] Arias, E., & Xu, J. (2022). "United States Life Tables, 2020." National Vital Statistics Reports, 71(1).

[3] Lewington, S., et al. (2002). "Age-specific relevance of usual blood pressure to vascular mortality." Lancet, 360(9349), 1903-1913.

[4] Go, A. S., et al. (2004). "Chronic kidney disease and the risks of death." New England Journal of Medicine, 351(13), 1296-1305.

[5] Doll, R., et al. (2004). "Mortality in relation to smoking: 50 years' observations on male British doctors." BMJ, 328(7455), 1519.

[6] Lubin, J. H., et al. (2016). "Risk of Cardiovascular Disease from Cumulative Cigarette Use and the Impact of Smoking Intensity." Epidemiology, 27(3), 395-404.
