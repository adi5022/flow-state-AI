# Physiological Drift Models

## Overview

The Physiological Drift Models capture how certain patient characteristics change naturally over time, independent of disease progression or treatment. These are the "background" changes that happen to everyone as they age.

Currently, the system models two types of physiological drift:

1. **Waist circumference drift**: Waist size tends to increase with age, with the rate depending on whether the patient is lean or obese
2. **Heart rate drift**: Resting heart rate tends to decrease slightly with age

These drifts are important because they affect the blood pressure progression models (waist circumference is a feature) and overall cardiovascular risk.

## Scientific / Theoretical Basis

### Why Model Physiological Drift?

As people age, their bodies change even without disease or treatment. These changes affect cardiovascular risk and disease progression. By modeling physiological drift, the simulation can:

- More accurately predict blood pressure progression (waist circumference is a risk factor)
- Capture age-related changes in cardiovascular physiology
- Provide realistic long-term trajectories

### Waist Circumference Drift

Research shows that waist circumference increases with age, but the rate depends on baseline body composition. The Janssen et al. (2004) study found [1]:

- **Lean patients** (waist < 80 cm): Gain about 0.15 cm per year
- **Obese patients** (waist > 110 cm): Gain about 0.60 cm per year
- **Intermediate patients**: Linear interpolation between these rates

This makes physiological sense - people who are already overweight tend to gain weight faster than lean people.

The drift stops at age 65, reflecting the observation that weight gain tends to plateau in older adults.

### Heart Rate Drift

Resting heart rate tends to decrease slightly with age. The Abhishekh et al. (2013) study found that heart rate decreases by about 0.4 bpm per year after age 50 [2].

This drift has a floor at 45 bpm - heart rate doesn't decrease below this level, which is physiologically reasonable (very low heart rates would be concerning).

## Data Structures

### Waist Drift Parameters

The waist drift model uses these parameters:

```
waist_lean = 80.0 cm      # Threshold for "lean"
waist_obese = 110.0 cm    # Threshold for "obese"
rate_lean = 0.15 cm/yr    # Annual gain for lean patients
rate_obese = 0.60 cm/yr   # Annual gain for obese patients
```

### Heart Rate Drift Parameters

The heart rate drift model uses these parameters:

```
drift_start_age = 50      # Age when drift begins
drift_rate = 0.4 bpm/yr   # Annual decrease
floor_hr = 45 bpm         # Minimum heart rate
```

## Step-by-Step Process

### Waist Circumference Drift

Each year of the simulation (before age 65), the system calculates waist gain:

**Step 1: Determine position between lean and obese**

```
t = (current_waist - waist_lean) / (waist_obese - waist_lean)
```

This value `t` is between 0 and 1:
- t = 0 means the patient is at the lean threshold (80 cm)
- t = 1 means the patient is at the obese threshold (110 cm)
- Values between 0 and 1 represent intermediate positions

**Step 2: Calculate annual gain rate**

```
waist_gain = rate_lean + t × (rate_obese - rate_lean)
```

This linearly interpolates between the lean rate (0.15) and obese rate (0.60).

**Step 3: Update waist circumference**

```
waist = waist + waist_gain
```

**Example**: A patient with waist = 95 cm:

```
t = (95 - 80) / (110 - 80) = 15 / 30 = 0.5
waist_gain = 0.15 + 0.5 × (0.60 - 0.15) = 0.15 + 0.5 × 0.45 = 0.15 + 0.225 = 0.375
waist = 95 + 0.375 = 95.375 cm
```

This patient is halfway between lean and obese, so they gain at a rate halfway between the lean and obese rates.

**Example**: A patient with waist = 70 cm (lean):

```
t = (70 - 80) / (110 - 80) = -10 / 30 = 0 (clipped to 0)
waist_gain = 0.15 + 0 × (0.60 - 0.15) = 0.15
waist = 70 + 0.15 = 70.15 cm
```

This patient is below the lean threshold, so they gain at the minimum rate.

### Heart Rate Drift

Each year of the simulation (after age 50), the system decreases heart rate:

```
if age > 50:
    heart_rate = max(45, heart_rate - 0.4)
```

**Example**: A 55-year-old with heart rate = 70 bpm:

```
heart_rate = max(45, 70 - 0.4) = max(45, 69.6) = 69.6 bpm
```

**Example**: A 70-year-old with heart rate = 46 bpm:

```
heart_rate = max(45, 46 - 0.4) = max(45, 45.6) = 45.6 bpm
```

**Example**: A 75-year-old with heart rate = 45 bpm:

```
heart_rate = max(45, 45 - 0.4) = max(45, 44.6) = 45 bpm
```

The heart rate stays at the floor of 45 bpm.

## Inputs and Outputs

### Inputs

The Physiological Drift Models take:

- **current_waist**: Current waist circumference in cm
- **current_age**: Patient age in years
- **current_heart_rate**: Current heart rate in bpm

### Outputs

The models produce:

- **updated_waist**: Waist circumference after annual drift
- **updated_heart_rate**: Heart rate after annual drift

## Design Decisions

### Why Linear Interpolation for Waist Drift?

We use linear interpolation between the lean and obese rates because:

- It's simple and computationally efficient
- It provides a smooth transition between the two extremes
- The underlying research doesn't provide more granular data

A more complex model (e.g., exponential) could be used, but linear interpolation is sufficient for the simulation's purposes.

### Why Stop Waist Drift at Age 65?

Research shows that weight gain tends to plateau or even reverse in older adults (age 65+). By stopping the drift at age 65, we:

- Match observed population patterns
- Avoid unrealistic continued weight gain in elderly patients
- Simplify the model without losing accuracy

### Why Floor Heart Rate at 45 bpm?

The floor of 45 bpm is physiologically reasonable because:

- Very low resting heart rates (< 45 bpm) are uncommon in the general population
- Rates below 45 bpm might indicate athletic conditioning or medical issues
- The floor prevents unrealistic heart rate values in long simulations

### Why Not Model Other Physiological Drifts?

Currently, we only model waist and heart rate drift. Other potential drifts (e.g., height loss with age, changes in body composition) could be added but are not currently included because:

- They have less direct impact on cardiovascular outcomes
- The evidence for their rates is less clear
- Adding them would increase model complexity without significant benefit

## Integration with Other Subsystems

The Physiological Drift Models connect to:

- **Blood Pressure Progression Models**: Waist circumference is a feature in the blood pressure progression models. As waist drifts upward, the models predict faster blood pressure progression.

- **Mortality Model**: Heart rate can affect cardiovascular risk (though it's not a direct feature in the current mortality model). Changes in waist circumference indirectly affect mortality through blood pressure.

- **Internal State Representation**: While waist and heart rate are not part of the internal state (MAP_z and PP_z), they are input features to the progression models.

The drift models ensure that the simulation captures age-related changes in these important risk factors, making long-term predictions more realistic.

## References

[1] Janssen, I., et al. (2004). "Waist circumference and not body mass index explains obesity-related health risk." American Journal of Clinical Nutrition, 79(3), 379-384.

[2] Abhishekh, K. H., et al. (2013). "Influence of age and gender on autonomic regulation of heart." Journal of Clinical Diagnostic Research, 7(7), 1303-1306.
