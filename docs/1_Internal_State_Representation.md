# Internal State Representation

## Overview

The Internal State Representation subsystem is the first step in the FlowState AI v.2 simulation. It transforms the blood pressure measurements you would see in a medical record (systolic blood pressure and diastolic blood pressure) into a different format that the computer can work with more effectively.

When you visit a doctor, your blood pressure is always reported as two numbers, like 120/80. The first number (120) is the systolic blood pressure (SBP) - the pressure when your heart beats. The second number (80) is the diastolic blood pressure (DBP) - the pressure when your heart rests between beats. These two numbers are related to each other - they are not independent measurements of two different things. They are two different views of the same underlying state of your blood vessels.

If we tried to model SBP and DBP independently in the simulation, we might get unrealistic results where the relationship between them drifts into impossible territory. To solve this problem, we convert these two numbers into two different quantities that truly are independent:

1. **Mean Arterial Pressure (MAP)** - the average pressure your heart works against
2. **Pulse Pressure (PP)** - the difference between the two numbers, which tells us about artery stiffness

Once we have MAP and PP, we convert them to "z-scores" using population data from a large US health survey (NHANES). A z-score tells us how far a value is from the average. A z-score of 0 means exactly average. A z-score of +1 means one standard deviation above average. This normalization helps the computer models work better and makes the numbers easier to interpret.

The simulation only uses these z-scored values (MAP_z and PP_z) to make predictions. When we need to display results to you or calculate mortality, we convert them back to the familiar SBP and DBP numbers.

## Scientific / Theoretical Basis

### Why MAP and PP?

The formulas we use are based on how the cardiovascular system actually works:

**Mean Arterial Pressure (MAP)** is calculated as:

```
MAP = (SBP + 2 × DBP) / 3
```

This formula weights diastolic pressure twice as heavily as systolic pressure. The 2:1 weighting reflects the fact that at normal resting heart rates, diastole occupies approximately two-thirds of the cardiac cycle - a derived interpretation of the formula's structure. This is the standard formula used in critical care medicine [1]. Note that the 1/3 factor is not constant and varies with heart rate; at high heart rates, MAP is closer to the arithmetic average of systolic and diastolic pressure [1].

**Pulse Pressure (PP)** is calculated as:

```
PP = SBP - DBP
```

Pulse pressure tells us about artery stiffness. Young, elastic arteries absorb the pressure wave from each heartbeat, keeping pulse pressure low. As arteries stiffen with age, they lose this ability. This causes systolic pressure to rise while diastolic pressure stays the same or even falls. This pattern - rising SBP with stable or falling DBP - is exactly what we see in older adults with isolated systolic hypertension [2].

### Population Data

Our z-score normalization uses data from NHANES 2017-2018, the most recent large-scale health survey in the United States. The population constants were derived from analysis of NHANES 2017-2018 data (n=4,705 adults aged 18+). The population averages we use are:

```
MAP mean = 90.11 mmHg,  MAP standard deviation = 12.55 mmHg
PP mean  = 53.74 mmHg,  PP standard deviation  = 18.75 mmHg
```

These numbers come from actual measurements of thousands of US adults, so they represent what's "normal" in the US population. If a patient has MAP_z = 0, they are exactly at the US average. If they have MAP_z = +1.5, they are 1.5 standard deviations above average - which a doctor would recognize as elevated blood pressure.

### Converting Back

To get back to clinical measurements, we use the inverse formulas:

```
SBP = MAP + (2/3) × PP
DBP = MAP - (1/3) × PP
```

These are exact mathematical inverses of the forward transformation. Nothing is lost or approximated. We can always get back exactly the SBP and DBP we started with.

## Data Structures

The subsystem uses a small set of constants and variables. There are no machine learning models here - this subsystem just prepares the data for the models that come later.

### Population Constants

These four numbers are defined at the top of the code and are the same throughout the project:

```
MAP_MEAN  = 90.11  (mmHg)
MAP_STD   = 12.55  (mmHg)
PP_MEAN   = 53.74  (mmHg)
PP_STD    = 18.75  (mmHg)
```

These are used to convert between actual blood pressure values and z-scores.

### Computed Variables

**MAP and PP** are calculated from your SBP and DBP using the formulas above. They are intermediate values that get converted to z-scores immediately.

**MAP_z and PP_z** are the core state variables that the simulation advances through time. They are dimensionless numbers (no units) that represent how many standard deviations above or below average you are. These are what the machine learning models actually use.

**SBP and DBP** are reconstructed from MAP_z and PP_z when needed for display or mortality calculation. We reconstruct them each year rather than storing them, to save memory and ensure they always reflect the current state.

### Feature List

The code defines a list called `PROGRESSION_FEATURES` that specifies what data gets passed to the blood pressure progression models:

```python
PROGRESSION_FEATURES = [
    "map_z", "pp_z", "age", "sex", "waist_cm",
    "heart_rate", "egfr", "pack_years",
    "alcohol_g_day", "sodium_mg_day"
]
```

Notice that `map_z` and `pp_z` are the first two items - they are the most important features for predicting how blood pressure will change over time.

## Step-by-Step Process

### Step 1: Convert Clinical BP to MAP and PP

When the simulation starts, it takes your SBP and DBP as input and converts them:

```
MAP = (SBP + 2 × DBP) / 3
PP  = SBP - DBP
```

For example, if your blood pressure is 120/80:

```
MAP = (120 + 2 × 80) / 3 = (120 + 160) / 3 = 280 / 3 = 93.3 mmHg
PP  = 120 - 80 = 40 mmHg
```

### Step 2: Convert to Z-Scores

Next, we convert MAP and PP to z-scores:

```
MAP_z = (MAP - MAP_MEAN) / MAP_STD
PP_z  = (PP - PP_MEAN) / PP_STD
```

Using our example of MAP = 93.3 and PP = 40:

```
MAP_z = (93.3 - 90.11) / 12.55 = 3.19 / 12.55 = 0.25
PP_z  = (40 - 53.74) / 18.75 = -13.74 / 18.75 = -0.73
```

So this patient has MAP slightly above average (+0.25 standard deviations) and PP below average (-0.73 standard deviations).

### Step 3: Store as State Variables

The simulation stores MAP_z and PP_z as the primary state variables. These are what get advanced through time by the machine learning models. The original SBP and DBP are not stored - they can always be reconstructed if needed.

### Step 4: Annual Loop - Denormalize

Each year of the simulation, the first thing we do is convert the z-scores back to actual blood pressure values:

```
MAP_t = MAP_z × MAP_STD + MAP_MEAN
PP_t  = PP_z × PP_STD + PP_MEAN
```

Using our example z-scores:

```
MAP_t = 0.25 × 12.55 + 90.11 = 3.14 + 90.11 = 93.25 mmHg
PP_t  = -0.73 × 18.75 + 53.74 = -13.69 + 53.74 = 40.05 mmHg
```

### Step 5: Reconstruct SBP and DBP

Then we reconstruct the clinical blood pressure numbers:

```
SBP = MAP_t + (2/3) × PP_t
DBP = MAP_t - (1/3) × PP_t
```

Using our example:

```
SBP = 93.25 + (2/3) × 40.05 = 93.25 + 26.70 = 119.95 mmHg
DBP = 93.25 - (1/3) × 40.05 = 93.25 - 13.35 = 79.90 mmHg
```

These values are stored for display and used to calculate mortality risk.

### Step 6: Advance State

After calculating mortality, the simulation uses machine learning models to predict how MAP_z and PP_z will change over the next year. These predicted changes are added to the current values to get the new state for the next year. Then the loop repeats.

## Inputs and Outputs

### Inputs

The subsystem takes the following as input:

- **SBP** (systolic blood pressure): A number in mmHg, typically between 70 and 220. This comes from your medical record or a blood pressure measurement.
- **DBP** (diastolic blood pressure): A number in mmHg, typically between 40 and 130. This also comes from your medical record or measurement.
- **Population constants** (MAP_MEAN, MAP_STD, PP_MEAN, PP_STD): These are built into the code and come from NHANES data.

### Outputs

The subsystem produces:

- **MAP_z**: A dimensionless number representing how many standard deviations your MAP is from the population average. This is what the simulation uses internally.
- **PP_z**: A dimensionless number representing how many standard deviations your PP is from the population average. This is also used internally.
- **Reconstructed SBP and DBP**: These are calculated on demand when we need to display results or calculate mortality.

The transformation is reversible - we can always get back the original SBP and DBP from MAP_z and PP_z. No information is lost.

## Design Decisions

### Why Not Use SBP and DBP Directly?

The main benefit of using MAP and PP is that it matches how the cardiovascular system actually works. MAP represents the average workload on your heart. PP represents artery stiffness. These are two different physiological processes. By modeling them separately, we can produce realistic trajectories - for example, showing how artery stiffness increases with age, causing SBP to rise while DBP stays the same.

If we tried to model SBP and DBP independently, we might get unrealistic relationships between them.

### Why Use Z-Scores?

Z-scoring helps the machine learning models work better. When all inputs are on a similar scale (around -2 to +2), the models can learn patterns that apply to patients with different baseline blood pressures. This is important for a digital twin that needs to work for many different patients.

### Trade-offs

The main trade-off is added complexity. Developers need to understand the transformation to debug issues. Also, the population constants need to be kept consistent across multiple files in the codebase.

Another limitation is that our population constants are based on US adults. If the simulation were used in other countries with different average blood pressures, the constants would need to be recalibrated.

## Integration with Other Subsystems

The Internal State Representation subsystem connects to several other parts of the system:

- **Blood Pressure Progression Models**: These models take MAP_z and PP_z as input and predict how they will change over time. This is the primary use of the internal state representation.

- **Mortality Model**: This uses the reconstructed SBP value to calculate mortality risk. The mortality model works with clinical measurements (SBP in mmHg), not z-scores.

- **Synthetic Cohort Generation**: When creating training data for the models, we use the same transformation logic to ensure consistency.

- **Model Training Pipeline**: The machine learning models are trained on data that has been transformed the same way, ensuring they work correctly with the simulation.

- **Intervention System**: When applying medication effects, we convert the effect from clinical units (mmHg change in SBP) to z-score units so it can be applied in the same space that the models use.

The Internal State Representation is the foundation of the entire simulation - it's the first step that converts clinical measurements into a format the rest of the system can work with effectively.

## References

[1] Klabunde, R. E. "Mean Arterial Pressure." Cardiovascular Physiology Concepts, 2nd ed. cvphysiology.com. (Provides the MAP formula: MAP ≈ DBP + 1/3(SBP - DBP) = (SBP + 2×DBP)/3, with note on heart-rate dependency)

[2] Magder, S. (2018). "The meaning of blood pressure." Critical Care, 22:257. (Provides physiological interpretation of arterial pressure as an emergent variable from flow and vascular resistance)

[3] Franklin, S. S., et al. (1999). "Is pulse pressure useful in predicting risk for coronary heart disease? The Framingham Heart Study." Circulation, 100(4):354-360. (Documents that PP rise with age is due to increased large-artery stiffness, and that DBP decreases after age 60 while SBP continues to rise)

[4] Population constants derived from analysis of NHANES 2017-2018 data (n=4,705 adults aged 18+)
