# Intervention System

## Overview

The Intervention System allows the simulation to model the effects of medications and treatments on patient outcomes. It defines interventions with evidence-based parameters for how they affect blood pressure, kidney function, and mortality risk, along with their potential adverse effects.

Currently, the system includes one fully implemented intervention: ACE Inhibitor. This is a commonly prescribed blood pressure medication that also protects kidney function and reduces cardiovascular mortality. All parameters are backed by peer-reviewed research citations.

The system allows users to:
- Select an intervention (currently ACE Inhibitor)
- Choose which year to start the intervention
- See the predicted effects on blood pressure, kidney function, and survival
- Model adverse effects that might cause treatment discontinuation

The intervention system runs two parallel simulations: one without treatment (baseline) and one with treatment. By comparing the results, we can quantify the years of life gained from the intervention.

## Scientific / Theoretical Basis

### Why Model Interventions?

A digital twin is most useful when it can predict "what if" scenarios. The intervention system allows us to answer questions like:

- "What if I start taking blood pressure medication at age 50 instead of 60?"
- "How much longer would I live if I take this medication?"
- "What are the chances of side effects?"

These predictions help patients and doctors make informed treatment decisions.

### ACE Inhibitor Effects

ACE Inhibitors (Angiotensin-Converting Enzyme inhibitors) are a class of medications commonly used to treat high blood pressure and protect kidney function. The effects are based on major clinical trials:

**Blood pressure reduction**: -8 mmHg SBP reduction from meta-analysis [1]

**Kidney protection**: 22% reduction in eGFR decline rate from the REIN trial [2]

**Cardiovascular mortality**: Hazard ratio of 0.74 (26% risk reduction) from the HOPE trial [3]

### Adverse Effects

All medications have potential side effects. The system models the annual risk of developing adverse effects based on clinical evidence:

- **Dry cough**: 15% annual risk (most common, usually not serious) [4]
- **Hypotension**: 9% annual risk (low blood pressure causing dizziness) [4]
- **Hyperkalemia**: 4% annual risk (high potassium levels) [4]
- **Renal dysfunction**: 7% annual risk (worsening kidney function) [4]
- **Angioedema**: 0.3% annual risk (swelling, potentially life-threatening) [4]

### Discontinuation Criteria

In clinical practice, medications are sometimes stopped due to side effects or safety concerns. The system models when discontinuation might occur:

- **Angioedema**: Always discontinues (life-threatening)
- **Hypotension**: Discontinue if SBP falls below 90 mmHg
- **Hyperkalemia**: Discontinue if eGFR falls below 45 ml/min/1.73m²
- **Renal dysfunction**: Discontinue if eGFR falls below 30 ml/min/1.73m²
- **Dry cough**: 30% chance patient chooses to discontinue (patient preference)

## Data Structures

### Intervention Dataclass

The Intervention dataclass defines all parameters for an intervention:

```python
@dataclass
class Intervention:
    name: str                          # Human-readable name
    start_year: int                     # Year to apply intervention
    delta_sbp: float                    # SBP change in mmHg (negative = reduction)
    egfr_protection: float              # Fraction of eGFR decline prevented (0.0-1.0)
    cv_hr_modifier: float               # Hazard ratio for CV mortality
    
    # Adverse effect risks (annual probability)
    dry_cough_risk: float
    hypotension_risk: float
    hyperkalemia_risk: float
    renal_dysfunction_risk: float
    angioedema_risk: float
    
    # Discontinuation thresholds
    discontinuation_sbp_threshold: float
    discontinuation_egfr_threshold: float
    
    # Citations
    citation_sbp: str
    citation_egfr: str
    citation_cv: str
    citation_adverse: str
```

### ACE Inhibitor Configuration

The ACE Inhibitor is defined with specific parameters:

```
name: "ACE Inhibitor"
delta_sbp: -8.0 mmHg
egfr_protection: 0.22 (22% reduction in decline)
cv_hr_modifier: 0.74 (26% mortality risk reduction)

dry_cough_risk: 0.15 (15% annual)
hypotension_risk: 0.09 (9% annual)
hyperkalemia_risk: 0.04 (4% annual)
renal_dysfunction_risk: 0.07 (7% annual)
angioedema_risk: 0.003 (0.3% annual)

discontinuation_sbp_threshold: 90.0 mmHg
discontinuation_egfr_threshold: 30.0 ml/min/1.73m²
```

## Step-by-Step Process

### Step 1: Select Intervention and Start Year

The user selects an intervention (currently only ACE Inhibitor) and chooses which year to start treatment:

```python
intervention = get_intervention("ACE Inhibitor")
intervention.start_year = user_selected_year
```

For example, starting at year 5 means the patient takes the medication from year 5 through year 30 (or until discontinuation).

### Step 2: Run Two Parallel Simulations

The system runs two simulations:

- **Baseline simulation**: No intervention, natural disease progression
- **Treated simulation**: Intervention applied from start_year onward

Both simulations start with the same initial patient characteristics.

### Step 3: Apply Intervention Effects (Each Year)

In the treated simulation, for each year from start_year onward:

**Apply blood pressure reduction**:

```
map_z = apply_sbp_effect(map_z, delta_sbp)
```

This converts the SBP reduction to MAP_z units. For ACE Inhibitor (-8 mmHg):

```
map_z_offset = (-8 × 0.67) / 12.55 = -0.43
map_z_new = map_z + (-0.43)
```

**Apply kidney protection**:

```
egfr_decline = apply_egfr_protection(egfr_decline, protection_factor)
```

For ACE Inhibitor (22% protection):

```
protected_decline = egfr_decline × (1 - 0.22) = egfr_decline × 0.78
```

**Apply mortality risk reduction**:

```
log_hazard = apply_cv_mortality_modifier(log_hazard, hr_modifier)
```

For ACE Inhibitor (hazard ratio 0.74):

```
log_hazard_new = log_hazard + log(0.74) = log_hazard - 0.301
```

### Step 4: Check for Adverse Effects

Each year on treatment, the system randomly determines if adverse effects occur based on annual risk:

```
adverse_effects = check_adverse_effects(intervention, year_on_treatment, sbp, egfr)
```

For each adverse effect, a random number is compared to the annual risk. If the random number is lower, the adverse effect occurs.

**Example**: For dry cough with 15% annual risk:

```
if random() < 0.15:
    dry_cough = True
```

### Step 5: Check Discontinuation Criteria

If adverse effects occur, the system checks if discontinuation is warranted:

```
should_continue = apply_intervention_continuity(intervention, is_active, adverse_effects)
```

- Angioedema always causes discontinuation
- Hypotension causes discontinuation if SBP < 90 mmHg
- Hyperkalemia causes discontinuation if eGFR < 45 ml/min/1.73m²
- Renal dysfunction causes discontinuation if eGFR < 30 ml/min/1.73m²
- Dry cough has 30% chance of patient choosing to discontinue

### Step 6: Continue or Stop Treatment

If discontinuation criteria are met, the intervention stops being applied for all subsequent years. The patient continues in the simulation but without treatment effects.

### Step 7: Compare Survival Curves

After both simulations complete, the system compares the survival curves:

```
years_of_life_gained = treated_survival_years - baseline_survival_years
```

This quantifies the benefit of the intervention in terms of additional years of life.

## Inputs and Outputs

### Inputs

The Intervention System takes:

- **Intervention name**: Which medication to model (currently only "ACE Inhibitor")
- **Start year**: When to begin treatment (1-29)
- **Patient characteristics**: Age, sex, blood pressure, kidney function, etc. (from the simulation)

### Outputs

The system produces:

- **Treated simulation results**: Blood pressure, kidney function, and survival with intervention
- **Baseline simulation results**: Same outcomes without intervention
- **Adverse effect events**: Which side effects occurred and when
- **Years of life gained**: Difference in survival between treated and baseline
- **Discontinuation information**: If and when treatment was stopped

## Design Decisions

### Why Only ACE Inhibitor Currently?

ACE Inhibitor was chosen as the first intervention because:

- It has strong evidence for multiple benefits (blood pressure, kidney protection, mortality)
- It's widely prescribed and well-studied
- The parameters are well-documented in major clinical trials

Future interventions (Beta Blockers, Calcium Channel Blockers, etc.) can be added to the registry using the same framework.

### Why Model Adverse Effects?

Modeling adverse effects is important because:

- Real patients do discontinue medications due to side effects
- This affects the actual benefit patients experience
- It helps set realistic expectations for treatment

Without adverse effect modeling, the simulation would overestimate treatment benefit.

### Why Use Hazard Ratio for Mortality?

We use a hazard ratio multiplier rather than an absolute risk reduction because:

- Mortality risk varies by age and other factors
- A hazard ratio applies proportionally across all risk levels
- This matches how clinical trials report mortality benefits (HOPE trial reported HR 0.74)

### Why Convert SBP Effect to MAP_z?

The internal state representation uses MAP_z, not SBP. To apply the intervention effect, we convert:

```
map_z_offset = (delta_sbp × 0.67) / MAP_STD
```

The 0.67 factor accounts for the relationship between SBP and MAP (SBP = MAP + 2/3×PP). This is a conservative simplification.

## Integration with Other Subsystems

The Intervention System connects to:

- **Blood Pressure Progression Models**: The intervention modifies the blood pressure trajectory by reducing SBP, which affects MAP_z
- **eGFR Progression Model**: The intervention reduces the eGFR decline rate through kidney protection
- **Mortality Model**: The intervention reduces cardiovascular mortality risk through the hazard ratio modifier
- **Internal State Representation**: The intervention effects are applied to MAP_z, which is then converted back to SBP for display and mortality calculation

The intervention system modifies the simulation loop to apply treatment effects from the specified start year onward, allowing direct comparison between treated and untreated scenarios.

## References

[1] Blood Pressure Lowering Treatment Trialists' Collaboration. (2003). "Effects of different blood-pressure-lowering regimens on major cardiovascular events: results of prospectively-designed overviews of randomised trials." Lancet, 362(9395), 1527-1535.

[2] Jafar, T. H., et al. (2001). "Angiotensin-converting enzyme inhibitors and progression of nondiabetic renal disease: A meta-analysis of patient-level data." Annals of Internal Medicine, 135(2), 73-87.

[3] Yusuf, S., et al. (2000). "Effects of an angiotensin-converting-enzyme inhibitor, ramipril, on cardiovascular events in high-risk patients." New England Journal of Medicine, 342(3), 145-153.

[4] StatPearls. "ACE Inhibitors." NCBI Bookshelf, PMID: 30570943.
