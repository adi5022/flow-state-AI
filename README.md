# FlowState AI

A patient-specific cardiovascular digital twin for hypertension progression modeling and intervention simulation. Given 10 clinical measurements, it simulates blood pressure trajectory, kidney function decline, and survival probability year-by-year over 30 years. Every parameter is traceable to a peer-reviewed citation.

## Overview

FlowState AI is a research prototype that demonstrates how machine learning can be combined with evidence-based clinical science to create personalized cardiovascular risk models. The application simulates:

- **Blood pressure progression** year-by-year using MAP (Mean Arterial Pressure) and PP (Pulse Pressure) z-scores
- **Kidney function decline** using a Partial Linear Model (PLM) that preserves explicit clinical equations
- **Mortality risk** using a log-additive hazard model with age, BP, eGFR, and smoking components
- **Intervention effects** (currently ACE Inhibitor) with cited clinical parameters
- **Comparative trajectories** showing treated vs untreated outcomes

The clinical message: show the divergence between doing nothing and treating early.

## Key Features

### Physiological Modeling
- **MAP/PP z-score representation**: Instead of tracking SBP/DBP directly, the model uses standardized z-scores for MAP (perfusion pressure) and PP (arterial stiffness). This is physiologically more accurate as these parameters respond differently to age and pathology (Franklin et al., 1999).
- **Partial Linear Model for eGFR**: A two-layer architecture that preserves explicit clinical equations (age-specific decline, sex adjustment, sodium load) while using Gradient Boosting to capture nonlinear residual patterns.
- **Evidence-based parameters**: Every value, coefficient, rate, threshold, and model architecture decision is backed by a peer-reviewed citation.

### Machine Learning Models
- **MAP progression**: Gradient Boosting model (CV R² = 0.583) trained on synthetic cohort
- **PP progression**: Fixed Franklin rate (+0.0373 PP_z/yr) based on cross-sectional data
- **eGFR progression**: PLM with explicit Layer 1 + GB residual Layer 2 (CV R² = 0.957 on residuals)

### Intervention System
- **ACE Inhibitor**: Fully implemented with cited parameters (SBP reduction -8 mmHg, eGFR protection 22%, CV mortality HR 0.74)
- **Adverse effect simulation**: Probabilistic modeling of dry cough, hypotension, and discontinuation
- **Comparative visualization**: Baseline vs treated trajectories on same chart

### User Interface
- **PyQt6 GUI**: Five-tab interface for patient input, trajectory visualization, metrics, intervention selection, and results
- **Interactive Plotly charts**: Scrollable year-by-year BP charts with slider, eGFR decline, and survival curves
- **Patient registry**: Save, load, and manage patient profiles as JSON
- **ARIA chatbot**: AI assistant (Gemini or Groq) for interpreting simulation results

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/adi5022/flowstate-ai-v2.git
cd flowstate-ai-v2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up API keys for ARIA chatbot:
```bash
cp .env.template .env
# Edit .env with your Gemini or Groq API keys
```

## Usage

### Prerequisites: Model Training

**Before running the application, you must train the ML models.** The repository does not include pre-trained models to keep the repository size small and ensure reproducibility.

#### Step 1: Generate Synthetic Cohorts
```bash
python generate_cohort.py          # Generate BP cohort (150K rows)
python generate_egfr_cohort_plm.py  # Generate eGFR cohort (130K rows)
```

#### Step 2: Train Models
```bash
cd models
python pipeline.py           # Train MAP and PP progression models
python train_egfr_plm.py     # Train eGFR PLM
python evaluate_models.py    # Evaluate all models
```

This will create the following model files in the `models/` directory:
- `progression_map.pkl` - MAP progression model
- `progression_pp.pkl` - PP progression model
- `progression_egfr_plm.pkl` - eGFR PLM model

### Running the Application
```bash
python main.py
```

The application will open with five tabs:
1. **Patient**: Enter clinical measurements and run simulation
2. **Trajectory**: View year-by-year BP progression
3. **Metrics**: View eGFR decline and survival curves
4. **Interventions**: Select intervention type and start year
5. **Intervention Results**: Compare baseline vs treated trajectories

### Running Test Simulation
```bash
python test_simulation.py
```

## Project Structure

```
flowstate-ai-v2/
├── main.py                    # Main application and simulation engine
├── aria.py                    # AI chatbot interface
├── intervention.py            # Intervention definitions and effect functions
├── patient_registry.py        # Patient management GUI
├── report_viewer.py           # Report viewer for saved simulations
├── generate_cohort.py         # Synthetic BP cohort generation
├── generate_egfr_cohort_plm.py # Synthetic eGFR cohort generation
├── models/
│   ├── pipeline.py            # BP model training pipeline
│   ├── train_egfr_plm.py      # eGFR PLM training
│   ├── train_egfr.py          # Original eGFR model (legacy)
│   └── evaluate_models.py     # Model evaluation and benchmarking
├── test_simulation.py         # Example usage
├── requirements.txt           # Python dependencies
├── .env.template              # API key template
└── aria_config.json.template  # Legacy config template
```

## How It Works

### Internal State Representation

The simulation does not track SBP and DBP directly. It uses two standardized z-scores:

```
MAP_z = (MAP − 90.11) / 12.55     where MAP = (SBP + 2×DBP) / 3
PP_z  = (PP  − 53.74) / 18.75     where PP  = SBP − DBP
```

Constants from NHANES 2017–2018 (Ostchega et al., 2020).

Reconstruction back to SBP/DBP at display time:
```
SBP = MAP + (2/3) × PP
DBP = MAP − (1/3) × PP
```

### Annual Simulation Loop

Each year, the simulation runs these steps in order:

1. Reconstruct SBP, DBP from current MAP_z and PP_z
2. Compute mortality hazard (log-additive):
   - Baseline life-table rate (Arias & Xu, 2022)
   - SBP term: β × (SBP − 115) / 20 (Lewington et al., 2002)
   - eGFR term: categorical hazard ratios (Go et al., 2004)
   - Smoking: 0.166 × pack_years / 10 (Doll et al., 2004)
3. Update cumulative survival
4. Advance MAP_z using Gradient Boosting model
5. Advance PP_z using fixed Franklin rate
6. Advance eGFR using Partial Linear Model
7. Advance waist, heart rate, age
8. Stop if survival < 1%

### eGFR Partial Linear Model

The eGFR model uses a two-layer design:

**Layer 1 (explicit clinical equations):**
- Age-specific base decline rate (Glassock & Winearls, 2009)
- Sex adjustment: Men decline 0.11 ml/min/yr faster (Eriksen & Ingebretsen, 2006)
- Sodium load: +0.0001 ml/min/yr per mg/day above 2300 threshold (de Boer et al., 2011)

**Layer 2 (Gradient Boosting residual):**
- Trained on actual cohort decline minus Layer 1 prediction
- Captures nonlinear interactions: CKD stage acceleration, smoking effects, etc.

**Note**: The MAP term was removed from the model because AASK and MDRD trials showed no significant independent BP effect on GFR slope in intention-to-treat analysis. BP affects eGFR indirectly through CKD stage progression.

## Key Constants

| Constant | Value | Source |
|---|---|---|
| MAP_MEAN | 90.11 mmHg | NHANES 2017–2018 |
| MAP_STD | 12.55 mmHg | NHANES 2017–2018 |
| PP_MEAN | 53.74 mmHg | NHANES 2017–2018 |
| PP_STD | 18.75 mmHg | NHANES 2017–2018 |
| SBP Stage 2 threshold | ≥140 mmHg | ACC/AHA 2017 |
| eGFR normal | ≥60 ml/min | KDIGO |
| eGFR CKD Stage 3 | 30–59 ml/min | KDIGO |
| PP drift rate | +0.0373 PP_z/yr | Franklin et al., 1999 |

## Model Performance

| Model | Architecture | CV R² | Target |
|---|---|---|---|
| MAP progression | Gradient Boosting | 0.583 | Annual ΔMAP_z |
| PP progression | Fixed rate | — | Annual ΔPP_z |
| eGFR progression | PLM (explicit + GB) | 0.957 | Annual eGFR decline |

## Validation

Reference patient (healthy male, age 41, SBP 115, DBP 75, eGFR 95):

Expected at year 30 (baseline, no treatment):
- SBP ≈ 139 mmHg
- eGFR ≈ 64 ml/min (just above CKD Stage 3 threshold)
- Cumulative survival ≈ 56%

## Technology Stack

- **PyQt6** — Desktop GUI
- **Plotly** — Interactive charts (rendered via QWebEngineView)
- **scikit-learn** — Gradient Boosting, Ridge regression
- **joblib** — Model serialization
- **pandas / numpy** — Data handling
- **Google Gemini or Groq** — ARIA LLM backend

## Citations

This project uses evidence-based parameters from peer-reviewed literature:

- **Lewington et al. (2002)** - Age-specific BP mortality (Lancet 360:1903)
- **Go et al. (2004)** - eGFR hazard ratios (NEJM 351:1296)
- **Glassock & Winearls (2009)** - Age-related GFR decline (Clin J Am Soc Nephrol 4:1797)
- **Eriksen & Ingebretsen (2006)** - Sex differences in GFR (Kidney Int 69:1657)
- **de Boer et al. (2011)** - Sodium load effect on eGFR (JAMA 305:2532)
- **Yusuf et al. (2000)** - ACE Inhibitor effects (NEJM 342:145)
- **Franklin et al. (1999)** - PP as risk predictor (Circulation 100:354)
- **Arias & Xu (2022)** - US life tables (NVSR 71:1)
- **Doll et al. (2004)** - Smoking mortality (BMJ 328:1519)

## License

This project is for research and educational purposes.

## Disclaimer

This is a research prototype and should not be used for clinical decision-making. The models are based on synthetic cohort data and have not been validated on real patient populations.
