# FlowState AI v.2

A patient-specific cardiovascular digital twin for hypertension progression modeling and intervention simulation.

## Features

- **Physiological Modeling**: MAP (Mean Arterial Pressure) and PP (Pulse Pressure) z-score representation
- **eGFR Progression**: Partial Linear Model (PLM) combining explicit clinical equations with Gradient Boosting residuals
- **Mortality Modeling**: Log-additive hazard model with evidence-based risk factors
- **Intervention System**: ACE Inhibitor with cited clinical parameters
- **Patient Registry**: GUI for managing patient profiles
- **Interactive Visualization**: Plotly-based trajectory visualization
- **AI Assistant**: ARIA chatbot for simulation interpretation

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys (optional, for ARIA chatbot):
```bash
cp .env.template .env
# Edit .env with your Gemini or Groq API keys
```

## Usage

### Run the Application
```bash
python main.py
```

### Generate Synthetic Cohort
```bash
python generate_cohort.py
python generate_egfr_cohort_plm.py
```

### Train Models
```bash
cd models
python pipeline.py
python train_egfr_plm.py
```

### Test Simulation
```bash
python test_simulation.py
```

## Project Structure

```
flowstate-ai-v2/
├── main.py                    # Main application and simulation engine
├── aria.py                    # AI chatbot interface
├── intervention.py            # Intervention definitions
├── patient_registry.py        # Patient management GUI
├── report_viewer.py           # Report viewer
├── generate_cohort.py         # Synthetic BP cohort generation
├── generate_egfr_cohort_plm.py # Synthetic eGFR cohort generation
├── models/
│   ├── pipeline.py            # BP model training pipeline
│   ├── train_egfr_plm.py      # eGFR PLM training
│   └── evaluate_models.py     # Model evaluation
└── test_simulation.py         # Example usage
```

## Citation

This project uses evidence-based parameters from peer-reviewed literature including:
- Lewington et al. (2002) - Age-specific BP mortality
- Go et al. (2004) - eGFR hazard ratios
- Glassock & Winearls (2009) - Age-related GFR decline
- Eriksen & Ingebretsen (2006) - Sex differences in GFR
- de Boer et al. (2011) - Sodium load effect on eGFR
- Yusuf et al. (2000) - ACE Inhibitor effects

## License

This project is for research and educational purposes.

## Disclaimer

This is a research prototype and should not be used for clinical decision-making.
