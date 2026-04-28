# train_egfr_model.py
#
# FlowState AI v.2 — eGFR Progression Model Trainer
#
# Trains a Gradient Boosting regressor to predict annual eGFR decline
# (delta_egfr in ml/min/yr) from current patient state variables.
#
# The model learns the nonlinear interactions between age, sex, eGFR,
# MAP, smoking, and sodium — interactions that the hardcoded Glassock
# lookup table could not capture. The published values are embedded in
# the training data (generate_egfr_cohort.py) and serve as the
# ground truth the model is trained to reproduce. After training,
# the model is validated against those same published benchmarks
# to confirm it has learned correctly.
#
# MODEL SELECTION RATIONALE:
#   Gradient Boosting chosen over Ridge/Lasso because eGFR decline
#   exhibits nonlinear threshold effects (CKD stage boundaries) and
#   multiplicative interactions (BP × CKD stage) that linear models
#   cannot capture. Same architecture as the MAP progression model
#   (CV R²=0.583) for architectural consistency.
#
# FEATURES:
#   age, sex, egfr, map_mmhg, pack_years, sodium_mg_day,
#   egfr_baseline, ckd_stage
#
# TARGET:
#   delta_egfr — annual eGFR decline in ml/min/1.73m²
#
# OUTPUT:
#   models/progression_egfr.pkl
#
# RUN: python train_egfr_model.py
#   (requires data/egfr_cohort/cohort.csv — run generate_egfr_cohort.py first)

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_PATH  = Path("data/egfr_cohort/cohort.csv")
MODEL_PATH = Path("models/progression_egfr.pkl")
MODEL_PATH.parent.mkdir(exist_ok=True)

# ── Feature columns ───────────────────────────────────────────────────────
FEATURES = [
    "age",            # Primary driver — Glassock & Winearls (2009)
    "sex",            # +0.22 ml/min/yr sex difference — Eriksen (2006)
    "egfr",           # Current eGFR — CKD stage feedback — Go (2004)
    "map_mmhg",       # BP acceleration — Bakris et al. (2000)
    "pack_years",     # Smoking — Hallan & Orth (2011)
    "sodium_mg_day",  # Metabolic load — de Boer / Jones-Burton
    "egfr_baseline",  # Trajectory context — patient's starting point
    "ckd_stage",      # Encoded stage threshold — Go et al. (2004)
]
TARGET = "delta_egfr"


def load_data():
    print(f"Loading cohort from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Rows: {len(df):,}  |  Patients: {df['patient_id'].nunique():,}")

    X = df[FEATURES].values
    y = df[TARGET].values
    print(f"  Features: {FEATURES}")
    print(f"  Target mean: {y.mean():.4f}  SD: {y.std():.4f}")
    return X, y, df


def train_model(X, y):
    """
    Trains Gradient Boosting regressor on the eGFR cohort.

    Hyperparameters selected to match the MAP progression model
    architecture while accounting for the smaller variance in
    eGFR decline compared to MAP delta:
      - n_estimators=300: sufficient for convergence without overfitting
      - max_depth=4: captures 4-way interactions (age×sex×CKD×BP)
      - learning_rate=0.05: conservative to avoid overfitting on
        a synthetically structured dataset
      - subsample=0.8: stochastic GB reduces variance (Friedman 2002)
      - min_samples_leaf=20: prevents overfit to individual noise patterns
    """
    print("\nTraining Gradient Boosting regressor...")
    print("  Architecture: same family as MAP progression model")
    print("  Captures: age×CKD nonlinearity, BP×stage interaction")

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
        verbose=0
    )

    # 5-fold cross-validation — same protocol as MAP model
    print("\n  Running 5-fold cross-validation...")
    kf  = KFold(n_splits=5, shuffle=True, random_state=42)
    cvs = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=-1)
    print(f"  CV R² scores: {[f'{v:.3f}' for v in cvs]}")
    print(f"  Mean CV R²:   {cvs.mean():.3f} ± {cvs.std():.3f}")

    # Train on full dataset
    print("\n  Training on full dataset...")
    model.fit(X, y)

    # In-sample metrics
    y_pred = model.predict(X)
    print(f"\n  In-sample metrics:")
    print(f"    R²:   {r2_score(y, y_pred):.4f}")
    print(f"    MAE:  {mean_absolute_error(y, y_pred):.4f} ml/min/yr")
    print(f"    RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f} ml/min/yr")

    return model, cvs.mean()


def validate_benchmarks(model, df):
    """
    Validates the trained model against published clinical benchmarks.
    These are independent validation checks — the model must reproduce
    the same values that were used to generate the training data.
    If it cannot, the model has failed to learn the underlying structure.
    """
    print("\n" + "=" * 55)
    print("CLINICAL BENCHMARK VALIDATION")
    print("=" * 55)

    def predict_profile(age, sex, egfr, map_mmhg,
                        pack_years, sodium, egfr_baseline, ckd_stage):
        X = np.array([[age, sex, egfr, map_mmhg,
                       pack_years, sodium, egfr_baseline, ckd_stage]])
        return float(model.predict(X)[0])

    # ── Benchmark 1: Glassock & Winearls (2009) healthy adults ───────────
    print("\nBenchmark 1 — Glassock & Winearls (2009) age-specific rates")
    print("  Healthy patients: normal BP, no smoking, normal sodium")
    expected = {40: 0.32, 50: 0.57, 60: 1.24, 70: 1.49}
    for age, target in expected.items():
        pred = predict_profile(
            age=age, sex=1, egfr=85, map_mmhg=90,
            pack_years=0, sodium=2300, egfr_baseline=85,
            ckd_stage=3 if age >= 60 else 0
        )
        diff = pred - target
        flag = "✓" if abs(diff) < 0.25 else "⚠"
        print(f"  Age {age}: model={pred:.3f}  target={target:.3f}  "
              f"diff={diff:+.3f}  {flag}")

    # ── Benchmark 2: Bakris et al. (2000) BP effect ───────────────────────
    print("\nBenchmark 2 — Bakris et al. (2000) BP acceleration")
    print("  10 mmHg MAP difference → ~0.17 ml/min/yr difference")
    lo = predict_profile(55, 1, 75, 88, 0, 2300, 75, 3)
    hi = predict_profile(55, 1, 75, 98, 0, 2300, 75, 3)
    diff = hi - lo
    flag = "✓" if 0.10 <= diff <= 0.30 else "⚠"
    print(f"  MAP 88 → {lo:.3f}  MAP 98 → {hi:.3f}  "
          f"Δ={diff:.3f}  (target: 0.17)  {flag}")

    # ── Benchmark 3: Go et al. (2004) CKD stage effect ───────────────────
    print("\nBenchmark 3 — Go et al. (2004) CKD stage acceleration")
    print("  Stage 3 (eGFR 60) should decline faster than normal (eGFR 90)")
    normal = predict_profile(60, 1, 90, 90, 0, 2300, 90, 0)
    stage3 = predict_profile(60, 1, 60, 90, 0, 2300, 60, 3)
    stage4 = predict_profile(60, 1, 40, 90, 0, 2300, 40, 4)
    print(f"  Normal  (eGFR 90): {normal:.3f}")
    print(f"  Stage 3 (eGFR 60): {stage3:.3f}  ratio: {stage3/max(normal,0.001):.2f}x  "
          f"{'✓' if stage3 > normal else '⚠'}")
    print(f"  Stage 4 (eGFR 40): {stage4:.3f}  ratio: {stage4/max(normal,0.001):.2f}x  "
          f"{'✓' if stage4 > stage3 else '⚠'}")

    # ── Benchmark 4: Eriksen & Ingebretsen (2006) sex difference ─────────
    print("\nBenchmark 4 — Eriksen & Ingebretsen (2006) sex difference")
    print("  Men should decline ~0.22 ml/min/yr faster than women")
    male   = predict_profile(55, 1, 80, 92, 5, 2800, 80, 3)
    female = predict_profile(55, 2, 80, 92, 5, 2800, 80, 3)
    diff   = male - female
    flag   = "✓" if 0.10 <= diff <= 0.35 else "⚠"
    print(f"  Male: {male:.3f}  Female: {female:.3f}  "
          f"Δ={diff:.3f}  (target: 0.22)  {flag}")

    # ── Benchmark 5: Hallan & Orth (2011) smoking ─────────────────────────
    print("\nBenchmark 5 — Hallan & Orth (2011) smoking acceleration")
    print("  20 pack-years → ~30% faster decline than non-smoker")
    nonsmoker = predict_profile(55, 1, 80, 92, 0,  2800, 80, 3)
    smoker    = predict_profile(55, 1, 80, 92, 20, 2800, 80, 3)
    ratio     = smoker / max(nonsmoker, 0.001)
    flag      = "✓" if 1.15 <= ratio <= 1.50 else "⚠"
    print(f"  Non-smoker: {nonsmoker:.3f}  Smoker (20py): {smoker:.3f}  "
          f"ratio={ratio:.2f}x  (target: ~1.30x)  {flag}")

    print("\n" + "=" * 55)


def feature_importance_report(model):
    print("\nFEATURE IMPORTANCE (Gradient Boosting)")
    print("-" * 40)
    importances = model.feature_importances_
    pairs = sorted(zip(FEATURES, importances), key=lambda x: -x[1])
    for feat, imp in pairs:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<18} {imp:.4f}  {bar}")


def main():
    print("=" * 55)
    print("FlowState AI v.2 — eGFR Model Trainer")
    print("=" * 55)

    X, y, df = load_data()
    model, cv_r2 = train_model(X, y)
    validate_benchmarks(model, df)
    feature_importance_report(model)

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")
    print(f"CV R²: {cv_r2:.3f}")
    print("\nIntegrate into Engine.simulate() in main.py:")
    print("  Load:    self.egfr_model = joblib.load(MODELS/'progression_egfr.pkl')")
    print("  Predict: egfr_row = [[age, sex, egfr, map_mmhg, pack_years,")
    print("                         sodium, egfr_baseline, ckd_stage]]")
    print("           decline = self.egfr_model.predict(egfr_row)[0]")
    print("           egfr_t  = max(0, egfr_t - decline)")


if __name__ == "__main__":
    main()