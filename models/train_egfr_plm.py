# models/train_egfr_plm.py
#
# FlowState AI v.2 — PLM eGFR Model Trainer
#
# Trains the Gradient Boosting RESIDUAL model (Layer 2 of PLM).
# The model learns: residual_decline = actual_decline - linear_prediction
#
# At inference time in main.py:
#   linear_pred  = glassock_rate(age) + sex_delta + sodium_term
#   residual     = egfr_plm_model.predict([[age, sex, egfr, map_mmhg,
#                                           pack_years, sodium_mg_day,
#                                           egfr_baseline, ckd_stage]])[0]
#   delta_egfr   = linear_pred + residual
#   egfr_t       = max(0, egfr_t - delta_egfr)
#
# This architecture means:
#   - Layer 1 terms are always traceable to their citations
#   - Layer 2 captures nonlinear residuals (CKD stage, smoking, variation)
#   - MAP coefficient is NOT a feature because it was removed (see cohort
#     generator for justification: RENIS-T6 PMID 28245797, AASK 2002)

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

# ── Paths ─────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent.parent
DATA     = BASE / "data" / "egfr_cohort_plm" / "cohort_plm.csv"
OUT_PKL  = BASE / "models" / "progression_egfr_plm.pkl"

# ── Layer 1 functions (mirrored from cohort generator) ───────────────────
GLASSOCK_RATES = {
    (0,  40):  0.00,
    (40, 50):  0.32,
    (50, 60):  0.57,
    (60, 70):  1.24,
    (70, 80):  1.49,
    (80, 999): 3.25,
}

def glassock_rate(age):
    for (lo, hi), rate in GLASSOCK_RATES.items():
        if lo <= age < hi:
            return rate
    return 3.25

SEX_DELTA        = {1: 0.11, 2: -0.11}
SODIUM_THRESHOLD = 2300.0
SODIUM_COEF      = 0.0001

def linear_layer1(age, sex, sodium_mg_day):
    """
    Explicit Layer 1 prediction.
    Used at inference time alongside the GB residual model.
    """
    return (glassock_rate(age)
            + SEX_DELTA.get(int(sex), 0.0)
            + SODIUM_COEF * max(0.0, sodium_mg_day - SODIUM_THRESHOLD))

# ── Features for GB residual model ───────────────────────────────────────
FEATURES = [
    "age", "sex", "egfr", "map_mmhg",
    "pack_years", "sodium_mg_day",
    "egfr_baseline", "ckd_stage"
]
TARGET = "residual_decline"

# ── Load cohort ───────────────────────────────────────────────────────────
print("=" * 55)
print("FlowState AI v.2 — PLM eGFR Residual Model Trainer")
print("=" * 55)
print(f"Loading cohort from {DATA}...")

df = pd.read_csv(DATA)
print(f"  Rows: {len(df):,}  |  Patients: {df.groupby(['age']).ngroups}")
print(f"  Features: {FEATURES}")
print(f"  Target (residual): mean={df[TARGET].mean():.4f}  "
      f"SD={df[TARGET].std():.4f}")
print()

X = df[FEATURES].values
y = df[TARGET].values

# ── Train residual GB model ───────────────────────────────────────────────
print("Training Gradient Boosting residual model...")
print("  Architecture: same family as MAP progression model")
print("  Target: residual = actual_decline - Layer1_prediction")
print()

model = GradientBoostingRegressor(
    n_estimators   = 100,
    max_depth      = 4,
    learning_rate  = 0.10,
    subsample      = 0.80,
    min_samples_leaf = 20,
    random_state   = 42
)

print("  Running 5-fold cross-validation...")
cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(f"  CV R² scores: {[f'{s:.3f}' for s in cv_scores]}")
print(f"  Mean CV R²  : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print()

print("  Training on full dataset...")
model.fit(X, y)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
y_pred = model.predict(X)
print(f"  In-sample R²  : {r2_score(y, y_pred):.4f}")
print(f"  MAE (residual): {mean_absolute_error(y, y_pred):.4f} ml/min/yr")
print(f"  RMSE          : {mean_squared_error(y, y_pred)**0.5:.4f} ml/min/yr")
print()

# ── Full PLM validation (Layer 1 + Layer 2) ───────────────────────────────
print("=" * 55)
print("FULL PLM VALIDATION (Layer 1 + Layer 2 combined)")
print("=" * 55)
print()

def plm_predict(row):
    """Full PLM prediction: Layer 1 + GB residual."""
    l1 = linear_layer1(row["age"], row["sex"], row["sodium_mg_day"])
    feat = np.array([[row["age"], row["sex"], row["egfr"],
                      row["map_mmhg"], row["pack_years"],
                      row["sodium_mg_day"], row["egfr_baseline"],
                      row["ckd_stage"]]])
    residual = float(model.predict(feat)[0])
    return max(0.0, l1 + residual)

# Benchmark 1 — Glassock & Winearls (2009) age-specific rates
print("Benchmark 1 — Glassock & Winearls (2009) age-specific rates")
print("  Healthy patients: normal BP, no smoking, normal sodium")
for target_age, target_rate in [(40, 0.32), (50, 0.57), (60, 1.24), (70, 1.49)]:
    test = df[
        (df.age.between(target_age, target_age+9)) &
        (df.sex == 1) &
        (df.pack_years == 0) &
        (df.sodium_mg_day < 2400) &
        (df.egfr >= 60)
    ]
    if len(test) > 0:
        preds = test.apply(plm_predict, axis=1)
        result = preds.mean()
        diff = result - target_rate
        status = "✓" if abs(diff) < 0.4 else "⚠"
        print(f"  Age {target_age}: PLM={result:.3f}  "
              f"target={target_rate:.3f}  diff={diff:+.3f}  {status}")
print()

# Benchmark 2 — Eriksen & Ingebretsen (2006) sex difference
print("Benchmark 2 — Eriksen & Ingebretsen (2006) sex difference")
print("  Men should decline ~0.22 ml/min/yr faster than women")
males = df[(df.sex==1) & (df.pack_years==0) &
           (df.sodium_mg_day < 2400) & (df.egfr >= 60)]
females = df[(df.sex==2) & (df.pack_years==0) &
             (df.sodium_mg_day < 2400) & (df.egfr >= 60)]
if len(males) > 0 and len(females) > 0:
    m_preds = males.apply(plm_predict, axis=1).mean()
    f_preds = females.apply(plm_predict, axis=1).mean()
    diff = m_preds - f_preds
    status = "✓" if 0.15 < diff < 0.35 else "⚠"
    print(f"  Male: {m_preds:.3f}  Female: {f_preds:.3f}  "
          f"Δ={diff:.3f}  (target: 0.22)  {status}")
print()

# Benchmark 3 — Go et al. (2004) CKD stage ordering
print("Benchmark 3 — Go et al. (2004) CKD stage acceleration")
print("  Stage 3 (eGFR 30-59) should decline faster than normal (eGFR ≥ 60)")
normal = df[(df.egfr >= 60) & (df.pack_years==0) & (df.sodium_mg_day < 2400)]
stage3 = df[(df.egfr.between(30, 59)) & (df.pack_years==0) &
            (df.sodium_mg_day < 2400)]
stage4 = df[(df.egfr < 30) & (df.pack_years==0) & (df.sodium_mg_day < 2400)]
if len(normal) > 0 and len(stage3) > 0:
    n_pred = normal.apply(plm_predict, axis=1).mean()
    s3_pred = stage3.apply(plm_predict, axis=1).mean()
    ratio = s3_pred / n_pred if n_pred > 0 else 0
    status = "✓" if s3_pred > n_pred else "✗"
    print(f"  Normal (eGFR ≥60): {n_pred:.3f}  "
          f"Stage 3 (eGFR 30-59): {s3_pred:.3f}  "
          f"ratio: {ratio:.2f}x  {status}")
if len(stage4) > 0 and len(stage3) > 0:
    s4_pred = stage4.apply(plm_predict, axis=1).mean()
    ratio = s4_pred / s3_pred if s3_pred > 0 else 0
    status = "✓" if s4_pred > s3_pred else "✗"
    print(f"  Stage 4 (eGFR <30): {s4_pred:.3f}  "
          f"ratio vs Stage3: {ratio:.2f}x  {status}")
print()

# Benchmark 4 — Hallan & Orth (2011) smoking
print("Benchmark 4 — Hallan & Orth (2011) smoking acceleration")
print("  20 pack-years → ~30% faster decline than non-smoker")
nonsmoker = df[(df.pack_years == 0) & (df.egfr >= 60) &
               (df.sodium_mg_day < 2400)]
smoker = df[(df.pack_years.between(18, 22)) & (df.egfr >= 60) &
            (df.sodium_mg_day < 2400)]
if len(nonsmoker) > 0 and len(smoker) > 0:
    ns_pred = nonsmoker.apply(plm_predict, axis=1).mean()
    s_pred  = smoker.apply(plm_predict, axis=1).mean()
    ratio   = s_pred / ns_pred if ns_pred > 0 else 0
    status  = "✓" if 1.15 < ratio < 1.50 else "⚠"
    print(f"  Non-smoker: {ns_pred:.3f}  "
          f"Smoker (20py): {s_pred:.3f}  "
          f"ratio={ratio:.2f}x  (target: ~1.30x)  {status}")
print()

# ── Feature importance of GB residual model ───────────────────────────────
print("=" * 55)
print("RESIDUAL MODEL FEATURE IMPORTANCE")
print("-" * 40)
importances = model.feature_importances_
for feat, imp in sorted(zip(FEATURES, importances),
                        key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 30)
    print(f"  {feat:<20} {imp:.4f}  {bar}")
print()
print("Note: age/sex/sodium in residual model should have LOW importance")
print("      because Layer 1 already handles them explicitly.")
print("      CKD stage and smoking should dominate the residual.")
print()

# ── Save model ────────────────────────────────────────────────────────────
joblib.dump(model, OUT_PKL)
print(f"Residual model saved → {OUT_PKL}")
print(f"CV R²: {cv_scores.mean():.3f}")
print()
print("=" * 55)
print("INFERENCE INTEGRATION (for main.py)")
print("=" * 55)
print("""
# Load alongside existing egfr model for parallel testing:
self.egfr_plm_model = joblib.load(MODELS / 'progression_egfr_plm.pkl')

# At inference time (replace existing eGFR block):
# --- Layer 1: explicit linear terms ---
from models.train_egfr_plm import linear_layer1
lin_pred = linear_layer1(age_t, sex_t, sodium_t)

# --- Layer 2: GB residual ---
egfr_feat = np.array([[age_t, sex_t, egfr_t, MAP_t,
                        pack_t, sodium_t,
                        egfr_baseline, ckd_stage_t]])
residual = float(self.egfr_plm_model.predict(egfr_feat)[0])

# --- Combined PLM prediction ---
egfr_decline_plm = max(0.0, lin_pred + residual)
egfr_t = max(0, egfr_t - egfr_decline_plm)
""")