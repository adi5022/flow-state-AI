# generate_egfr_cohort_plm.py
#
# FlowState AI v.2 — PLM eGFR Cohort Generator
#
# Generates a synthetic eGFR longitudinal cohort for training
# the Partial Linear Model (PLM) eGFR progression model.
#
# Architecture:
#   Layer 1 (explicit linear terms, NOT learned by model):
#     - Age-specific base rates     Glassock & Winearls (2009) PMID 19768194
#     - Sex difference              Eriksen & Ingebretsen (2006) PMID 16407643
#     - Sodium load above 2300      de Boer et al. (2011) JAMA 305:2532
#
#   Layer 2 (GB residual, what the model DOES learn):
#     - CKD stage nonlinearity      Go et al. (2004) NEJM 351:1296
#     - Smoking acceleration        Hallan & Orth (2011) PMID 21486100
#     - Individual variation        Laird & Ware (1982) Biometrics 38:963
#     - Within-visit noise          Stevens et al. (2006) AJKD 48:11
#
#   MAP acceleration term: REMOVED
#     Justification: RENIS-T6 (Eriksen et al. 2017, BMC Nephrology 18:77,
#     PMID 28245797) found no significant BP-GFR association in normotensive
#     general population. AASK RCT (Wright et al. 2002, JAMA 288:2421)
#     found no significant difference in GFR slope between lower MAP
#     (92 mmHg) and usual MAP (102-107 mmHg) groups in ITT analysis.
#     No open-access RCT provides a verified continuous MAP-eGFR
#     coefficient applicable to the general untreated population.
#
# Output columns:
#   age, sex, egfr, map_mmhg, pack_years, sodium_mg_day,
#   egfr_baseline, ckd_stage,
#   linear_prediction,   <- Layer 1 explicit prediction
#   actual_decline,      <- total decline (linear + residual + noise)
#   residual_decline     <- target for GB: actual - linear

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# ── Constants ─────────────────────────────────────────────────────────────
N_PATIENTS  = 5000
MAX_YEARS   = 30
OUT_DIR     = Path("data/egfr_cohort_plm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Layer 1 coefficients ──────────────────────────────────────────────────

# Glassock & Winearls (2009) PMID 19768194
# Age-specific annual eGFR decline rates from Baltimore Longitudinal Study
GLASSOCK_RATES = {
    (0,  40):  0.00,
    (40, 50):  0.32,
    (50, 60):  0.57,
    (60, 70):  1.24,
    (70, 80):  1.49,
    (80, 999): 3.25,
}

def glassock_rate(age):
    """Annual eGFR decline from age-specific Glassock rates."""
    for (lo, hi), rate in GLASSOCK_RATES.items():
        if lo <= age < hi:
            return rate
    return 3.25

# Eriksen & Ingebretsen (2006) PMID 16407643, Table 5
# Men decline 0.22 ml/min/yr faster than women
# Split symmetrically: men +0.11, women -0.11
SEX_DELTA = {1: 0.11, 2: -0.11}  # 1=Male, 2=Female

# de Boer et al. (2011) JAMA 305:2532
# Sodium load above 2300 mg/day contributes to eGFR decline
# Coefficient: 0.0001 ml/min/yr per mg/day above 2300
SODIUM_THRESHOLD  = 2300.0
SODIUM_COEF       = 0.0001

def linear_prediction(age, sex, sodium_mg_day):
    """
    Layer 1: explicit sum of all cited linear effects.
    This is subtracted from actual_decline to produce the GB target.
    """
    age_term    = glassock_rate(age)
    sex_term    = SEX_DELTA.get(int(sex), 0.0)
    sodium_term = SODIUM_COEF * max(0.0, sodium_mg_day - SODIUM_THRESHOLD)
    return age_term + sex_term + sodium_term

# ── Layer 2 multipliers (Go et al. 2004) ─────────────────────────────────
# CKD stage nonlinearity captured in residual layer
# Go et al. (2004) NEJM 351:1296 — adjusted for model independence from
# linear terms to avoid double-counting age effects.
# Applied as multiplier to the BASE age rate only (not to sex/sodium terms).
CKD_MULTIPLIER = {
    0: 1.00,   # eGFR >= 60: reference
    3: 1.60,   # eGFR 30-59: 1.6x Go et al.
    4: 2.20,   # eGFR < 30:  2.2x Go et al.
}

def ckd_stage(egfr):
    if egfr >= 60: return 0
    if egfr >= 30: return 3
    return 4

# ── Smoking (Hallan & Orth 2011, PMID 21486100) ──────────────────────────
# Log-linear: 1.30x at 20 pack-years
# ln(1.30) / 20 = 0.01310 per pack-year
SMOKING_COEF = 0.01310

def smoking_multiplier(pack_years):
    return np.exp(SMOKING_COEF * pack_years)

# ── Noise structure (Laird & Ware 1982; Stevens et al. 2006) ─────────────
BETWEEN_PERSON_SD = 0.40   # ml/min/yr individual trajectory variation
WITHIN_VISIT_SD   = 0.25   # ml/min/yr measurement noise

# ── Baseline distributions ────────────────────────────────────────────────
# NHANES 2017-2018 / Saran et al. 2020 AJKD 77:A7
def sample_baseline():
    age         = np.random.uniform(35, 75)
    sex         = np.random.choice([1, 2])
    egfr        = np.clip(np.random.normal(90, 18), 15, 130)
    pack_years  = np.clip(np.random.exponential(8), 0, 80)
    sodium      = np.clip(np.random.normal(3400, 800), 1000, 6000)
    map_mmhg    = np.clip(np.random.normal(90.11, 12.55), 60, 140)
    return age, sex, egfr, pack_years, sodium, map_mmhg

# ── Cohort generation ─────────────────────────────────────────────────────
print("=" * 60)
print("FlowState AI v.2 — PLM eGFR Cohort Generator")
print("=" * 60)
print(f"Patients : {N_PATIENTS:,}")
print(f"Max years: {MAX_YEARS}")
print()

rows = []
n_complete = 0

for pid in range(N_PATIENTS):
    if (pid + 1) % 1000 == 0:
        print(f"  {pid+1:,} / {N_PATIENTS:,} patients...")

    age0, sex0, egfr0, pack0, sodium0, map0 = sample_baseline()

    # Per-patient random effect (between-person variation)
    # Laird & Ware (1982): individual has a persistent offset from mean
    personal_offset = np.random.normal(0, BETWEEN_PERSON_SD)

    egfr_t       = float(egfr0)
    egfr_baseline = float(egfr0)
    age_t        = float(age0)
    map_t        = float(map0)

    for yr in range(MAX_YEARS):
        if egfr_t <= 0:
            break

        stage = ckd_stage(egfr_t)

        # ── Layer 1: explicit linear prediction ───────────────────────
        lin_pred = linear_prediction(age_t, sex0, sodium0)

        # ── Layer 2: actual decline with nonlinear and noisy effects ──
        # Base age rate × CKD multiplier (Go et al.)
        base_rate = glassock_rate(age_t) * CKD_MULTIPLIER[stage]

        # Smoking multiplier (Hallan & Orth 2011)
        smoke_mult = smoking_multiplier(pack0)

        # Full actual decline with all effects
        actual = (
            base_rate * smoke_mult           # age × CKD × smoking
            + SEX_DELTA.get(int(sex0), 0.0)  # sex term
            + SODIUM_COEF * max(0.0, sodium0 - SODIUM_THRESHOLD)  # sodium
            + personal_offset                # between-person variation
            + np.random.normal(0, WITHIN_VISIT_SD)  # within-visit noise
        )
        actual = max(0.0, actual)

        # ── Residual: what GB must learn ──────────────────────────────
        # This is everything EXCEPT the clean linear Layer 1 prediction
        residual = actual - lin_pred

        rows.append({
            "age":              round(age_t, 1),
            "sex":              int(sex0),
            "egfr":             round(egfr_t, 2),
            "map_mmhg":         round(map_t, 1),
            "pack_years":       round(pack0, 1),
            "sodium_mg_day":    round(sodium0, 0),
            "egfr_baseline":    round(egfr_baseline, 2),
            "ckd_stage":        stage,
            "linear_prediction": round(lin_pred, 4),
            "actual_decline":   round(actual, 4),
            "residual_decline": round(residual, 4),
        })

        egfr_t  = max(0, egfr_t - actual)
        age_t  += 1

    n_complete += 1

# ── Save ──────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
out_path = OUT_DIR / "cohort_plm.csv"
df.to_csv(out_path, index=False)

print()
print(f"Rows generated : {len(df):,}")
print(f"Patients       : {n_complete:,}")
print(f"Mean yrs/patient: {len(df)/n_complete:.1f}")
print()

# ── Validation checks ─────────────────────────────────────────────────────
print("Layer 1 linear_prediction validation:")
healthy_50 = df[(df.age.between(50,59)) &
                (df.sex==1) &
                (df.pack_years==0) &
                (df.sodium_mg_day < 2400) &
                (df.egfr >= 60)]
if len(healthy_50) > 0:
    print(f"  Healthy male 50-59: linear_pred mean = "
          f"{healthy_50.linear_prediction.mean():.3f} ml/min/yr "
          f"(Glassock target: 0.57)")

healthy_40 = df[(df.age.between(40,49)) &
                (df.sex==1) &
                (df.pack_years==0) &
                (df.sodium_mg_day < 2400) &
                (df.egfr >= 60)]
if len(healthy_40) > 0:
    print(f"  Healthy male 40-49: linear_pred mean = "
          f"{healthy_40.linear_prediction.mean():.3f} ml/min/yr "
          f"(Glassock target: 0.32)")

print()
print("Residual statistics (target for GB model):")
print(f"  Mean residual : {df.residual_decline.mean():.4f} ml/min/yr")
print(f"  SD residual   : {df.residual_decline.std():.4f} ml/min/yr")
print(f"  Min residual  : {df.residual_decline.min():.4f}")
print(f"  Max residual  : {df.residual_decline.max():.4f}")

print()
print(f"Saved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
print("=" * 60)