# Trains and saves all models for FlowState AI v.2
#
# Models trained:
#   1. Baseline model  — predicts starting MAP_z and PP_z from
#      patient demographics and risk factors at entry.
#      Ridge regression. Hoerl & Kennard (1970) Technometrics 12(1):55-67
#
#   2. Progression model — predicts annual delta_MAP_z from current
#      vascular state and risk factors.
#      Compared across Ridge, Random Forest, Gradient Boosting.
#      Best model saved as progression_map.pkl
#
# All models saved to models/ directory.
# Load with joblib.load() in main.py
MAP_MEAN, MAP_STD = 90.11, 12.55
PP_MEAN,  PP_STD  = 53.74, 18.75
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

DATA    = Path("data/synthetic_cohort/cohort.csv")
MODELS  = Path("models")

# ─────────────────────────────────────────────────────────────────────────
# LOAD COHORT
# ─────────────────────────────────────────────────────────────────────────

def load_cohort():
    df = pd.read_csv(DATA)
    print(f"Loaded cohort: {len(df):,} rows, "
          f"{df.patient_id.nunique():,} patients")
    return df

# ─────────────────────────────────────────────────────────────────────────
# MODEL 1 — BASELINE
# Predicts MAP_z and PP_z at visit_year == 0 from risk factors.
# Used to initialise a new patient's vascular state from their
# clinical measurements when no prior BP history is available.
# ─────────────────────────────────────────────────────────────────────────

BASELINE_FEATURES = [
    "age", "sex", "waist_cm", "heart_rate",
    "egfr", "pack_years", "alcohol_g_day", "sodium_mg_day"
]

def train_baseline(df):
    """
    Ridge regression on baseline visit only.

    Features → MAP_z and PP_z at entry.
    Each feature has a cited quantitative effect on BP —
    see generate_cohort.py for full citations.

    Ridge chosen because waist and eGFR are moderately correlated
    (r~0.35 in this cohort). Ridge handles collinearity more
    stably than OLS.
    Hoerl & Kennard (1970) Technometrics 12(1):55-67.
    """
    baseline = df[df.visit_year == 0].copy()
    X        = baseline[BASELINE_FEATURES]
    y_map    = baseline["map_z"]
    y_pp     = baseline["pp_z"]

    def build():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=1.0))
        ])

    map_model = build()
    pp_model  = build()
    map_model.fit(X, y_map)
    pp_model.fit(X,  y_pp)

    map_r2 = r2_score(y_map, map_model.predict(X))
    pp_r2  = r2_score(y_pp,  pp_model.predict(X))

    print(f"\nBaseline model:")
    print(f"  MAP_z R²: {map_r2:.4f}")
    print(f"  PP_z  R²: {pp_r2:.4f}")

    # Sanity check — 55M, waist 95, HR 70, eGFR 85,
    # non-smoker, light drinker, moderate sodium
    test = pd.DataFrame([[55, 1, 95, 70, 85, 0, 10, 3000]],
                        columns=BASELINE_FEATURES)
    mz = map_model.predict(test)[0]
    pz = pp_model.predict(test)[0]
    MAP = mz * 12.55 + 90.11
    PP  = pz * 18.75 + 53.74
    sbp = MAP + (2/3)*PP
    dbp = MAP - (1/3)*PP
    print(f"  Sanity 55M: SBP={sbp:.1f} DBP={dbp:.1f} "
          f"(expected SBP 120-140, DBP 75-90)")

    joblib.dump(map_model, MODELS / "baseline_map.pkl")
    joblib.dump(pp_model,  MODELS / "baseline_pp.pkl")
    print(f"  Saved: baseline_map.pkl, baseline_pp.pkl")

    return map_model, pp_model

# ─────────────────────────────────────────────────────────────────────────
# MODEL 2 — PROGRESSION
# Predicts annual delta_MAP_z from current state + risk factors.
# Trained on all consecutive visit pairs in the cohort.
# ─────────────────────────────────────────────────────────────────────────

PROGRESSION_FEATURES = [
    "map_z", "pp_z", "age", "sex", "waist_cm",
    "heart_rate", "egfr", "pack_years",
    "alcohol_g_day", "sodium_mg_day"
]

def build_transitions(df):
    df_sorted = df.sort_values(["patient_id", "visit_year"])
    rows = []
    for pid, group in df_sorted.groupby("patient_id"):
        group = group.reset_index(drop=True)
        for i in range(len(group) - 1):
            curr = group.iloc[i]
            nxt  = group.iloc[i + 1]
            if nxt["visit_year"] != curr["visit_year"] + 1:
                continue
            row = {f: curr[f] for f in PROGRESSION_FEATURES}
            row["delta_map_z"] = nxt["map_z"] - curr["map_z"]
            row["delta_pp_z"]  = nxt["pp_z"]  - curr["pp_z"]
            rows.append(row)
    transitions = pd.DataFrame(rows)
    print(f"\nTransition dataset: {len(transitions):,} rows")
    print(f"  delta_MAP_z mean: {transitions.delta_map_z.mean():.4f} "
          f"std: {transitions.delta_map_z.std():.4f}")
    print(f"  delta_PP_z  mean: {transitions.delta_pp_z.mean():.4f} "
          f"std: {transitions.delta_pp_z.std():.4f}")
    return transitions

def train_progression(df):
    transitions = build_transitions(df)
    X      = transitions[PROGRESSION_FEATURES]
    y_map  = transitions["delta_map_z"]
    y_pp   = transitions["delta_pp_z"]

    # Quartile validation on MAP
    transitions["q"] = pd.qcut(
        transitions.map_z, 4,
        labels=["Q1 Low","Q2","Q3","Q4 High"]
    )
    print("\n  Quartile validation (Q4 must be highest):")
    print(transitions.groupby("q", observed=True)["delta_map_z"]
          .mean().round(4).to_string())
    transitions.drop(columns=["q"], inplace=True)

    candidates = {
        "Ridge": Pipeline([
            ("sc", StandardScaler()),
            ("m",  Ridge(alpha=1.0))
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=6,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=4,
            learning_rate=0.05, random_state=42
        ),
    }

    print(f"\n  MAP progression models:")
    print(f"  {'Model':<22} {'Train R²':>10} {'CV R²':>10}")
    print(f"  {'-'*44}")

    map_results = {}
    map_fitted  = {}

    for name, model in candidates.items():
        import copy
        m = copy.deepcopy(model)
        m.fit(X, y_map)
        train_r2 = r2_score(y_map, m.predict(X))
        cv_r2    = cross_val_score(
            m, X, y_map, cv=5, scoring="r2", n_jobs=-1
        ).mean()
        print(f"  {name:<22} {train_r2:>10.4f} {cv_r2:>10.4f}")
        map_results[name] = cv_r2
        map_fitted[name]  = m

    best_map_name  = max(map_results, key=map_results.get)
    best_map_model = map_fitted[best_map_name]
    print(f"\n  Best MAP model: {best_map_name} "
          f"(CV R² = {map_results[best_map_name]:.4f})")

    # PP model — always Ridge since Franklin rate dominates
    # Franklin et al. (1999) PMID 10421594
    print(f"\n  PP progression model (Ridge only):")
    pp_model = Pipeline([
        ("sc", StandardScaler()),
        ("m",  Ridge(alpha=1.0))
    ])
    pp_model.fit(X, y_pp)
    pp_r2    = r2_score(y_pp, pp_model.predict(X))
    pp_cv_r2 = cross_val_score(
        pp_model, X, y_pp, cv=5, scoring="r2", n_jobs=-1
    ).mean()
    print(f"  Train R²: {pp_r2:.4f}  CV R²: {pp_cv_r2:.4f}")

    # Clinical validation
    print("\n  Clinical validation — 47F SBP168 DBP102:")
    print(f"  {'Model':<22} {'Yr0':>6} {'Yr1':>6} {'Yr5':>6}")
    print(f"  {'-'*42}")

    for name, model in map_fitted.items():
        map_z = ((168 + 2*102)/3 - MAP_MEAN) / MAP_STD
        pp_z  = (168 - 102 - PP_MEAN) / PP_STD
        age   = 47.0

        sbp_traj = []
        for yr in range(6):
            MAP = map_z * MAP_STD + MAP_MEAN
            PP  = pp_z  * PP_STD  + PP_MEAN
            sbp = MAP + (2/3)*PP
            sbp_traj.append(sbp)
            row = pd.DataFrame(
                [[map_z, pp_z, age, 2, 88, 72, 52, 5, 10, 3200]],
                columns=PROGRESSION_FEATURES
            )
            map_z += model.predict(row)[0]
            pp_z  += pp_model.predict(row)[0]
            age   += 1

        print(f"  {name:<22} {sbp_traj[0]:>6.1f} "
              f"{sbp_traj[1]:>6.1f} {sbp_traj[5]:>6.1f}")

    joblib.dump(best_map_model, MODELS / "progression_map.pkl")
    joblib.dump(pp_model,       MODELS / "progression_pp.pkl")
    print(f"\n  Saved: progression_map.pkl, progression_pp.pkl")

    return best_map_model, pp_model
# ─────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*55)
    print("FlowState AI v.2 — Model Training Pipeline")
    print("="*55)

    df = load_cohort()
    train_baseline(df)
    train_progression(df)

    print("\n" + "="*55)
    print("Done. Models saved to models/")
    print("="*55)