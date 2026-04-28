# main.py
#
# FlowState AI v.2 — Main Simulation UI
# Patient-Specific Digital Twin for Hypertension
#
# Run: python main.py

import sys
import json
import time
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import date

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTabWidget, QSpinBox, QFileDialog,
    QMessageBox, QComboBox, QDoubleSpinBox, QSlider, QCheckBox, QFrame,
    QGridLayout, QLineEdit, QTextEdit, QInputDialog, QSplitter,
    QProgressBar, QSizePolicy, QScrollArea, QGroupBox, QRadioButton,
    QButtonGroup, QToolButton, QSpacerItem, QDateEdit, QTimeEdit,
    QCalendarWidget, QDockWidget, QStatusBar, QToolBar, QMenuBar,
    QMenu, QTabBar, QStyle, QStyleFactory, QAbstractSpinBox
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QTimer
from PyQt6.QtGui import QFont

from patient_registry import PatientRegistry
from aria import ARIAPanel
from intervention import (Intervention, apply_sbp_effect, apply_egfr_protection, 
                          apply_cv_mortality_modifier, check_adverse_effects, 
                          apply_intervention_continuity, list_available_interventions, 
                          get_intervention)

# ── Paths ─────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
MODELS = BASE / "models"

# ── Population constants ──────────────────────────────────────────────────
MAP_MEAN, MAP_STD = 90.11, 12.55
PP_MEAN,  PP_STD  = 53.74, 18.75

# ── Life tables: Arias & Xu (2022) ───────────────────────────────────────
QX_MALE = {
    35:0.001932, 40:0.002585, 45:0.003681, 50:0.005737,
    55:0.008729, 60:0.013599, 65:0.020636, 70:0.032154,
    75:0.049400, 80:0.076066, 85:0.116485
}
QX_FEMALE = {
    35:0.001186, 40:0.001579, 45:0.002350, 50:0.003637,
    55:0.005571, 60:0.007928, 65:0.012261, 70:0.019826,
    75:0.031721, 80:0.052210, 85:0.085374
}
LEWINGTON_BETA = {40:0.706, 50:0.693, 60:0.615, 70:0.513, 80:0.398}

def baseline_qx(age, sex):
    t    = QX_MALE if sex == 1 else QX_FEMALE
    age  = float(np.clip(age, 35, 85))
    ages = sorted(t.keys())
    lo   = max(a for a in ages if a <= age)
    hi   = min(a for a in ages if a >= age)
    if lo == hi:
        return t[lo]
    return t[lo] + (age-lo)/(hi-lo)*(t[hi]-t[lo])

# ── Engine ────────────────────────────────────────────────────────────────
PROGRESSION_FEATURES = [
    "map_z", "pp_z", "age", "sex", "waist_cm",
    "heart_rate", "egfr", "pack_years",
    "alcohol_g_day", "sodium_mg_day"
]

class Engine:
    def __init__(self):
        self.prog_map = joblib.load(MODELS / "progression_map.pkl")
        self.prog_pp  = joblib.load(MODELS / "progression_pp.pkl")
        self.egfr_model = joblib.load(MODELS / "progression_egfr_plm.pkl")

    def simulate(self, age, sex, sbp, dbp, waist_cm, heart_rate,
                 egfr, pack_years, alcohol_g_day, sodium_mg_day, years=30):
        MAP   = (sbp + 2*dbp) / 3
        PP    = sbp - dbp
        map_z = (MAP - MAP_MEAN) / MAP_STD
        pp_z  = (PP  - PP_MEAN)  / PP_STD
        egfr_baseline = float(egfr)

        age_t      = float(age)
        sex_t      = int(sex)
        waist_t    = float(waist_cm)
        hr_t       = float(heart_rate)
        egfr_t     = float(egfr)
        pack_t     = float(pack_years)
        alcohol_t  = float(alcohol_g_day)
        sodium_t   = float(sodium_mg_day)
        survival   = 1.0

        res = {
            "year": [], "age": [], "sbp": [], "dbp": [],
            "pp": [], "egfr": [], "survival": [],
            "death_prob_annual": []
        }

        for yr in range(int(years) + 1):
            MAP_t = map_z * MAP_STD + MAP_MEAN
            PP_t  = pp_z  * PP_STD  + PP_MEAN
            s     = MAP_t + (2/3)*PP_t
            d     = MAP_t - (1/3)*PP_t

            age_dec  = int(np.clip((age_t//10)*10, 40, 80))
            beta_sbp = LEWINGTON_BETA.get(age_dec, 0.398)
            # Lewington (2002): hazard ratio validated above 115 mmHg.
            # Floor SBP contribution at zero — no protective effect
            # below 115 mmHg as this extrapolates beyond the study range.
            sbp_term = max(0.0, beta_sbp * (s - 115) / 20)
            
            # Go et al. (2004) NEJM 351:1296 Table 2 — categorical hazard ratios
            # HR: eGFR 45-59 = 1.2, 30-44 = 1.8, 15-29 = 3.2, <15 = 5.9
            # Applied as additive log-hazard terms directly from paper.
            # No linear approximation — exact categorical values.
            if   egfr_t >= 60: egfr_hr = 0.0
            elif egfr_t >= 45: egfr_hr = np.log(1.2)
            elif egfr_t >= 30: egfr_hr = np.log(1.8)
            elif egfr_t >= 15: egfr_hr = np.log(3.2)
            else:              egfr_hr = np.log(5.9)

            # Smoking coefficient: 0.166 per 10 pack-years (0.0166 per pack-year)
            # Source: Lubin et al. (2016) Epidemiology 27(3):395-404 - excess RR/pack-year range 0.011-0.046
            log_h    = (np.log(baseline_qx(age_t, sex_t))
                        + sbp_term
                        + egfr_hr
                        + 0.166 * pack_t / 10) 
            p_death  = float(np.clip(1 - np.exp(-np.exp(log_h)), 0, 0.999))
            survival *= (1 - p_death)

            res["year"].append(yr)
            res["age"].append(round(age_t, 1))
            res["sbp"].append(round(s, 1))
            res["dbp"].append(round(d, 1))
            res["pp"].append(round(PP_t, 1))
            res["egfr"].append(round(egfr_t, 1))
            res["survival"].append(round(survival * 100, 1))
            res["death_prob_annual"].append(round(p_death * 100, 2))

            if survival < 0.01:
                break

            if yr < int(years):
                row = pd.DataFrame(
                    [[map_z, pp_z, age_t, sex_t, waist_t,
                      hr_t, egfr_t, pack_t, alcohol_t, sodium_t]],
                    columns=PROGRESSION_FEATURES
                )
                map_z  += self.prog_map.predict(row)[0]
                pp_z   += self.prog_pp.predict(row)[0]
                        
                # eGFR progression — Partial Linear Model (PLM)
                # Layer 1 (explicit, cited): Glassock 2009, Eriksen 2006, de Boer 2011
                # Layer 2 (GB residual): CKD nonlinearity, smoking, individual variation
                # MAP term removed: RENIS-T6 (PMID 28245797) + AASK (JAMA 2002)
                GLASSOCK_PLM = [(0,40,0.00),(40,50,0.32),(50,60,0.57),
                                (60,70,1.24),(70,80,1.49),(80,999,3.25)]
                def _glassock(age):
                    for lo,hi,r in GLASSOCK_PLM:
                        if lo <= age < hi: return r
                    return 3.25

                lin_pred    = (_glassock(age_t)
                            + {1:0.11,2:-0.11}.get(int(sex_t),0.0)
                            + 0.0001 * max(0.0, sodium_t - 2300.0))
                ckd_stage_t = (0 if egfr_t >= 60 else 3 if egfr_t >= 30 else 4)
                egfr_feat   = np.array([[age_t, sex_t, egfr_t, MAP_t,
                                        pack_t, sodium_t,
                                        egfr_baseline, ckd_stage_t]])
                residual    = float(self.egfr_model.predict(egfr_feat)[0])
                egfr_decline = max(0.0, lin_pred + residual)
                egfr_t      = max(0, egfr_t - egfr_decline)


                # Waist drift: Janssen et al. (2004) PMID 14985210
                # Gain is baseline-dependent — lean patients gain ~0.15/yr,
                # obese patients gain ~0.60/yr, linear interpolation between.
                # Reference thresholds: 80 cm (lean) → 110 cm (obese)
                if age_t < 65:
                    waist_lean  = 80.0
                    waist_obese = 110.0
                    rate_lean   = 0.15
                    rate_obese  = 0.60
                    t = float(np.clip(
                        (waist_t - waist_lean) / (waist_obese - waist_lean),
                        0.0, 1.0))
                    waist_gain = rate_lean + t * (rate_obese - rate_lean)
                    waist_t += waist_gain
                if age_t > 50:
                    hr_t = max(45, hr_t - 0.4)
                age_t += 1

        return res

    def simulate_with_intervention(self, age, sex, sbp, dbp, waist_cm, heart_rate,
                                  egfr, pack_years, alcohol_g_day, sodium_mg_day, 
                                  intervention: Intervention, years=30):
        """
        Simulate patient trajectory with medication intervention.
        
        This method runs two simulations:
        1. Baseline trajectory (no treatment) using existing simulate() method
        2. Treated trajectory with intervention applied from intervention.start_year
        
        The baseline simulation is preserved completely, while the treated
        simulation applies evidence-based effects from the intervention year onward.
        
        Args:
            age, sex, sbp, dbp, waist_cm, heart_rate, egfr, pack_years, 
            alcohol_g_day, sodium_mg_day: Patient baseline parameters
            intervention: Intervention configuration with effects and start year
            years: Total simulation duration (default 30)
            
        Returns:
            tuple: (baseline_results, treated_results) - Both are standard 
                   result dictionaries with keys: year, age, sbp, dbp, pp, 
                   egfr, survival, death_prob_annual
                   
        Example:
            >>> from intervention import get_intervention
            >>> ace = get_intervention("ACE Inhibitor")
            >>> ace.start_year = 5
            >>> baseline, treated = engine.simulate_with_intervention(
            ...     age=41, sex=1, sbp=115, dbp=75, waist_cm=85, 
            ...     heart_rate=65, egfr=95, pack_years=0, alcohol_g_day=5, 
            ...     sodium_mg_day=2300, intervention=ace)
        """
        
        # Step 1: Run baseline simulation (unchanged)
        baseline_results = self.simulate(
            age, sex, sbp, dbp, waist_cm, heart_rate,
            egfr, pack_years, alcohol_g_day, sodium_mg_day, years
        )
        
        # Step 2: Run treated simulation with intervention
        # Initialize with same baseline parameters
        MAP   = (sbp + 2*dbp) / 3
        PP    = sbp - dbp
        map_z = (MAP - MAP_MEAN) / MAP_STD
        pp_z  = (PP  - PP_MEAN)  / PP_STD
        egfr_baseline = float(egfr)

        age_t      = float(age)
        sex_t      = int(sex)
        waist_t    = float(waist_cm)
        hr_t       = float(heart_rate)
        egfr_t     = float(egfr)
        pack_t     = float(pack_years)
        alcohol_t  = float(alcohol_g_day)
        sodium_t   = float(sodium_mg_day)
        survival   = 1.0

        treated_res = {
            "year": [], "age": [], "sbp": [], "dbp": [],
            "pp": [], "egfr": [], "survival": [],
            "death_prob_annual": []
        }
        
        intervention_year = intervention.start_year
        sbp_applied = False
        intervention_active = False
        year_on_treatment = 0
        adverse_effects_history = []
        
        for yr in range(int(years) + 1):
            # Check if intervention should start/continue
            if yr >= intervention_year and not intervention_active:
                intervention_active = True
                year_on_treatment = 0
            
            # Apply intervention effects from intervention year onward
            if intervention_active and not sbp_applied:
                # Apply SBP reduction to MAP_z (once at intervention year)
                map_z = apply_sbp_effect(map_z, intervention.delta_sbp)
                sbp_applied = True
            
            # Calculate current blood pressure values
            MAP_t = map_z * MAP_STD + MAP_MEAN
            PP_t  = pp_z  * PP_STD  + PP_MEAN
            s     = MAP_t + (2/3)*PP_t  # SBP
            d     = MAP_t - (1/3)*PP_t  # DBP

            # Check adverse effects if intervention is active
            if intervention_active:
                adverse_effects = check_adverse_effects(
                    intervention, year_on_treatment, s, egfr_t
                )
                adverse_effects_history.append({
                    'year': yr,
                    'effects': adverse_effects
                })
                
                # Apply clinical decision logic
                intervention_active = apply_intervention_continuity(
                    intervention, intervention_active, adverse_effects
                )
                
                # If discontinued, reverse SBP effect (gradual return to baseline)
                if not intervention_active and sbp_applied:
                    # Reverse the SBP effect over 2 years
                    if year_on_treatment > 0:
                        reversal_factor = min(1.0, 2.0 / (yr - intervention_year + 1))
                        map_z = map_z - (apply_sbp_effect(0, intervention.delta_sbp) - 0) * reversal_factor
                
                year_on_treatment += 1

            # Calculate mortality hazard
            age_dec  = int(np.clip((age_t//10)*10, 40, 80))
            beta_sbp = LEWINGTON_BETA.get(age_dec, 0.398)
            # Lewington (2002): hazard ratio validated above 115 mmHg.
            # Floor SBP contribution at zero — no protective effect
            # below 115 mmHg as this extrapolates beyond the study range.
            sbp_term = max(0.0, beta_sbp * (s - 115) / 20)
            
            if   egfr_t >= 60: egfr_hr = 0.0
            elif egfr_t >= 45: egfr_hr = np.log(1.2)
            elif egfr_t >= 30: egfr_hr = np.log(1.8)
            elif egfr_t >= 15: egfr_hr = np.log(3.2)
            else:              egfr_hr = np.log(5.9)

            # Smoking coefficient: 0.166 per 10 pack-years (0.0166 per pack-year)
            # Source: Lubin et al. (2016) Epidemiology 27(3):395-404 - excess RR/pack-year range 0.011-0.046
            log_h    = (np.log(baseline_qx(age_t, sex_t))
                        + sbp_term
                        + egfr_hr
                        + 0.166 * pack_t / 10)

            # Apply CV mortality modifier from intervention year onward
            if yr >= intervention_year:
                log_h = apply_cv_mortality_modifier(log_h, intervention.cv_hr_modifier)
                
            p_death  = float(np.clip(1 - np.exp(-np.exp(log_h)), 0, 0.999))
            survival *= (1 - p_death)

            # Store results
            treated_res["year"].append(yr)
            treated_res["age"].append(round(age_t, 1))
            treated_res["sbp"].append(round(s, 1))
            treated_res["dbp"].append(round(d, 1))
            treated_res["pp"].append(round(PP_t, 1))
            treated_res["egfr"].append(round(egfr_t, 1))
            treated_res["survival"].append(round(survival * 100, 1))
            treated_res["death_prob_annual"].append(round(p_death * 100, 2))

            if survival < 0.01:
                break

            # Progress to next year (only if not final year)
            if yr < int(years):
                row = pd.DataFrame(
                    [[map_z, pp_z, age_t, sex_t, waist_t,
                      hr_t, egfr_t, pack_t, alcohol_t, sodium_t]],
                    columns=PROGRESSION_FEATURES
                )
                map_z  += self.prog_map.predict(row)[0]
                pp_z   += self.prog_pp.predict(row)[0]
                        
                # eGFR progression with intervention protection
                GLASSOCK_PLM = [(0,40,0.00),(40,50,0.32),(50,60,0.57),
                                (60,70,1.24),(70,80,1.49),(80,999,3.25)]
                def _glassock(age):
                    for lo,hi,r in GLASSOCK_PLM:
                        if lo <= age < hi: return r
                    return 3.25

                lin_pred    = (_glassock(age_t)
                            + {1:0.11,2:-0.11}.get(int(sex_t),0.0)
                            + 0.0001 * max(0.0, sodium_t - 2300.0))
                ckd_stage_t = (0 if egfr_t >= 60 else 3 if egfr_t >= 30 else 4)
                egfr_feat   = np.array([[age_t, sex_t, egfr_t, MAP_t,
                                        pack_t, sodium_t,
                                        egfr_baseline, ckd_stage_t]])
                residual    = float(self.egfr_model.predict(egfr_feat)[0])
                egfr_decline = max(0.0, lin_pred + residual)
                
                # Apply eGFR protection from intervention year onward
                if yr >= intervention_year:
                    egfr_decline = apply_egfr_protection(egfr_decline, intervention.egfr_protection)
                    
                egfr_t      = max(0, egfr_t - egfr_decline)

                # Waist drift and other age-related changes
                if age_t < 65:
                    waist_lean  = 80.0
                    waist_obese = 110.0
                    rate_lean   = 0.15
                    rate_obese  = 0.60
                    t = float(np.clip(
                        (waist_t - waist_lean) / (waist_obese - waist_lean),
                        0.0, 1.0))
                    waist_gain = rate_lean + t * (rate_obese - rate_lean)
                    waist_t += waist_gain
                if age_t > 50:
                    hr_t = max(45, hr_t - 0.4)
                age_t += 1

        # Add adverse effects history to treated results
        treated_res["adverse_effects"] = adverse_effects_history
        treated_res["intervention_discontinued"] = not intervention_active
        
        return baseline_results, treated_res

# ── Worker thread ─────────────────────────────────────────────────────────
class SimWorker(QThread):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)

    def __init__(self, engine, params):
        super().__init__()
        self.engine = engine
        self.params = params

    def run(self):
        for i in range(1, 85):
            self.progress.emit(i)
            time.sleep(0.008)
        results = self.engine.simulate(**self.params)
        self.progress.emit(100)
        self.finished.emit(results)

# ── Chart builder ─────────────────────────────────────────────────────────
C = {
    "sbp":      "#ef4444",
    "dbp":      "#f97316",
    "pp":       "#a855f7",
    "egfr":     "#22d3ee",
    "survival": "#4ade80",
    "bg":       "#12121a",
    "border":   "#1e1e2e",
    "text":     "#e2e8f0",
    "muted":    "#64748b",
    "accent":   "#6366f1",
}

def base_layout(title):
    return dict(
        title=dict(text=title,
                   font=dict(color=C["text"], size=13,
                             family="Georgia, serif"),
                   x=0.01),
        paper_bgcolor=C["bg"],
        plot_bgcolor =C["bg"],
        font=dict(color=C["text"], family="Georgia, serif"),
        margin=dict(l=50, r=20, t=50, b=40),
        xaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"],
                   tickfont=dict(color=C["muted"], size=10)),
        yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"],
                   tickfont=dict(color=C["muted"], size=10)),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=C["bg"], font_color=C["text"],
                        bordercolor=C["border"]),
        legend=dict(orientation="h", y=1.08, x=0,
                    font=dict(color=C["text"], size=11),
                    bgcolor="rgba(0,0,0,0)")
    )

def build_bp_chart(results, highlight=0):
    years    = results["year"]
    sbp_full = results["sbp"]
    dbp_full = results["dbp"]
    pp_full  = results["pp"]

    # Slice up to current year
    yr     = min(highlight, len(years) - 1)
    x      = years[:yr+1]
    sbp    = sbp_full[:yr+1]
    dbp    = dbp_full[:yr+1]
    pp     = pp_full[:yr+1]

    fig = go.Figure()

    # Faded full trajectory hint
    fig.add_trace(go.Scatter(
        x=years, y=sbp_full, name="",
        line=dict(color="rgba(239,68,68,0.12)", width=1.5, dash="dot"),
        mode="lines", showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=years, y=dbp_full, name="",
        line=dict(color="rgba(249,115,22,0.12)", width=1.5, dash="dot"),
        mode="lines", showlegend=False, hoverinfo="skip"))

    # Active trajectory
    fig.add_trace(go.Scatter(
        x=x, y=sbp, name="SBP",
        line=dict(color=C["sbp"], width=2.5), mode="lines"))
    fig.add_trace(go.Scatter(
        x=x, y=dbp, name="DBP",
        line=dict(color=C["dbp"], width=2.5),
        fill="tonexty", fillcolor="rgba(239,68,68,0.07)",
        mode="lines"))
    fig.add_trace(go.Scatter(
        x=x, y=pp, name="Pulse Pressure",
        line=dict(color=C["pp"], width=1.5, dash="dot"),
        mode="lines"))

    # Current year dot
    if len(x) > 0:
        fig.add_trace(go.Scatter(
            x=[x[-1]], y=[sbp[-1]],
            mode="markers",
            marker=dict(color=C["sbp"], size=10,
                        line=dict(color="white", width=2)),
            showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[x[-1]], y=[dbp[-1]],
            mode="markers",
            marker=dict(color=C["dbp"], size=10,
                        line=dict(color="white", width=2)),
            showlegend=False, hoverinfo="skip"))

    # Stage thresholds
    fig.add_hline(y=140, line_dash="dash",
                  line_color="rgba(239,68,68,0.3)",
                  annotation_text="Stage 2",
                  annotation_font_color="#ef4444",
                  annotation_font_size=10)
    fig.add_hline(y=130, line_dash="dash",
                  line_color="rgba(249,115,22,0.3)",
                  annotation_text="Stage 1",
                  annotation_font_color="#f97316",
                  annotation_font_size=10)

    # Milestone flags
    sbp_list  = results["sbp"]
    egfr_list = results["egfr"]

    for yr_idx, s in enumerate(sbp_list):
        if yr_idx > 0 and sbp_list[yr_idx-1] < 140 and s >= 140:
            if years[yr_idx] <= years[yr]:
                fig.add_vline(x=years[yr_idx],
                              line_color="rgba(239,68,68,0.5)",
                              line_width=1.5, line_dash="dot")
                fig.add_annotation(
                    x=years[yr_idx], y=140,
                    text="Stage 2 reached",
                    showarrow=True, arrowhead=2,
                    arrowcolor="#ef4444",
                    font=dict(color="#ef4444", size=10),
                    bgcolor="#0a0a0f", bordercolor="#ef4444",
                    yshift=20)
            break

    for yr_idx, e in enumerate(egfr_list):
        if yr_idx > 0 and egfr_list[yr_idx-1] >= 60 and e < 60:
            if years[yr_idx] <= years[yr]:
                fig.add_annotation(
                    x=years[yr_idx], y=sbp_list[yr_idx],
                    text="CKD Stage 3",
                    showarrow=True, arrowhead=2,
                    arrowcolor="#f97316",
                    font=dict(color="#f97316", size=10),
                    bgcolor="#0a0a0f", bordercolor="#f97316",
                    yshift=-20)
            break

    fig.update_layout(**base_layout("Blood Pressure Trajectory (mmHg)"))
    fig.update_xaxes(range=[0, max(years)],
                     title_text="Years from baseline",
                     title_font_color=C["muted"])
    fig.update_yaxes(title_text="mmHg",
                     title_font_color=C["muted"])
    return fig
def build_egfr_chart(results):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results["year"], y=results["egfr"], name="eGFR",
        line=dict(color=C["egfr"], width=2.5),
        fill="tozeroy", fillcolor="rgba(34,211,238,0.07)",
        mode="lines"))

    for val, label, col in [
        (90, "Normal",      "rgba(74,222,128,0.4)"),
        (60, "CKD Stage 3", "rgba(249,115,22,0.4)"),
        (30, "CKD Stage 4", "rgba(239,68,68,0.4)"),
    ]:
        fig.add_hline(y=val, line_dash="dash", line_color=col,
                      annotation_text=label,
                      annotation_font_size=10,
                      annotation_font_color=col.replace("0.4","1"))

    fig.update_layout(**base_layout(
        "Kidney Function — eGFR (ml/min/1.73m²)"))
    fig.update_yaxes(title_text="eGFR",
                     title_font_color=C["muted"])
    fig.update_xaxes(title_text="Years from baseline",
                     title_font_color=C["muted"])
    return fig

def build_survival_chart(results):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results["year"], y=results["survival"],
        name="Survival probability",
        line=dict(color=C["survival"], width=2.5),
        fill="tozeroy", fillcolor="rgba(74,222,128,0.07)",
        mode="lines"))

    # Calculate dynamic y-axis range based on baseline survival
    baseline_surv = results["survival"][0]
    min_surv = min(results["survival"])
    max_surv = max(results["survival"])
    y_range = [min_surv - 5, max_surv + 5]

    fig.add_hline(y=50, line_dash="dash",
                  line_color="rgba(239,68,68,0.3)",
                  annotation_text="50% survival",
                  annotation_font_color="#ef4444",
                  annotation_font_size=10)

    fig.update_layout(**base_layout("Cumulative Survival Probability (%)"))
    fig.update_yaxes(range=y_range, title_text="%",
                     title_font_color=C["muted"])
    fig.update_xaxes(title_text="Years from baseline",
                     title_font_color=C["muted"])
    return fig

def build_intervention_comparison_chart(baseline_results, treated_results, intervention_year, intervention_name):
    """
    Build comparison chart overlaying baseline vs treated trajectories.
    
    Creates a comprehensive visualization showing:
    - Baseline trajectory (solid lines)
    - Treated trajectory (dashed lines) 
    - Vertical line marking intervention start
    - Delta summary cards with quantified differences
    
    Args:
        baseline_results: Results dict from Engine.simulate()
        treated_results: Results dict from Engine.simulate_with_intervention()
        intervention_year: Year when treatment began
        intervention_name: Name of intervention for display
        
    Returns:
        Plotly figure with comparison visualization
    """
    years = baseline_results["year"]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Blood Pressure (mmHg)", "Kidney Function (eGFR)", 
                       "Survival Probability (%)", "Annual Death Risk (%)"),
        vertical_spacing=0.08,
        horizontal_spacing=0.06
    )
    
    # Blood Pressure comparison
    fig.add_trace(go.Scatter(
        x=years, y=baseline_results["sbp"], 
        name="Baseline SBP", line=dict(color=C["sbp"], width=2.5),
        mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=years, y=treated_results["sbp"], 
        name="Treated SBP", line=dict(color=C["sbp"], width=2.5, dash="dash"),
        mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=years, y=baseline_results["dbp"], 
        name="Baseline DBP", line=dict(color=C["dbp"], width=2.5),
        mode="lines", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=years, y=treated_results["dbp"], 
        name="Treated DBP", line=dict(color=C["dbp"], width=2.5, dash="dash"),
        mode="lines", showlegend=False), row=1, col=1)
    
    # eGFR comparison
    fig.add_trace(go.Scatter(
        x=years, y=baseline_results["egfr"], 
        name="Baseline eGFR", line=dict(color=C["egfr"], width=2.5),
        mode="lines"), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=years, y=treated_results["egfr"], 
        name="Treated eGFR", line=dict(color=C["egfr"], width=2.5, dash="dash"),
        mode="lines"), row=1, col=2)
    
    # Survival comparison
    fig.add_trace(go.Scatter(
        x=years, y=baseline_results["survival"], 
        name="Baseline Survival", line=dict(color=C["survival"], width=2.5),
        mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=years, y=treated_results["survival"], 
        name="Treated Survival", line=dict(color=C["survival"], width=2.5, dash="dash"),
        mode="lines"), row=2, col=1)
    
    # Death risk comparison
    fig.add_trace(go.Scatter(
        x=years, y=baseline_results["death_prob_annual"], 
        name="Baseline Risk", line=dict(color="#ef4444", width=2.5),
        mode="lines"), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=years, y=treated_results["death_prob_annual"], 
        name="Treated Risk", line=dict(color="#ef4444", width=2.5, dash="dash"),
        mode="lines"), row=2, col=2)
    
    # Add intervention year vertical lines
    for row in [1, 2]:
        for col in [1, 2]:
            fig.add_vline(x=intervention_year, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                         annotation_text=f"{intervention_name} Started", 
                         annotation_font_color="white", annotation_font_size=9,
                         row=row, col=col)
    
    # Update layout
    fig.update_layout(
        title=dict(text=f"{intervention_name} vs Baseline Comparison", 
                   font=dict(color=C["text"], size=14, family="Georgia, serif")),
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        font=dict(color=C["text"], family="Georgia, serif"),
        height=600, showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0.5, 
                   font=dict(color=C["text"], size=10),
                   bgcolor="rgba(0,0,0,0)")
    )
    
    # Update axes styling
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(gridcolor=C["border"], zerolinecolor=C["border"],
                           tickfont=dict(color=C["muted"], size=9), row=i, col=j)
            fig.update_yaxes(gridcolor=C["border"], zerolinecolor=C["border"],
                           tickfont=dict(color=C["muted"], size=9), row=i, col=j)
    
    return fig

def calculate_intervention_deltas(baseline_results, treated_results, intervention_year):
    """
    Calculate quantified differences between baseline and treated trajectories.
    
    Args:
        baseline_results: Results dict from Engine.simulate()
        treated_results: Results dict from Engine.simulate_with_intervention()
        intervention_year: Year when treatment began
        
    Returns:
        Dictionary with delta metrics for summary display
    """
    deltas = {}
    
    # Find key timepoints for comparison
    years = baseline_results["year"]
    comparison_years = [10, 20, 30]  # Years to compare
    
    for year in comparison_years:
        if year < len(years):
            idx = year
            deltas[f"year_{year}"] = {
                "sbp_diff": treated_results["sbp"][idx] - baseline_results["sbp"][idx],
                "egfr_diff": treated_results["egfr"][idx] - baseline_results["egfr"][idx],
                "survival_diff": treated_results["survival"][idx] - baseline_results["survival"][idx],
                "death_risk_diff": treated_results["death_prob_annual"][idx] - baseline_results["death_prob_annual"][idx]
            }
    
    # Calculate overall benefit metrics
    if len(treated_results["survival"]) > 0 and len(baseline_results["survival"]) > 0:
        final_baseline_survival = baseline_results["survival"][-1]
        final_treated_survival = treated_results["survival"][-1]
        deltas["overall"] = {
            "survival_gain": final_treated_survival - final_baseline_survival,
            "years_gained": None,  # Could calculate life expectancy gain
            "egfr_preserved": None  # Could calculate cumulative eGFR preservation
        }
    
    return deltas

def fig_to_html(fig):
    return fig.to_html(
        include_plotlyjs="cdn", full_html=True,
        config={"displayModeBar": False, "responsive": True})

def placeholder_html(msg="Build the twin to see results"):
    return f"""<html><body style="background:#12121a;display:flex;
    align-items:center;justify-content:center;height:100vh;margin:0;">
    <p style="color:#334155;font-family:'Georgia',serif;font-size:14px;
    letter-spacing:1px;">{msg}</p></body></html>"""

# ── Style ─────────────────────────────────────────────────────────────────
STYLE = """
QMainWindow, QWidget {
    background-color: #0a0a0f;
    color: #e2e8f0;
    font-family: 'Georgia', serif;
}
QTabWidget::pane {
    border: none;
    border-top: 1px solid #1e1e2e;
    background: #0a0a0f;
    border-radius: 0 0 8px 8px;
}
QTabBar::tab {
    background: #12121a;
    color: #64748b;
    padding: 10px 28px;
    border: 1px solid transparent;
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    margin-right: 2px;
    font-size: 12px;
    font-family: 'Georgia', serif;
}
QTabBar::tab:selected {
    background: #0a0a0f;
    color: #e2e8f0;
    border: 1px solid #1e1e2e;
    border-bottom: 2px solid #0a0a0f;
}
QTabBar::tab:hover:!selected { color: #94a3b8; background: #16161f; }
QLineEdit, QComboBox {
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-radius: 6px;
    padding: 8px 12px;
    color: #e2e8f0;
    font-size: 13px;
    font-family: 'Georgia', serif;
}
QLineEdit:focus { border: 1px solid #6366f1; }
QComboBox::drop-down { border: none; padding-right: 8px; }
QComboBox QAbstractItemView {
    background: #12121a;
    border: 1px solid #1e1e2e;
    color: #e2e8f0;
    selection-background-color: #6366f1;
}
QPushButton#primary {
    background: #6366f1; color: #ffffff; border: none;
    border-radius: 8px; padding: 12px 32px;
    font-size: 13px; font-family: 'Georgia', serif;
}
QPushButton#primary:hover { background: #818cf8; }
QPushButton#primary:pressed { background: #4f46e5; }
QPushButton#primary:disabled { background: #1e1e2e; color: #334155; }
QPushButton#secondary {
    background: #1e1e2e; color: #94a3b8;
    border: 1px solid #334155; border-radius: 6px;
    padding: 6px 16px; font-size: 12px;
    font-family: 'Georgia', serif;
}
QPushButton#secondary:hover { background: #2d2d3f; color: #e2e8f0; }
QProgressBar {
    background: #1e1e2e; border: none;
    border-radius: 4px; height: 6px;
    color: transparent;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #6366f1, stop:1 #a855f7);
    border-radius: 4px;
}
QFrame#card {
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
}
QLabel#heading {
    font-size: 22px; color: #e2e8f0;
    font-family: 'Georgia', serif;
}
QLabel#subheading {
    font-size: 12px; color: #64748b;
    font-family: 'Georgia', serif;
}
QLabel#field_label {
    font-size: 11px; color: #94a3b8;
    font-family: 'Georgia', serif; letter-spacing: 0.5px;
}
QLabel#metric_value {
    font-size: 28px; color: #e2e8f0;
    font-family: 'Georgia', serif;
}
QLabel#metric_label {
    font-size: 10px; color: #64748b;
    font-family: 'Georgia', serif; letter-spacing: 1px;
}
QScrollArea { border: none; background: transparent; }
"""

def card(parent=None):
    f = QFrame(parent)
    f.setObjectName("card")
    return f

def lbl(text, obj="", parent=None):
    l = QLabel(text, parent)
    if obj:
        l.setObjectName(obj)
    return l

def inp_field(label_text, placeholder="", parent=None):
    w = QWidget(parent)
    v = QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(4)
    v.addWidget(lbl(label_text, "field_label"))
    i = QLineEdit()
    i.setPlaceholderText(placeholder)
    v.addWidget(i)
    return w, i

# ── Tab 1: Patient Input ──────────────────────────────────────────────────
class PatientTab(QWidget):
    simulate_requested = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 40, 40, 40)
        root.setSpacing(0)

        root.addWidget(lbl("Patient Twin Initialisation", "heading"))
        root.addSpacing(4)
        root.addWidget(lbl(
            "Enter measurements manually or load from the patient registry",
            "subheading"))
        root.addSpacing(24)

        # Load buttons
        btn_row = QWidget()
        bl      = QHBoxLayout(btn_row)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(12)

        load_json_btn = QPushButton("Load Patient JSON")
        load_json_btn.setObjectName("secondary")
        load_json_btn.clicked.connect(self._load_json)
        bl.addWidget(load_json_btn)
        bl.addStretch()
        root.addWidget(btn_row)
        root.addSpacing(16)

        # Form
        form_card = card()
        fl        = QGridLayout(form_card)
        fl.setContentsMargins(32, 32, 32, 32)
        fl.setSpacing(20)
        for c in range(4):
            fl.setColumnStretch(c, 1)

        fields = [
            ("Age",            "e.g. 55",   0, 0, "years"),
            ("Systolic BP",    "e.g. 135",  0, 1, "mmHg"),
            ("Diastolic BP",   "e.g. 85",   0, 2, "mmHg"),
            ("Waist",          "e.g. 95",   0, 3, "cm"),
            ("Heart Rate",     "e.g. 70",   1, 0, "bpm"),
            ("eGFR",           "e.g. 85",   1, 1, "ml/min"),
            ("Pack-Years",     "0 if never",1, 2, "pack-yrs"),
            ("Alcohol",        "e.g. 10",   1, 3, "g/day"),
            ("Sodium",         "e.g. 3500", 2, 0, "mg/day"),
        ]

        self.inputs = {}
        for label_t, ph, row, col, units in fields:
            w, i = inp_field(f"{label_t}  ·  {units}")
            i.setPlaceholderText(ph)
            fl.addWidget(w, row, col)
            self.inputs[label_t] = i

        # Sex
        sex_w = QWidget()
        sv    = QVBoxLayout(sex_w)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(4)
        sv.addWidget(lbl("Sex", "field_label"))
        self.sex_combo = QComboBox()
        self.sex_combo.addItems(["Male", "Female"])
        sv.addWidget(self.sex_combo)
        fl.addWidget(sex_w, 2, 1)

        # Years
        yr_w = QWidget()
        yl   = QVBoxLayout(yr_w)
        yl.setContentsMargins(0, 0, 0, 0)
        yl.setSpacing(4)
        yl.addWidget(lbl("Simulation Horizon  ·  years", "field_label"))
        self.years_combo = QComboBox()
        self.years_combo.addItems(["10", "20", "30"])
        self.years_combo.setCurrentIndex(2)
        yl.addWidget(self.years_combo)
        fl.addWidget(yr_w, 2, 2)

        root.addWidget(form_card)
        root.addSpacing(20)

        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setFixedHeight(6)
        root.addWidget(self.progress)
        root.addSpacing(8)

        self.status_lbl = lbl("", "subheading")
        self.status_lbl.setVisible(False)
        root.addWidget(self.status_lbl)
        root.addSpacing(16)

        # Simulate button
        self.sim_btn = QPushButton("  Build Digital Twin")
        self.sim_btn.setObjectName("primary")
        self.sim_btn.setFixedHeight(48)
        self.sim_btn.setFixedWidth(280)
        self.sim_btn.clicked.connect(self._on_simulate)
        root.addWidget(self.sim_btn)
        root.addSpacing(24)

        # Age range notice
        notice = card()
        nl     = QVBoxLayout(notice)
        nl.setContentsMargins(24, 16, 24, 16)
        nl.addWidget(lbl(
            "Validated range: Age 35–75. Predictions outside this "
            "range are extrapolations. See documentation.",
            "subheading"))
        root.addWidget(notice)
        root.addStretch()

    def load_patient(self, patient):
        """Populate form from a patient dict (from registry or JSON)."""
        sex_map = {"Male": 0, "Female": 1, 1: 0, 2: 1}
        self.sex_combo.setCurrentIndex(
            sex_map.get(patient.get("sex", "Male"), 0))
        mapping = {
            "Age":         str(patient.get("age", "")),
            "Systolic BP": str(patient.get("sbp", "")),
            "Diastolic BP":str(patient.get("dbp", "")),
            "Waist":       str(patient.get("waist_cm", "")),
            "Heart Rate":  str(patient.get("heart_rate", "")),
            "eGFR":        str(patient.get("egfr", "")),
            "Pack-Years":  str(patient.get("pack_years", "")),
            "Alcohol":     str(patient.get("alcohol_g_day", "")),
            "Sodium":      str(patient.get("sodium_mg_day", "")),
        }
        for key, val in mapping.items():
            if key in self.inputs:
                self.inputs[key].setText(val)

    def _load_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Patient", "patients/", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path) as f:
                p = json.load(f)
            self.load_patient(p)
            self.status_lbl.setText(
                f"Loaded: {p.get('name', path)}")
            self.status_lbl.setStyleSheet(
                "color:#4ade80;font-size:12px;")
            self.status_lbl.setVisible(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_simulate(self):
        try:
            params = {
                "age":           float(self.inputs["Age"].text()),
                "sex":           1 if self.sex_combo.currentText()=="Male" else 2,
                "sbp":           float(self.inputs["Systolic BP"].text()),
                "dbp":           float(self.inputs["Diastolic BP"].text()),
                "waist_cm":      float(self.inputs["Waist"].text()),
                "heart_rate":    float(self.inputs["Heart Rate"].text()),
                "egfr":          float(self.inputs["eGFR"].text()),
                "pack_years":    float(self.inputs["Pack-Years"].text()),
                "alcohol_g_day": float(self.inputs["Alcohol"].text()),
                "sodium_mg_day": float(self.inputs["Sodium"].text()),
                "years":         int(self.years_combo.currentText()),
            }
        except ValueError:
            self.status_lbl.setText(
                "Please fill in all fields with valid numbers.")
            self.status_lbl.setStyleSheet("color:#ef4444;font-size:12px;")
            self.status_lbl.setVisible(True)
            return

        self.sim_btn.setEnabled(False)
        self.sim_btn.setText("  Building twin…")
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.status_lbl.setText("Initialising vascular state…")
        self.status_lbl.setStyleSheet("color:#64748b;font-size:12px;")
        self.status_lbl.setVisible(True)
        self.simulate_requested.emit(params)

    def set_progress(self, v):
        self.progress.setValue(v)
        msgs = {
            20: "Computing baseline vascular state…",
            40: "Advancing MAP and PP annually…",
            60: "Applying eGFR decline subsystem…",
            80: "Computing Lewington mortality…",
            100: "Twin built successfully.",
        }
        for t, m in sorted(msgs.items()):
            if v >= t:
                self.status_lbl.setText(m)

    def set_done(self):
        self.sim_btn.setEnabled(True)
        self.sim_btn.setText("  Rebuild Twin")
        self.status_lbl.setStyleSheet("color:#4ade80;font-size:12px;")

# ── Tab 2: Trajectory ─────────────────────────────────────────────────────
class TrajectoryTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.results = None
        self._build()

    def _on_slider(self, value):
        self.year_lbl.setText(f"Year {value}")
        if self.results:
            self._update_chart_inplace(value)

    def _build(self):
        from PyQt6.QtWidgets import QSlider
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 32, 40, 32)
        root.setSpacing(0)

        root.addWidget(lbl("Vascular Trajectory", "heading"))
        root.addSpacing(4)
        root.addWidget(lbl(
            "Year-by-year blood pressure simulation",
            "subheading"))
        root.addSpacing(16)

        # Slider
        slider_row = QWidget()
        sl = QHBoxLayout(slider_row)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(12)
        sl.addWidget(lbl("Year", "field_label"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 30)
        self.slider.setValue(0)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #1e1e2e; height: 6px; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #6366f1; width: 18px; height: 18px;
                border-radius: 9px; margin: -6px 0;
                border: 2px solid #818cf8;
            }
            QSlider::handle:horizontal:hover {
                background: #818cf8;
            }
            QSlider::sub-page:horizontal {
                background: #6366f1; border-radius: 3px;
            }
        """)
        self.slider.valueChanged.connect(self._on_slider)
        sl.addWidget(self.slider, stretch=1)
        self.year_lbl = QLabel("Year 0")
        self.year_lbl.setStyleSheet(
            "color:#6366f1;font-size:13px;"
            "font-family:'Georgia',serif;min-width:60px;")
        sl.addWidget(self.year_lbl)
        root.addWidget(slider_row)
        self.slider.raise_()
        root.addSpacing(20)

        # Metric cards
        metrics_row = QWidget()
        ml = QHBoxLayout(metrics_row)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(12)

        self.metrics = {}
        for name, color in [
            ("SBP", "#ef4444"), ("DBP", "#f97316"),
            ("PP",  "#a855f7"), ("eGFR","#22d3ee"),
            ("Survival","#4ade80")
        ]:
            c  = card()
            cl = QVBoxLayout(c)
            cl.setContentsMargins(16, 14, 16, 14)
            cl.setSpacing(4)
            vl = QLabel("—")
            vl.setStyleSheet(
                f"font-size:26px;color:{color};"
                f"font-family:'Georgia',serif;")
            chg = QLabel("")
            chg.setObjectName("metric_label")
            nl2 = lbl(name, "metric_label")
            cl.addWidget(vl)
            cl.addWidget(chg)
            cl.addWidget(nl2)
            ml.addWidget(c)
            self.metrics[name] = (vl, chg)

        root.addWidget(metrics_row)
        root.addSpacing(16)

        # Save report button row — above chart so it stays visible
        save_row = QWidget()
        srl = QHBoxLayout(save_row)
        srl.setContentsMargins(0, 0, 0, 8)
        srl.setSpacing(8)
        srl.addStretch()
        self.save_btn = QPushButton("↓  Save Report")
        self.save_btn.setObjectName("secondary")
        self.save_btn.setFixedHeight(30)
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._on_save_report)
        srl.addWidget(self.save_btn)
        root.addWidget(save_row)

        # Chart — load once, update in place
        self.chart = QWebEngineView()
        self.chart.setMinimumHeight(400)
        root.addWidget(self.chart, stretch=1)
        
        # Hoverable summary block
        self.summary_block = QPushButton("Hover for summary")
        self.summary_block.setStyleSheet("""
            QPushButton {
                background: #1e1e2e;
                color: #94a3b8;
                padding: 12px 16px;
                border-radius: 6px;
                font-size: 12px;
                font-family: 'Georgia', serif;
                border: none;
            }
            QPushButton:hover {
                background: #2a2a3e;
            }
        """)
        self.summary_block.setCursor(Qt.CursorShape.PointingHandCursor)
        self.summary_block.setToolTip("Click to ask ARIA about this trajectory")
        self.summary_block.clicked.connect(self._ask_aria_about_trajectory)
        root.addWidget(self.summary_block)
        root.addSpacing(12)


    def update(self, results):
        self.results = results
        self.slider.setRange(0, len(results["year"]) - 1)
        self.slider.setValue(0)
        self.year_lbl.setText("Year 0")
        self.save_btn.setEnabled(True)
        # Load full chart HTML once with animation
        self.chart.setHtml(self._build_animated_html(results))
        # After animation finishes update metrics for year 0
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(3400, lambda: self._update_metrics(0))
        
        # Update summary block
        r = results
        n = len(r["year"]) - 1
        sbp_rise = r["sbp"][-1] - r["sbp"][0]
        egfr_drop = r["egfr"][0] - r["egfr"][-1]
        self.summary_block.setText(
            f"Over {n} years: SBP {sbp_rise:+.1f} mmHg, eGFR -{egfr_drop:.1f} ml/min, "
            f"Final survival {r['survival'][-1]:.1f}%")
    
    def _ask_aria_about_trajectory(self):
        """Open ARIA panel and ask about the current trajectory"""
        main_window = self.window()
        if hasattr(main_window, 'aria_panel'):
            if not main_window.aria_panel.isVisible():
                main_window._toggle_aria()
            main_window.aria_panel._send_message(
                "Explain this patient's blood pressure trajectory. "
                "What are the key trends, risk factors, and clinical implications?"
            )

    #Saving the report
    def _on_save_report(self):
        self._save_report_signal()

    def set_save_callback(self, callback):
        """Called by MainWindow to wire up the save action."""
        self._save_report_signal = callback


    def _build_animated_html(self, results):
        """Full HTML load with draw-on animation. Called once."""
        years = results["year"]
        sbp   = results["sbp"]
        dbp   = results["dbp"]
        pp    = results["pp"]
        n     = len(years)

        frames = []
        for i in range(1, n + 1):
            frames.append(go.Frame(
                data=[
                    go.Scatter(
                        x=years[:i], y=sbp[:i],
                        line=dict(color=C["sbp"], width=2.5),
                        mode="lines", name="SBP"),
                    go.Scatter(
                        x=years[:i], y=dbp[:i],
                        line=dict(color=C["dbp"], width=2.5),
                        fill="tonexty",
                        fillcolor="rgba(239,68,68,0.07)",
                        mode="lines", name="DBP"),
                    go.Scatter(
                        x=years[:i], y=pp[:i],
                        line=dict(color=C["pp"], width=1.5, dash="dot"),
                        mode="lines", name="Pulse Pressure",
                        opacity=min(1.0, i / max(1, n * 0.4))),
                    go.Scatter(
                        x=[years[i-1]], y=[sbp[i-1]],
                        mode="markers",
                        marker=dict(color=C["sbp"], size=10,
                                    line=dict(color="white", width=2)),
                        showlegend=False),
                    go.Scatter(
                        x=years, y=sbp,
                        line=dict(color="rgba(239,68,68,0.1)",
                                  width=1, dash="dot"),
                        mode="lines", showlegend=False,
                        hoverinfo="skip"),
                ],
                name=str(i)
            ))

        fig = go.Figure(
            data=[
                go.Scatter(x=[], y=[],
                           line=dict(color=C["sbp"], width=2.5),
                           mode="lines", name="SBP"),
                go.Scatter(x=[], y=[],
                           line=dict(color=C["dbp"], width=2.5),
                           fill="tonexty",
                           fillcolor="rgba(239,68,68,0.07)",
                           mode="lines", name="DBP"),
                go.Scatter(x=[], y=[],
                           line=dict(color=C["pp"], width=1.5,
                                     dash="dot"),
                           mode="lines", name="Pulse Pressure"),
                go.Scatter(x=[], y=[], mode="markers",
                           marker=dict(color=C["sbp"], size=10,
                                       line=dict(color="white",
                                                 width=2)),
                           showlegend=False),
                go.Scatter(x=[], y=[],
                           line=dict(color="rgba(239,68,68,0.1)",
                                     width=1, dash="dot"),
                           mode="lines", showlegend=False,
                           hoverinfo="skip"),
            ],
            frames=frames
        )

        fig.add_hline(y=140, line_dash="dash",
                      line_color="rgba(239,68,68,0.3)",
                      annotation_text="Stage 2",
                      annotation_font_color="#ef4444",
                      annotation_font_size=10)
        fig.add_hline(y=130, line_dash="dash",
                      line_color="rgba(249,115,22,0.3)",
                      annotation_text="Stage 1",
                      annotation_font_color="#f97316",
                      annotation_font_size=10)

        layout = base_layout("Blood Pressure Trajectory (mmHg)")
        layout.update(
            xaxis=dict(range=[0, max(years)],
                       gridcolor=C["border"],
                       zerolinecolor=C["border"],
                       tickfont=dict(color=C["muted"], size=10),
                       title_text="Years from baseline",
                       title_font_color=C["muted"]),
            yaxis=dict(gridcolor=C["border"],
                       zerolinecolor=C["border"],
                       tickfont=dict(color=C["muted"], size=10),
                       title_text="mmHg",
                       title_font_color=C["muted"]),
            updatemenus=[dict(
                type="buttons", showactive=False, visible=False,
                buttons=[dict(
                    label="Play", method="animate",
                    args=[None, dict(
                        frame=dict(duration=55, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=0),
                        mode="immediate"
                    )]
                )]
            )]
        )
        fig.update_layout(**layout)

        html = fig.to_html(
            include_plotlyjs="cdn", full_html=True,
            config={"displayModeBar": False, "responsive": True})

        # Store results as JS variable for in-place slider updates
        import json as _json
        data_js = _json.dumps({
            "years": years,
            "sbp":   sbp,
            "dbp":   dbp,
            "pp":    pp,
        })

        autoplay_js = f"""
<script>
var _fsdata = {data_js};

document.addEventListener('DOMContentLoaded', function() {{
    setTimeout(function() {{
        var gd = document.querySelector('.plotly-graph-div');
        if (gd) {{
            Plotly.animate(gd, null, {{
                frame: {{duration: 55, redraw: true}},
                transition: {{duration: 0}},
                mode: 'immediate'
            }});
        }}
    }}, 300);
}});

// Called by Python via runJavaScript to update chart in place
function updateYear(yr) {{
    var gd = document.querySelector('.plotly-graph-div');
    if (!gd) return;
    var years = _fsdata.years;
    var sbp   = _fsdata.sbp;
    var dbp   = _fsdata.dbp;
    var pp    = _fsdata.pp;
    var x     = years.slice(0, yr + 1);
    Plotly.react(gd, [
        {{
            x: x, y: sbp.slice(0, yr+1),
            type: 'scatter', mode: 'lines', name: 'SBP',
            line: {{color: '#ef4444', width: 2.5}}
        }},
        {{
            x: x, y: dbp.slice(0, yr+1),
            type: 'scatter', mode: 'lines', name: 'DBP',
            fill: 'tonexty', fillcolor: 'rgba(239,68,68,0.07)',
            line: {{color: '#f97316', width: 2.5}}
        }},
        {{
            x: x, y: pp.slice(0, yr+1),
            type: 'scatter', mode: 'lines', name: 'Pulse Pressure',
            line: {{color: '#a855f7', width: 1.5, dash: 'dot'}}
        }},
        {{
            x: [years[yr]], y: [sbp[yr]],
            type: 'scatter', mode: 'markers', showlegend: false,
            marker: {{color: '#ef4444', size: 10,
                      line: {{color: 'white', width: 2}}}}
        }},
        {{
            x: years, y: sbp,
            type: 'scatter', mode: 'lines',
            showlegend: false, hoverinfo: 'skip',
            line: {{color: 'rgba(239,68,68,0.1)', width: 1, dash: 'dot'}}
        }},
    ], gd.layout);
}}
</script>
"""
        html = html.replace("</body>", autoplay_js + "</body>")
        return html

    def _update_chart_inplace(self, yr):
        """Update chart without reloading HTML — no white flash."""
        self.chart.page().runJavaScript(
            f"if(typeof updateYear === 'function') updateYear({yr});")
        self._update_metrics(yr)

    def _update_metrics(self, yr):
        r    = self.results
        keys = {"SBP":"sbp","DBP":"dbp","PP":"pp",
                "eGFR":"egfr","Survival":"survival"}
        for name, (vl, chg) in self.metrics.items():
            key = keys[name]
            val = r[key][min(yr, len(r[key])-1)]
            vl.setText(f"{val:.0f}")
            diff = val - r[key][0]
            if diff != 0:
                col  = "#ef4444" if diff > 0 else "#4ade80"
                sign = "+" if diff > 0 else ""
                chg.setText(
                    f'<span style="color:{col};font-size:11px;">'
                    f'{sign}{diff:.1f}</span>')
            else:
                chg.setText("")

    def _refresh(self, yr=0):
        self._update_chart_inplace(yr)

# ── Tab 3: Metrics ────────────────────────────────────────────────────────
class MetricsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 32, 40, 32)
        root.setSpacing(0)

        root.addWidget(lbl("Clinical Metrics", "heading"))
        root.addSpacing(4)
        root.addWidget(lbl(
            "Kidney function decline and survival probability",
            "subheading"))
        root.addSpacing(24)

        charts = QWidget()
        cl     = QHBoxLayout(charts)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(16)

        from PyQt6.QtGui import QColor
        self.egfr_view     = QWebEngineView()
        self.survival_view = QWebEngineView()
        self.egfr_view.setMinimumHeight(340)
        self.survival_view.setMinimumHeight(340)
        self.egfr_view.page().setBackgroundColor(QColor("#12121a"))
        self.survival_view.page().setBackgroundColor(QColor("#12121a"))
        self.egfr_view.setHtml(placeholder_html())
        self.survival_view.setHtml(placeholder_html())
        cl.addWidget(self.egfr_view)
        cl.addWidget(self.survival_view)
        root.addWidget(charts, stretch=1)

        root.addSpacing(20)
        self.summary = card()
        sl = QVBoxLayout(self.summary)
        sl.setContentsMargins(28, 20, 28, 20)
        sl.addWidget(lbl("Summary", "field_label"))
        self.summary_lbl = QLabel("")
        self.summary_lbl.setStyleSheet(
            "color:#94a3b8;font-size:12px;"
            "font-family:'Georgia',serif;")
        self.summary_lbl.setWordWrap(True)
        sl.addWidget(self.summary_lbl)
        root.addWidget(self.summary)

    def update(self, results):
        self.egfr_view.setHtml(
            fig_to_html(build_egfr_chart(results)))
        self.survival_view.setHtml(
            fig_to_html(build_survival_chart(results)))

        r = results
        n = len(r["year"]) - 1
        sbp_rise = r["sbp"][-1] - r["sbp"][0]
        egfr_drop = r["egfr"][0] - r["egfr"][-1]
        self.summary_lbl.setText(
            f"Over {n} simulated years:  "
            f"SBP {'+' if sbp_rise>=0 else ''}{sbp_rise:.1f} mmHg  ·  "
            f"eGFR -{egfr_drop:.1f} ml/min  ·  "
            f"Final survival {r['survival'][-1]:.1f}%"
        )

# ── Tab 4: Interventions (Phase 3 - ACE Inhibitor) ──────────────────────────
class InterventionsTab(QWidget):
    intervention_run = pyqtSignal(dict, str, dict, dict)  # params, intervention_name, baseline_results, treated_results
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.baseline_results = None
        self.treated_results = None
        self.current_intervention = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 40, 40, 40)
        root.setSpacing(0)

        root.addWidget(lbl("Medical Interventions", "heading"))
        root.addSpacing(4)
        root.addWidget(lbl(
            "Compare baseline trajectory against evidence-based clinical interventions",
            "subheading"))
        root.addSpacing(30)

        # Intervention selection card
        selection_card = card()
        sl = QVBoxLayout(selection_card)
        sl.setContentsMargins(32, 24, 32, 24)
        sl.setSpacing(16)

        # Intervention selector with checkboxes
        self.intervention_checkboxes = {}
        interventions = list_available_interventions()
        
        for intervention_name in interventions:
            checkbox_row = QWidget()
            cb_layout = QHBoxLayout(checkbox_row)
            cb_layout.setContentsMargins(0, 0, 0, 0)
            cb_layout.setSpacing(12)
            
            checkbox = QCheckBox(intervention_name)
            checkbox.stateChanged.connect(self._on_intervention_toggled)
            self.intervention_checkboxes[intervention_name] = checkbox
            
            cb_layout.addWidget(checkbox)
            cb_layout.addStretch()
            
            sl.addWidget(checkbox_row)

        # Start year selector
        year_row = QWidget()
        yl = QHBoxLayout(year_row)
        yl.setContentsMargins(0, 0, 0, 0)
        yl.setSpacing(12)
        yl.addWidget(lbl("Start Year:", "field_label"))
        
        self.year_spin = QSpinBox()
        self.year_spin.setRange(1, 29)
        self.year_spin.setValue(5)
        yl.addWidget(self.year_spin)
        yl.addStretch()
        
        sl.addWidget(year_row)

        # Run button
        self.run_btn = QPushButton("Run Comparison")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self._run_intervention)
        sl.addWidget(self.run_btn)
        
        root.addWidget(selection_card)
        root.addSpacing(20)
        
        # Info message
        self.info_lbl = QLabel("After running an intervention, results will appear in the Intervention Results tab")
        self.info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.info_lbl)
        
        # Loading animation overlay (hidden by default)
        self.loading_overlay = QWidget()
        self.loading_overlay.setStyleSheet("background:#0a0a0f;")
        self.loading_overlay.setVisible(False)
        lo = QVBoxLayout(self.loading_overlay)
        lo.setContentsMargins(0, 0, 0, 0)
        
        loading_center = QWidget()
        lc = QVBoxLayout(loading_center)
        lc.setContentsMargins(0, 0, 0, 0)
        lc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.loading_dots = QLabel("Running simulation")
        self.loading_dots.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_dots.setStyleSheet("color:#6366f1;font-size:16px;font-family:'Georgia',serif;")
        lc.addWidget(self.loading_dots)
        
        lo.addWidget(loading_center)
        root.addWidget(self.loading_overlay, stretch=1)
        root.addStretch()

    def _on_intervention_toggled(self, state):
        """Handle intervention checkbox toggle"""
        # Ensure only one intervention is selected at a time
        for intervention_name, checkbox in self.intervention_checkboxes.items():
            if checkbox != self.sender():
                checkbox.setChecked(False)
        
        # Get the checked intervention
        checked_intervention = None
        for intervention_name, checkbox in self.intervention_checkboxes.items():
            if checkbox.isChecked():
                checked_intervention = intervention_name
                break
        
        # Load or clear the intervention
        if checked_intervention:
            try:
                self.current_intervention = get_intervention(checked_intervention)
            except Exception as e:
                self.current_intervention = None
        else:
            self.current_intervention = None
            
    def _run_intervention(self):
        """Run intervention comparison simulation"""
        if not self.current_intervention:
            QMessageBox.warning(self, "Warning", "Please select an intervention first.")
            return
            
        main_window = self.window()
        if not hasattr(main_window, '_last_results') or not main_window._last_results:
            QMessageBox.warning(self, "Warning", "Please run baseline simulation first.")
            return
            
        self.current_intervention.start_year = self.year_spin.value()
        params = main_window._last_params
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Running...")
        
        # Show loading animation
        self.info_lbl.setVisible(False)
        self.loading_overlay.setVisible(True)
        self._loading_dots_count = 0
        self._loading_timer = QTimer()
        self._loading_timer.timeout.connect(self._update_loading_dots)
        self._loading_timer.start(500)
        
        # Process events to ensure UI updates before blocking
        QApplication.processEvents()
        
        try:
            baseline_results, treated_results = main_window.engine.simulate_with_intervention(
                **params, 
                intervention=self.current_intervention
            )
            
            # Stop loading animation
            self._loading_timer.stop()
            self.loading_overlay.setVisible(False)
            self.info_lbl.setVisible(True)
            
            # Trigger signal to load results in InterventionResultsTab
            self.intervention_run.emit(params, self.current_intervention.name, baseline_results, treated_results)
            
        except Exception as e:
            self._loading_timer.stop()
            self.loading_overlay.setVisible(False)
            self.info_lbl.setVisible(True)
            QMessageBox.critical(self, "Error", f"Simulation failed: {str(e)}")
        finally:
            self.run_btn.setEnabled(True)
            self.run_btn.setText("Run Comparison")
    
    def _update_loading_dots(self):
        """Animate loading dots"""
        self._loading_dots_count = (self._loading_dots_count + 1) % 4
        dots = "  ·" * self._loading_dots_count + "  ○" * (3 - self._loading_dots_count)
        self.loading_dots.setText(f"Running simulation{dots}")

# ── Tab 5: Intervention Results ─────────────────────────────────────────────
class InterventionResultsTab(QWidget):
    """Main tab for displaying intervention comparison graphs"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.baseline_results = None
        self.treated_results = None
        self.current_intervention = None
        self._build()
    
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 32, 40, 32)
        root.setSpacing(0)
        
        root.addWidget(lbl("Intervention Results", "heading"))
        root.addSpacing(4)
        root.addWidget(lbl("Baseline vs Treated trajectory comparison", "subheading"))
        root.addSpacing(20)
        
        self.placeholder_lbl = QLabel("Run an intervention from the Interventions tab to see results here")
        self.placeholder_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.result_tabs = QTabWidget()
        self.result_tabs.setDocumentMode(True)
        self.result_tabs.setVisible(False)
        
        # BP sub-tab
        bp_tab = QWidget()
        bp_layout = QVBoxLayout(bp_tab)
        bp_layout.setContentsMargins(0, 16, 0, 0)
        bp_layout.setSpacing(12)
        
        bp_slider_row = QWidget()
        bp_sl = QHBoxLayout(bp_slider_row)
        bp_sl.setContentsMargins(0, 0, 0, 0)
        bp_sl.setSpacing(12)
        bp_sl.addWidget(lbl("Year", "field_label"))
        self.bp_slider = QSlider(Qt.Orientation.Horizontal)
        self.bp_slider.setRange(0, 30)
        self.bp_slider.setValue(0)
        self.bp_slider.setStyleSheet("""
            QSlider::groove:horizontal { background: #1e1e2e; height: 4px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #6366f1; width: 16px; height: 16px; border-radius: 8px; margin: -6px 0; }
            QSlider::sub-page:horizontal { background: #6366f1; border-radius: 2px; }
        """)
        self.bp_slider.valueChanged.connect(lambda v: self._update_bp_chart(v))
        bp_sl.addWidget(self.bp_slider, stretch=1)
        self.bp_year_lbl = QLabel("Year 0")
        self.bp_year_lbl.setStyleSheet("color:#6366f1;font-size:13px;font-family:'Georgia',serif;min-width:60px;")
        bp_sl.addWidget(self.bp_year_lbl)
        bp_layout.addWidget(bp_slider_row)
        
        bp_metrics_row = QWidget()
        bp_ml = QHBoxLayout(bp_metrics_row)
        bp_ml.setContentsMargins(0, 0, 0, 0)
        bp_ml.setSpacing(12)
        self.bp_metrics = {}
        for name, color in [("SBP", "#ef4444"), ("DBP", "#f97316"), ("PP", "#a855f7")]:
            c = card()
            cl = QVBoxLayout(c)
            cl.setContentsMargins(16, 14, 16, 14)
            cl.setSpacing(4)
            vl = QLabel("—")
            vl.setStyleSheet(f"font-size:26px;color:{color};font-family:'Georgia',serif;")
            chg = QLabel("")
            chg.setObjectName("metric_label")
            nl2 = lbl(name, "metric_label")
            cl.addWidget(vl); cl.addWidget(chg); cl.addWidget(nl2)
            bp_ml.addWidget(c)
            self.bp_metrics[name] = (vl, chg)
        bp_layout.addWidget(bp_metrics_row)
        
        self.bp_chart = QWebEngineView()
        self.bp_chart.setMinimumHeight(400)
        bp_layout.addWidget(self.bp_chart, stretch=1)
        
        # BP summary block
        self.bp_summary = QPushButton("Hover for summary")
        self.bp_summary.setStyleSheet("""
            QPushButton {
                background: #1e1e2e;
                color: #94a3b8;
                padding: 10px 14px;
                border-radius: 6px;
                font-size: 11px;
                font-family: 'Georgia', serif;
                border: none;
            }
            QPushButton:hover {
                background: #2a2a3e;
            }
        """)
        self.bp_summary.setCursor(Qt.CursorShape.PointingHandCursor)
        self.bp_summary.setToolTip("Click to ask ARIA about blood pressure changes")
        self.bp_summary.clicked.connect(self._ask_aria_about_bp)
        bp_layout.addWidget(self.bp_summary)
        
        self.result_tabs.addTab(bp_tab, "  Blood Pressure  ")
        
        # eGFR sub-tab
        egfr_tab = QWidget()
        egfr_layout = QVBoxLayout(egfr_tab)
        egfr_layout.setContentsMargins(0, 16, 0, 0)
        egfr_layout.setSpacing(12)
        
        egfr_slider_row = QWidget()
        egfr_sl = QHBoxLayout(egfr_slider_row)
        egfr_sl.setContentsMargins(0, 0, 0, 0)
        egfr_sl.setSpacing(12)
        egfr_sl.addWidget(lbl("Year", "field_label"))
        self.egfr_slider = QSlider(Qt.Orientation.Horizontal)
        self.egfr_slider.setRange(0, 30)
        self.egfr_slider.setValue(0)
        self.egfr_slider.setStyleSheet("""
            QSlider::groove:horizontal { background: #1e1e2e; height: 4px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #6366f1; width: 16px; height: 16px; border-radius: 8px; margin: -6px 0; }
            QSlider::sub-page:horizontal { background: #6366f1; border-radius: 2px; }
        """)
        self.egfr_slider.valueChanged.connect(lambda v: self._update_egfr_chart(v))
        egfr_sl.addWidget(self.egfr_slider, stretch=1)
        self.egfr_year_lbl = QLabel("Year 0")
        self.egfr_year_lbl.setStyleSheet("color:#6366f1;font-size:13px;font-family:'Georgia',serif;min-width:60px;")
        egfr_sl.addWidget(self.egfr_year_lbl)
        egfr_layout.addWidget(egfr_slider_row)
        
        egfr_metrics_row = QWidget()
        egfr_ml = QHBoxLayout(egfr_metrics_row)
        egfr_ml.setContentsMargins(0, 0, 0, 0)
        egfr_ml.setSpacing(12)
        self.egfr_metrics = {}
        for name, color in [("eGFR", "#22d3ee")]:
            c = card()
            cl = QVBoxLayout(c)
            cl.setContentsMargins(16, 14, 16, 14)
            cl.setSpacing(4)
            vl = QLabel("—")
            vl.setStyleSheet(f"font-size:26px;color:{color};font-family:'Georgia',serif;")
            chg = QLabel("")
            chg.setObjectName("metric_label")
            nl2 = lbl(name, "metric_label")
            cl.addWidget(vl); cl.addWidget(chg); cl.addWidget(nl2)
            egfr_ml.addWidget(c)
            self.egfr_metrics[name] = (vl, chg)
        egfr_layout.addWidget(egfr_metrics_row)
        
        self.egfr_chart = QWebEngineView()
        self.egfr_chart.setMinimumHeight(400)
        egfr_layout.addWidget(self.egfr_chart, stretch=1)
        
        # eGFR summary block
        self.egfr_summary = QPushButton("Hover for summary")
        self.egfr_summary.setStyleSheet("""
            QPushButton {
                background: #1e1e2e;
                color: #94a3b8;
                padding: 10px 14px;
                border-radius: 6px;
                font-size: 11px;
                font-family: 'Georgia', serif;
                border: none;
            }
            QPushButton:hover {
                background: #2a2a3e;
            }
        """)
        self.egfr_summary.setCursor(Qt.CursorShape.PointingHandCursor)
        self.egfr_summary.setToolTip("Click to ask ARIA about kidney function changes")
        self.egfr_summary.clicked.connect(self._ask_aria_about_egfr)
        egfr_layout.addWidget(self.egfr_summary)
        
        self.result_tabs.addTab(egfr_tab, "  Kidney Function  ")
        
        # Survival sub-tab
        surv_tab = QWidget()
        surv_layout = QVBoxLayout(surv_tab)
        surv_layout.setContentsMargins(0, 16, 0, 0)
        surv_layout.setSpacing(12)
        
        surv_slider_row = QWidget()
        surv_sl = QHBoxLayout(surv_slider_row)
        surv_sl.setContentsMargins(0, 0, 0, 0)
        surv_sl.setSpacing(12)
        surv_sl.addWidget(lbl("Year", "field_label"))
        self.surv_slider = QSlider(Qt.Orientation.Horizontal)
        self.surv_slider.setRange(0, 30)
        self.surv_slider.setValue(0)
        self.surv_slider.setStyleSheet("""
            QSlider::groove:horizontal { background: #1e1e2e; height: 4px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #6366f1; width: 16px; height: 16px; border-radius: 8px; margin: -6px 0; }
            QSlider::sub-page:horizontal { background: #6366f1; border-radius: 2px; }
        """)
        self.surv_slider.valueChanged.connect(lambda v: self._update_surv_chart(v))
        surv_sl.addWidget(self.surv_slider, stretch=1)
        self.surv_year_lbl = QLabel("Year 0")
        self.surv_year_lbl.setStyleSheet("color:#6366f1;font-size:13px;font-family:'Georgia',serif;min-width:60px;")
        surv_sl.addWidget(self.surv_year_lbl)
        surv_layout.addWidget(surv_slider_row)
        
        surv_metrics_row = QWidget()
        surv_ml = QHBoxLayout(surv_metrics_row)
        surv_ml.setContentsMargins(0, 0, 0, 0)
        surv_ml.setSpacing(12)
        self.surv_metrics = {}
        for name, color in [("Survival", "#4ade80"), ("Annual Risk", "#ef4444")]:
            c = card()
            cl = QVBoxLayout(c)
            cl.setContentsMargins(16, 14, 16, 14)
            cl.setSpacing(4)
            vl = QLabel("—")
            vl.setStyleSheet(f"font-size:26px;color:{color};font-family:'Georgia',serif;")
            chg = QLabel("")
            chg.setObjectName("metric_label")
            nl2 = lbl(name, "metric_label")
            cl.addWidget(vl); cl.addWidget(chg); cl.addWidget(nl2)
            surv_ml.addWidget(c)
            self.surv_metrics[name] = (vl, chg)
        surv_layout.addWidget(surv_metrics_row)
        
        self.surv_chart = QWebEngineView()
        self.surv_chart.setMinimumHeight(400)
        surv_layout.addWidget(self.surv_chart, stretch=1)
        
        # Survival summary block
        self.surv_summary = QPushButton("Hover for summary")
        self.surv_summary.setStyleSheet("""
            QPushButton {
                background: #1e1e2e;
                color: #94a3b8;
                padding: 10px 14px;
                border-radius: 6px;
                font-size: 11px;
                font-family: 'Georgia', serif;
                border: none;
            }
            QPushButton:hover {
                background: #2a2a3e;
            }
        """)
        self.surv_summary.setCursor(Qt.CursorShape.PointingHandCursor)
        self.surv_summary.setToolTip("Click to ask ARIA about survival changes")
        self.surv_summary.clicked.connect(self._ask_aria_about_survival)
        surv_layout.addWidget(self.surv_summary)
        
        self.result_tabs.addTab(surv_tab, "  Survival  ")
        
        root.addWidget(self.placeholder_lbl)
        root.addWidget(self.result_tabs)
        root.addStretch()
    
    def load_results(self, intervention_name, baseline_results, treated_results):
        self.current_intervention = get_intervention(intervention_name)
        self.baseline_results = baseline_results
        self.treated_results = treated_results
        self.placeholder_lbl.setVisible(False)
        self.result_tabs.setVisible(True)
        self._load_bp_chart()
        self._load_egfr_chart()
        self._load_surv_chart()
        n = len(baseline_results["year"]) - 1
        for s in [self.bp_slider, self.egfr_slider, self.surv_slider]:
            s.setRange(0, n)
            s.setValue(0)
        
        # Update summary blocks
        b, t = baseline_results, treated_results
        sbp_diff = t["sbp"][-1] - b["sbp"][-1]
        self.bp_summary.setText(f"{self.current_intervention.name}: SBP {sbp_diff:+.1f} mmHg vs baseline")
        egfr_diff = t["egfr"][-1] - b["egfr"][-1]
        self.egfr_summary.setText(f"{self.current_intervention.name}: eGFR {egfr_diff:+.1f} ml/min vs baseline")
        surv_diff = t["survival"][-1] - b["survival"][-1]
        self.surv_summary.setText(f"{self.current_intervention.name}: Survival {surv_diff:+.1f}pp vs baseline")
    
    def _ask_aria_about_bp(self):
        """Open ARIA panel and ask about blood pressure changes"""
        main_window = self.window()
        if hasattr(main_window, 'aria_panel'):
            if not main_window.aria_panel.isVisible():
                main_window._toggle_aria()
            main_window.aria_panel._send_message(
                f"Explain the blood pressure changes from the {self.current_intervention.name} intervention. "
                "How does it affect SBP and DBP compared to baseline, and what are the clinical implications?"
            )
    
    def _ask_aria_about_egfr(self):
        """Open ARIA panel and ask about kidney function changes"""
        main_window = self.window()
        if hasattr(main_window, 'aria_panel'):
            if not main_window.aria_panel.isVisible():
                main_window._toggle_aria()
            main_window.aria_panel._send_message(
                f"Explain the kidney function changes from the {self.current_intervention.name} intervention. "
                "How does it affect eGFR compared to baseline, and what are the clinical implications?"
            )
    
    def _ask_aria_about_survival(self):
        """Open ARIA panel and ask about survival changes"""
        main_window = self.window()
        if hasattr(main_window, 'aria_panel'):
            if not main_window.aria_panel.isVisible():
                main_window._toggle_aria()
            main_window.aria_panel._send_message(
                f"Explain the survival changes from the {self.current_intervention.name} intervention. "
                "How does it affect survival probability compared to baseline, and what are the clinical implications?"
            )
    
    def _update_bp_chart(self, yr):
        self.bp_year_lbl.setText(f"Year {yr}")
        if self.baseline_results:
            self.bp_chart.page().runJavaScript(f"if(typeof updateBPYear==='function') updateBPYear({yr});")
            self._update_bp_metrics(yr)
    
    def _update_egfr_chart(self, yr):
        self.egfr_year_lbl.setText(f"Year {yr}")
        if self.baseline_results:
            self.egfr_chart.page().runJavaScript(f"if(typeof updateEgfrYear==='function') updateEgfrYear({yr});")
            self._update_egfr_metrics(yr)
    
    def _update_surv_chart(self, yr):
        self.surv_year_lbl.setText(f"Year {yr}")
        if self.baseline_results:
            self.surv_chart.page().runJavaScript(f"if(typeof updateSurvYear==='function') updateSurvYear({yr});")
            self._update_surv_metrics(yr)
    
    def _update_bp_metrics(self, yr):
        b, t = self.baseline_results, self.treated_results
        for name, (vl, chg) in self.bp_metrics.items():
            key = {"SBP": "sbp", "DBP": "dbp", "PP": "pp"}[name]
            bval = b[key][min(yr, len(b[key])-1)]
            tval = t[key][min(yr, len(t[key])-1)]
            vl.setText(f"{tval:.0f}")
            diff = tval - bval
            if diff != 0:
                col = "#ef4444" if diff > 0 else "#4ade80"
                sign = "+" if diff > 0 else ""
                chg.setText(f'<span style="color:{col};font-size:11px;">{sign}{diff:.1f} vs baseline</span>')
            else:
                chg.setText("")
    
    def _update_egfr_metrics(self, yr):
        b, t = self.baseline_results, self.treated_results
        for name, (vl, chg) in self.egfr_metrics.items():
            bval = b["egfr"][min(yr, len(b["egfr"])-1)]
            tval = t["egfr"][min(yr, len(t["egfr"])-1)]
            vl.setText(f"{tval:.0f}")
            diff = tval - bval
            if diff != 0:
                col = "#4ade80" if diff > 0 else "#ef4444"
                sign = "+" if diff > 0 else ""
                chg.setText(f'<span style="color:{col};font-size:11px;">{sign}{diff:.1f} vs baseline</span>')
            else:
                chg.setText("")
    
    def _update_surv_metrics(self, yr):
        b, t = self.baseline_results, self.treated_results
        idx = min(yr, len(b["survival"])-1)
        vl_s, chg_s = self.surv_metrics["Survival"]
        vl_s.setText(f"{t['survival'][idx]:.0f}%")
        diff = t["survival"][idx] - b["survival"][idx]
        if diff != 0:
            col = "#4ade80" if diff > 0 else "#ef4444"
            sign = "+" if diff > 0 else ""
            chg_s.setText(f'<span style="color:{col};font-size:11px;">{sign}{diff:.1f}pp vs baseline</span>')
        else:
            chg_s.setText("")
        vl_r, chg_r = self.surv_metrics["Annual Risk"]
        vl_r.setText(f"{t['death_prob_annual'][idx]:.2f}%")
        diff = t["death_prob_annual"][idx] - b["death_prob_annual"][idx]
        if diff != 0:
            col = "#4ade80" if diff < 0 else "#ef4444"
            sign = "+" if diff > 0 else ""
            chg_r.setText(f'<span style="color:{col};font-size:11px;">{sign}{diff:.2f}pp vs baseline</span>')
        else:
            chg_r.setText("")
    
    def _load_bp_chart(self):
        b, t = self.baseline_results, self.treated_results
        years = b["year"]
        n = len(years)
        iv_year = self.current_intervention.start_year
        frames = []
        for i in range(1, n + 1):
            frames.append(go.Frame(data=[
                go.Scatter(x=years[:i], y=b["sbp"][:i], line=dict(color=C["sbp"], width=2.5), mode="lines", name="Baseline SBP"),
                go.Scatter(x=years[:i], y=t["sbp"][:i], line=dict(color=C["sbp"], width=2.5, dash="dash"), mode="lines", name="Treated SBP"),
                go.Scatter(x=years[:i], y=b["dbp"][:i], line=dict(color=C["dbp"], width=2.5), mode="lines", name="Baseline DBP", showlegend=False),
                go.Scatter(x=years[:i], y=t["dbp"][:i], line=dict(color=C["dbp"], width=2.5, dash="dash"), mode="lines", name="Treated DBP", showlegend=False),
                go.Scatter(x=years[:i], y=b["pp"][:i], line=dict(color=C["pp"], width=1.5, dash="dot"), mode="lines", name="Baseline PP", opacity=min(1.0, i / max(1, n*0.4))),
                go.Scatter(x=years[:i], y=t["pp"][:i], line=dict(color=C["pp"], width=1.5, dash="dot"), mode="lines", name="Treated PP", opacity=min(1.0, i / max(1, n*0.4)), showlegend=False),
            ], name=str(i)))
        fig = go.Figure(data=[
            go.Scatter(x=[], y=[], line=dict(color=C["sbp"], width=2.5), mode="lines", name="Baseline SBP"),
            go.Scatter(x=[], y=[], line=dict(color=C["sbp"], width=2.5, dash="dash"), mode="lines", name="Treated SBP"),
            go.Scatter(x=[], y=[], line=dict(color=C["dbp"], width=2.5), mode="lines", name="Baseline DBP", showlegend=False),
            go.Scatter(x=[], y=[], line=dict(color=C["dbp"], width=2.5, dash="dash"), mode="lines", name="Treated DBP", showlegend=False),
            go.Scatter(x=[], y=[], line=dict(color=C["pp"], width=1.5, dash="dot"), mode="lines", name="Baseline PP"),
            go.Scatter(x=[], y=[], line=dict(color=C["pp"], width=1.5, dash="dot"), mode="lines", name="Treated PP", showlegend=False),
        ], frames=frames)
        fig.add_hline(y=140, line_dash="dash", line_color="rgba(239,68,68,0.3)", annotation_text="Stage 2", annotation_font_color="#ef4444", annotation_font_size=10)
        fig.add_hline(y=130, line_dash="dash", line_color="rgba(249,115,22,0.3)", annotation_text="Stage 1", annotation_font_color="#f97316", annotation_font_size=10)
        fig.add_vline(x=iv_year, line_dash="dot", line_color="rgba(255,255,255,0.4)", annotation_text=f"{self.current_intervention.name} starts", annotation_font_color="white", annotation_font_size=9)
        layout = base_layout("Blood Pressure — Baseline vs Treated (mmHg)")
        layout.update(xaxis=dict(range=[0, max(years)], gridcolor=C["border"], zerolinecolor=C["border"], tickfont=dict(color=C["muted"], size=10), title_text="Years from baseline", title_font_color=C["muted"]), yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"], tickfont=dict(color=C["muted"], size=10), title_text="mmHg", title_font_color=C["muted"]), updatemenus=[dict(type="buttons", showactive=False, visible=False, buttons=[dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=55, redraw=True), fromcurrent=True, transition=dict(duration=0), mode="immediate")])])], legend=dict(orientation="h", y=1.02, x=0.5, font=dict(color=C["text"], size=10), bgcolor="rgba(0,0,0,0)"))
        fig.update_layout(**layout)
        import json as _json
        data_js = _json.dumps({"years": years, "bsbp": b["sbp"], "tsbp": t["sbp"], "bdbp": b["dbp"], "tdbp": t["dbp"], "bpp": b["pp"], "tpp": t["pp"]})
        autoplay_js = f"""
<script>
var _fsdata_bp = {data_js};
document.addEventListener('DOMContentLoaded', function() {{
    setTimeout(function() {{
        var gd = document.querySelector('.plotly-graph-div');
        if (gd) Plotly.animate(gd, null, {{frame:{{duration:55,redraw:true}},transition:{{duration:0}},mode:'immediate'}});
    }}, 300);
}});
function updateBPYear(yr) {{
    var gd = document.querySelector('.plotly-graph-div');
    if (!gd) return;
    var d = _fsdata_bp, x = d.years.slice(0, yr+1);
    Plotly.react(gd, [
        {{x:x, y:d.bsbp.slice(0,yr+1), type:'scatter',mode:'lines',name:'Baseline SBP',line:{{color:'#ef4444',width:2.5}}}},
        {{x:x, y:d.tsbp.slice(0,yr+1), type:'scatter',mode:'lines',name:'Treated SBP',line:{{color:'#ef4444',width:2.5,dash:'dash'}}}},
        {{x:x, y:d.bdbp.slice(0,yr+1), type:'scatter',mode:'lines',name:'Baseline DBP',line:{{color:'#f97316',width:2.5}},showlegend:false}},
        {{x:x, y:d.tdbp.slice(0,yr+1), type:'scatter',mode:'lines',name:'Treated DBP',line:{{color:'#f97316',width:2.5,dash:'dash'}},showlegend:false}},
        {{x:x, y:d.bpp.slice(0,yr+1), type:'scatter',mode:'lines',name:'Baseline PP',line:{{color:'#a855f7',width:1.5,dash:'dot'}}}},
        {{x:x, y:d.tpp.slice(0,yr+1), type:'scatter',mode:'lines',name:'Treated PP',line:{{color:'#a855f7',width:1.5,dash:'dot'}},showlegend:false}},
    ], gd.layout);
}}
</script>"""
        html = fig.to_html(include_plotlyjs="cdn", full_html=True, config={"displayModeBar": False, "responsive": True})
        html = html.replace("</body>", autoplay_js + "</body>")
        self.bp_chart.setHtml(html)
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(3400, lambda: self._update_bp_metrics(0))
    
    def _load_egfr_chart(self):
        b, t = self.baseline_results, self.treated_results
        years = b["year"]
        n = len(years)
        iv_year = self.current_intervention.start_year
        frames = []
        for i in range(1, n + 1):
            frames.append(go.Frame(data=[go.Scatter(x=years[:i], y=b["egfr"][:i], line=dict(color=C["egfr"], width=2.5), mode="lines", name="Baseline eGFR", fill="tozeroy", fillcolor="rgba(34,211,238,0.05)"), go.Scatter(x=years[:i], y=t["egfr"][:i], line=dict(color=C["egfr"], width=2.5, dash="dash"), mode="lines", name="Treated eGFR", fill="tozeroy", fillcolor="rgba(34,211,238,0.03)")], name=str(i)))
        fig = go.Figure(data=[go.Scatter(x=[], y=[], line=dict(color=C["egfr"], width=2.5), mode="lines", name="Baseline eGFR", fill="tozeroy", fillcolor="rgba(34,211,238,0.05)"), go.Scatter(x=[], y=[], line=dict(color=C["egfr"], width=2.5, dash="dash"), mode="lines", name="Treated eGFR", fill="tozeroy", fillcolor="rgba(34,211,238,0.03)")], frames=frames)
        for val, label, col in [(90,"Normal","rgba(74,222,128,0.4)"),(60,"CKD Stage 3","rgba(249,115,22,0.4)"),(30,"CKD Stage 4","rgba(239,68,68,0.4)")]:
            fig.add_hline(y=val, line_dash="dash", line_color=col, annotation_text=label, annotation_font_size=10, annotation_font_color=col.replace("0.4","1"))
        fig.add_vline(x=iv_year, line_dash="dot", line_color="rgba(255,255,255,0.4)", annotation_text=f"{self.current_intervention.name} starts", annotation_font_color="white", annotation_font_size=9)
        layout = base_layout("Kidney Function — Baseline vs Treated (eGFR ml/min)")
        layout.update(xaxis=dict(range=[0, max(years)], gridcolor=C["border"], zerolinecolor=C["border"], tickfont=dict(color=C["muted"], size=10), title_text="Years from baseline", title_font_color=C["muted"]), yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"], tickfont=dict(color=C["muted"], size=10), title_text="ml/min", title_font_color=C["muted"]), updatemenus=[dict(type="buttons", showactive=False, visible=False, buttons=[dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=55, redraw=True), fromcurrent=True, transition=dict(duration=0), mode="immediate")])])], legend=dict(orientation="h", y=1.02, x=0.5, font=dict(color=C["text"], size=10), bgcolor="rgba(0,0,0,0)"))
        fig.update_layout(**layout)
        import json as _json
        data_js = _json.dumps({"years": years, "begfr": b["egfr"], "tegfr": t["egfr"]})
        autoplay_js = f"""
<script>
var _fsdata_egfr = {data_js};
document.addEventListener('DOMContentLoaded', function() {{
    setTimeout(function() {{
        var gd = document.querySelector('.plotly-graph-div');
        if (gd) Plotly.animate(gd, null, {{frame:{{duration:55,redraw:true}},transition:{{duration:0}},mode:'immediate'}});
    }}, 300);
}});
function updateEgfrYear(yr) {{
    var gd = document.querySelector('.plotly-graph-div');
    if (!gd) return;
    var d = _fsdata_egfr, x = d.years.slice(0, yr+1);
    Plotly.react(gd, [
        {{x:x, y:d.begfr.slice(0,yr+1), type:'scatter',mode:'lines',name:'Baseline eGFR',line:{{color:'#22d3ee',width:2.5}},fill:'tozeroy',fillcolor:'rgba(34,211,238,0.05)'}},
        {{x:x, y:d.tegfr.slice(0,yr+1), type:'scatter',mode:'lines',name:'Treated eGFR',line:{{color:'#22d3ee',width:2.5,dash:'dash'}},fill:'tozeroy',fillcolor:'rgba(34,211,238,0.03)'}},
    ], gd.layout);
}}
</script>"""
        html = fig.to_html(include_plotlyjs="cdn", full_html=True, config={"displayModeBar": False, "responsive": True})
        html = html.replace("</body>", autoplay_js + "</body>")
        self.egfr_chart.setHtml(html)
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(3400, lambda: self._update_egfr_metrics(0))
    
    def _load_surv_chart(self):
        b, t = self.baseline_results, self.treated_results
        years = b["year"]
        n = len(years)
        iv_year = self.current_intervention.start_year
        frames = []
        for i in range(1, n + 1):
            frames.append(go.Frame(data=[go.Scatter(x=years[:i], y=b["survival"][:i], line=dict(color=C["survival"], width=2.5), mode="lines", name="Baseline Survival", fill="tozeroy", fillcolor="rgba(74,222,128,0.05)"), go.Scatter(x=years[:i], y=t["survival"][:i], line=dict(color=C["survival"], width=2.5, dash="dash"), mode="lines", name="Treated Survival", fill="tozeroy", fillcolor="rgba(74,222,128,0.03)"), go.Scatter(x=years[:i], y=[x*100 for x in b["death_prob_annual"][:i]], line=dict(color="#ef4444", width=1.5), mode="lines", name="Baseline Risk", yaxis="y2"), go.Scatter(x=years[:i], y=[x*100 for x in t["death_prob_annual"][:i]], line=dict(color="#ef4444", width=1.5, dash="dash"), mode="lines", name="Treated Risk", yaxis="y2")], name=str(i)))
        fig = go.Figure(data=[go.Scatter(x=[], y=[], line=dict(color=C["survival"], width=2.5), mode="lines", name="Baseline Survival", fill="tozeroy", fillcolor="rgba(74,222,128,0.05)"), go.Scatter(x=[], y=[], line=dict(color=C["survival"], width=2.5, dash="dash"), mode="lines", name="Treated Survival", fill="tozeroy", fillcolor="rgba(74,222,128,0.03)"), go.Scatter(x=[], y=[], line=dict(color="#ef4444", width=1.5), mode="lines", name="Baseline Risk", yaxis="y2"), go.Scatter(x=[], y=[], line=dict(color="#ef4444", width=1.5, dash="dash"), mode="lines", name="Treated Risk", yaxis="y2")], frames=frames)
        fig.add_vline(x=iv_year, line_dash="dot", line_color="rgba(255,255,255,0.4)", annotation_text=f"{self.current_intervention.name} starts", annotation_font_color="white", annotation_font_size=9)
        # Calculate y-axis range starting from baseline final survival to show improvement
        baseline_final_surv = b["survival"][-1]
        max_surv = max(max(b["survival"]), max(t["survival"]))
        y_range = [baseline_final_surv - 5, max_surv + 5]
        layout = base_layout("Survival & Annual Death Risk — Baseline vs Treated")
        layout.update(xaxis=dict(range=[0, max(years)], gridcolor=C["border"], zerolinecolor=C["border"], tickfont=dict(color=C["muted"], size=10), title_text="Years from baseline", title_font_color=C["muted"]), yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"], tickfont=dict(color=C["muted"], size=10), title_text="Survival %", title_font_color=C["muted"], range=y_range), yaxis2=dict(gridcolor=C["border"], zerolinecolor=C["border"], tickfont=dict(color="#ef4444", size=9), title_text="Risk %", title_font_color="#ef4444", overlaying="y", side="right", showgrid=False), updatemenus=[dict(type="buttons", showactive=False, visible=False, buttons=[dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=55, redraw=True), fromcurrent=True, transition=dict(duration=0), mode="immediate")])])], legend=dict(orientation="h", y=1.02, x=0.5, font=dict(color=C["text"], size=10), bgcolor="rgba(0,0,0,0)"))
        fig.update_layout(**layout)
        import json as _json
        data_js = _json.dumps({"years": years, "bsurv": b["survival"], "tsurv": t["survival"], "brisk": [x*100 for x in b["death_prob_annual"]], "trisk": [x*100 for x in t["death_prob_annual"]]})
        autoplay_js = f"""
<script>
var _fsdata_surv = {data_js};
document.addEventListener('DOMContentLoaded', function() {{
    setTimeout(function() {{
        var gd = document.querySelector('.plotly-graph-div');
        if (gd) Plotly.animate(gd, null, {{frame:{{duration:55,redraw:true}},transition:{{duration:0}},mode:'immediate'}});
    }}, 300);
}});
function updateSurvYear(yr) {{
    var gd = document.querySelector('.plotly-graph-div');
    if (!gd) return;
    var d = _fsdata_surv, x = d.years.slice(0, yr+1);
    Plotly.react(gd, [
        {{x:x, y:d.bsurv.slice(0,yr+1), type:'scatter',mode:'lines',name:'Baseline Survival',line:{{color:'#4ade80',width:2.5}},fill:'tozeroy',fillcolor:'rgba(74,222,128,0.05)'}},
        {{x:x, y:d.tsurv.slice(0,yr+1), type:'scatter',mode:'lines',name:'Treated Survival',line:{{color:'#4ade80',width:2.5,dash:'dash'}},fill:'tozeroy',fillcolor:'rgba(74,222,128,0.03)'}},
        {{x:x, y:d.brisk.slice(0,yr+1), type:'scatter',mode:'lines',name:'Baseline Risk',line:{{color:'#ef4444',width:1.5}},yaxis:'y2'}},
        {{x:x, y:d.trisk.slice(0,yr+1), type:'scatter',mode:'lines',name:'Treated Risk',line:{{color:'#ef4444',width:1.5,dash:'dash'}},yaxis:'y2'}},
    ], gd.layout);
}}
</script>"""
        html = fig.to_html(include_plotlyjs="cdn", full_html=True, config={"displayModeBar": False, "responsive": True})
        html = html.replace("</body>", autoplay_js + "</body>")
        self.surv_chart.setHtml(html)
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(3400, lambda: self._update_surv_metrics(0))

# ── Main Window ───────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.engine   = Engine()
        self.worker   = None
        self.registry = None
        self.aria_panel  = None
        self._build()

    def _build(self):
        self.setWindowTitle(
            "FlowstateAI v.2  —  Hypertension Digital Twin")
        self.setMinimumSize(1280, 820)
        self.setWindowFlags(
        Qt.WindowType.Window |
        Qt.WindowType.WindowMinimizeButtonHint |
        Qt.WindowType.WindowMaximizeButtonHint |
        Qt.WindowType.WindowCloseButtonHint
        )
        self.setStyleSheet(STYLE)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Top bar
        topbar = QWidget()
        topbar.setFixedHeight(56)
        topbar.setStyleSheet(
            "background:#06060d;border-bottom:1px solid #1e1e2e;")
        tbl = QHBoxLayout(topbar)
        tbl.setContentsMargins(32, 0, 32, 0)

        brand = QLabel("FlowstateAI")
        brand.setStyleSheet(
            "color:#e2e8f0;font-size:15px;"
            "font-family:'Georgia',serif;letter-spacing:2px;")
        version = QLabel("v.2  ·  Hypertension Digital Twin")
        version.setStyleSheet(
            "color:#334155;font-size:11px;"
            "font-family:'Georgia',serif;letter-spacing:1px;")
        tbl.addWidget(brand)
        tbl.addSpacing(12)
        tbl.addWidget(version)
        tbl.addStretch()

        registry_btn = QPushButton("Patient Registry")
        registry_btn.setObjectName("secondary")
        registry_btn.setFixedHeight(32)
        registry_btn.clicked.connect(self._open_registry)
        tbl.addWidget(registry_btn)

        tbl.addSpacing(8)
        aria_btn = QPushButton("✦ ARIA")
        aria_btn.setFixedHeight(32)
        aria_btn.setStyleSheet("""
            QPushButton {
                background: #1e1e2e;
                color: #6366f1;
                border: 1px solid #6366f1;
                border-radius: 6px;
                padding: 6px 16px;
                font-size: 12px;
                font-family: 'Georgia', serif;
            }
            QPushButton:hover { background: #2d2d3f; }
        """)
        aria_btn.clicked.connect(self._toggle_aria)
        tbl.addWidget(aria_btn)

        root.addWidget(topbar)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.tab_patient       = PatientTab()
        self.tab_trajectory    = TrajectoryTab()
        self.tab_metrics       = MetricsTab()
        self.tab_interventions = InterventionsTab()
        self.tab_intervention_results = InterventionResultsTab()

        self.tabs.addTab(self.tab_patient,        "  Patient  ")
        self.tabs.addTab(self.tab_trajectory,     "  Trajectory  ")
        self.tabs.addTab(self.tab_metrics,        "  Metrics  ")
        self.tabs.addTab(self.tab_interventions,  "  Interventions  ")
        self.tabs.addTab(self.tab_intervention_results,  "  Intervention Results  ")

        # Content row — tabs + ARIA panel side by side

        self.aria_panel = ARIAPanel()
        self.aria_panel.setVisible(False)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(4)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setOpaqueResize(True)
        self.splitter.setStyleSheet("""
            QSplitter {
                background: #0a0a0f;
            }
            QSplitter::handle:horizontal {
                background: #1a1a2e;
                width: 4px;
            }
            QSplitter::handle:horizontal:hover {
                background: #6366f1;
            }
            QSplitter::handle:horizontal:pressed {
                background: #4f46e5;
            }
        """)
        self.splitter.addWidget(self.tabs)
        self.splitter.addWidget(self.aria_panel)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setSizes([900, 420])

        root.addWidget(self.splitter)

        self.tab_patient.simulate_requested.connect(self._run_sim)
        self.tab_trajectory.set_save_callback(self._save_report)
        self.tab_interventions.intervention_run.connect(self._on_intervention_run)

    def _open_registry(self):
        if self.registry is None:
            self.registry = PatientRegistry(parent=self)
            self.registry.simulate_requested.connect(
                self._simulate_from_registry)
        self.registry.setWindowFlags(Qt.WindowType.Window)
        self.registry.show()
        self.registry.raise_()

    def _simulate_from_registry(self, patient):
        if self.registry:
            self.registry.close()
        self.tab_patient.load_patient(patient)
        self.tabs.setCurrentIndex(0)
        self.tab_patient._on_simulate()

    def _run_sim(self, params):
        self._last_params = params
        self.worker = SimWorker(self.engine, params)
        self.worker.progress.connect(self.tab_patient.set_progress)
        self.worker.finished.connect(self._on_results)
        self.worker.start()
        
    def _toggle_aria(self):
        visible = not self.aria_panel.isVisible()
        self.aria_panel.setVisible(visible)
        if visible:
            total = self.splitter.width()
            self.splitter.setSizes([total - 420, 420])
        else:
            self.splitter.setSizes([self.splitter.width(), 0])
    def _on_results(self, results):
        self._last_results = results
        self.tab_patient.set_done()
        self.tab_trajectory.update(results)
        self.tab_metrics.update(results)
        if self.aria_panel:
            self.aria_panel.set_context(self._last_params, results, 0)
        self._autosave_patient(self._last_params)
        self.tabs.setCurrentIndex(1)

    def _autosave_patient(self, params):
        """
        Silently saves the simulated patient to the registry if not
        already present (matched by age + sex + sbp + dbp).
        Does not interrupt the user — no dialog, no confirmation.
        """
        import datetime
        registry_path = Path("patients/registry.json")
        registry_path.parent.mkdir(exist_ok=True)

        try:
            if registry_path.exists():
                with open(registry_path) as f:
                    registry = json.load(f)
            else:
                registry = []

            # Check for duplicate — same age, sex, sbp, dbp
            for existing in registry:
                if (existing.get("age")  == params.get("age")  and
                    existing.get("sex")  == params.get("sex")  and
                    existing.get("sbp")  == params.get("sbp")  and
                    existing.get("dbp")  == params.get("dbp")):
                    return  # already saved, skip silently

            # Generate next patient ID
            existing_ids = [p.get("id", "P000") for p in registry]
            max_num = 0
            for pid in existing_ids:
                try:
                    max_num = max(max_num, int(pid[1:]))
                except (ValueError, IndexError):
                    pass
            new_id = f"P{max_num + 1:03d}"

            sex_str = "Male" if params.get("sex") == 1 else "Female"
            now     = datetime.datetime.now().strftime("%Y-%m-%d")

            patient = {
                "id":            new_id,
                "name":          f"Auto — {sex_str}, Age {params.get('age')}",
                "sex":           sex_str,
                "age":           params.get("age"),
                "sbp":           params.get("sbp"),
                "dbp":           params.get("dbp"),
                "waist_cm":      params.get("waist_cm"),
                "heart_rate":    params.get("heart_rate"),
                "egfr":          params.get("egfr"),
                "pack_years":    params.get("pack_years"),
                "alcohol_g_day": params.get("alcohol_g_day"),
                "sodium_mg_day": params.get("sodium_mg_day"),
                "notes":         "Auto-saved from simulation.",
                "created":       now,
                "last_updated":  now,
            }

            registry.append(patient)
            with open(registry_path, "w") as f:
                json.dump(registry, f, indent=2)

        except Exception:
            pass  # Never interrupt the user for a background save failure

    def _save_report(self):
        """
        Generates ARIA summary then saves .fsa file.
        Called when user clicks Save Report on Trajectory tab.
        """
        if not hasattr(self, '_last_params') or not hasattr(self, '_last_results'):
            return

        from pathlib import Path
        import datetime

        # Ask ARIA to generate a summary first
        self.aria_panel.set_context(
            self._last_params, self._last_results, 0)

        # Build the summary via a one-shot API call
        self._generate_aria_summary(self._last_params, self._last_results)

    def _generate_aria_summary(self, params, results):
        """Fires a background API call to get ARIA summary, then saves."""
        from aria import APIWorker, build_system_prompt, load_config

        cfg = load_config()
        provider = cfg.get("provider", "groq")
        api_key = (cfg.get("gemini_api_key")
                   if provider == "gemini"
                   else cfg.get("groq_api_key"))
        model = ("gemini-2.5-flash"
                 if provider == "gemini"
                 else "llama-3.3-70b-versatile")

        if not api_key:
            self._write_fsa(params, results, "No API key configured.")
            return

        system = build_system_prompt(params, results, 0)
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content":
            "Generate a concise clinical summary of this patient's "
            "simulated trajectory. Cover their baseline risk, how their "
            "blood pressure and kidney function evolve, key milestones, "
            "and what the survival probability indicates. "
            "3-4 paragraphs, plain language."}
        ]

        self._summary_worker = APIWorker(provider, api_key, messages, model)
        self._summary_worker.finished.connect(
            lambda text: self._write_fsa(params, results, text))
        self._summary_worker.error.connect(
            lambda err: self._write_fsa(params, results,
                                        f"Summary unavailable: {err}"))

        # Show saving status
        self.tab_trajectory.save_btn.setEnabled(False)
        self.tab_trajectory.save_btn.setText("Generating summary…")
        self._summary_worker.start()

    def _write_fsa(self, params, results, aria_summary):
        """Writes the .fsa file to the reports/ folder."""
        import datetime

        Path("reports").mkdir(exist_ok=True)

        sex_str  = "M" if params.get("sex") == 1 else "F"
        stamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default  = f"patient_{params.get('age')}_{sex_str}_{stamp}"

        from PyQt6.QtWidgets import QInputDialog
        report_name, ok = QInputDialog.getText(
            self, "Save Report",
            "Report name:",
            text=default
        )
        if not ok:
            self.tab_trajectory.save_btn.setEnabled(True)
            self.tab_trajectory.save_btn.setText("↓  Save Report")
            return
        report_name = report_name.strip() or default
        # Sanitise — remove characters invalid in filenames
        import re
        report_name = re.sub(r'[<>:"/\\|?*]', '_', report_name)
        filename = f"reports/{report_name}.fsa"

        data = {
            "fsa_version":  "2.0",
            "created":      datetime.datetime.now().isoformat(),
            "patient":      params,
            "results":      results,
            "aria_summary": aria_summary,
            "notes":        ""
        }

        try:
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)

            self.tab_trajectory.save_btn.setEnabled(True)
            self.tab_trajectory.save_btn.setText("↓  Save Report")

            # Ask user if they want to open the viewer
            from PyQt6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self, "Report Saved",
                f"Report saved to:\n{filename}\n\nOpen in Report Viewer?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                import subprocess
                subprocess.Popen(
                    [sys.executable, "report_viewer.py", filename],
                    cwd=Path(__file__).parent
                )
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", str(e))
            self.tab_trajectory.save_btn.setEnabled(True)
            self.tab_trajectory.save_btn.setText("↓  Save Report")

    def _on_intervention_run(self, params, intervention_name, baseline_results, treated_results):
        """Handle intervention run - load results in InterventionResultsTab"""
        # Load results in InterventionResultsTab
        self.tab_intervention_results.load_results(intervention_name, baseline_results, treated_results)
        
        # Update ARIA with intervention context
        if self.aria_panel:
            self.aria_panel.set_context(params, treated_results, 0, intervention_name, baseline_results)
        
        # Switch to InterventionResultsTab
        self.tabs.setCurrentIndex(4)

###############

# ── Entry point ───────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", default=None,
                        help="Path to patient JSON to pre-load")
    args, _ = parser.parse_known_args()

    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style to avoid font issues
    
    try:
        win = MainWindow()
        win.show()
    except Exception as e:
        print(f"Error creating MainWindow: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if args.load:
        try:
            with open(args.load) as f:
                patient = json.load(f)
            win.tab_patient.load_patient(patient)
            win.tab_patient._on_simulate()
        except Exception as e:
            print(f"Could not load patient: {e}")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
