# report_viewer.py
#
# FlowState AI v.2 — Report Viewer
# Opens .fsa (FlowState Archive) files and displays patient simulation summaries.
#
# Usage:
#   python report_viewer.py                  — opens file picker
#   python report_viewer.py path/to/file.fsa — opens directly

import sys
import json
import subprocess
import tempfile
import os
from pathlib import Path
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from aria import ARIAPanel

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QFrame, QScrollArea,
    QTextEdit, QFileDialog, QMessageBox, QSplitter, QGridLayout
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

# ── Colour palette (matches main.py) ─────────────────────────────────────
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

STYLE = """
QMainWindow, QWidget {
    background-color: #0a0a0f;
    color: #e2e8f0;
    font-family: 'Segoe UI', Arial, sans-serif;
}
QFrame#card {
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
}
QLabel#heading {
    font-size: 20px;
    color: #e2e8f0;
    font-family: 'Georgia', serif;
}
QLabel#subheading {
    font-size: 11px;
    color: #64748b;
    font-family: 'Georgia', serif;
}
QLabel#field_label {
    font-size: 10px;
    color: #64748b;
    letter-spacing: 1px;
}
QLabel#value {
    font-size: 13px;
    color: #e2e8f0;
}
QTextEdit {
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    color: #e2e8f0;
    font-size: 12px;
    font-family: 'Segoe UI', Arial, sans-serif;
    padding: 8px;
}
QPushButton#primary {
    background: #6366f1;
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 12px;
    font-family: 'Segoe UI', Arial, sans-serif;
}
QPushButton#primary:hover { background: #818cf8; }
QPushButton#secondary {
    background: #1e1e2e;
    color: #94a3b8;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 12px;
    font-family: 'Segoe UI', Arial, sans-serif;
}
QPushButton#secondary:hover { background: #2d2d3f; color: #e2e8f0; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical {
    background: #0a0a0f; width: 6px; border-radius: 3px;
}
QScrollBar::handle:vertical {
    background: #1e1e2e; border-radius: 3px;
}
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

def base_layout(title):
    return dict(
        title=dict(text=title,
                   font=dict(color=C["text"], size=12,
                             family="Georgia, serif"),
                   x=0.01),
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["bg"],
        font=dict(color=C["text"], family="Georgia, serif"),
        margin=dict(l=48, r=16, t=44, b=36),
        xaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"],
                   tickfont=dict(color=C["muted"], size=10)),
        yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"],
                   tickfont=dict(color=C["muted"], size=10)),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=C["bg"], font_color=C["text"],
                        bordercolor=C["border"]),
        legend=dict(orientation="h", y=1.08, x=0,
                    font=dict(color=C["text"], size=10),
                    bgcolor="rgba(0,0,0,0)")
    )

def fig_to_html(fig):
    return fig.to_html(
        include_plotlyjs="cdn", full_html=True,
        config={"displayModeBar": False, "responsive": True})

def build_bp_chart(results):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results["year"], y=results["sbp"], name="SBP",
        line=dict(color=C["sbp"], width=2.5), mode="lines"))
    fig.add_trace(go.Scatter(
        x=results["year"], y=results["dbp"], name="DBP",
        line=dict(color=C["dbp"], width=2.5),
        fill="tonexty", fillcolor="rgba(239,68,68,0.07)",
        mode="lines"))
    fig.add_trace(go.Scatter(
        x=results["year"], y=results["pp"], name="Pulse Pressure",
        line=dict(color=C["pp"], width=1.5, dash="dot"),
        mode="lines"))
    fig.add_hline(y=140, line_dash="dash",
                  line_color="rgba(239,68,68,0.3)",
                  annotation_text="Stage 2",
                  annotation_font_color="#ef4444",
                  annotation_font_size=9)
    fig.add_hline(y=130, line_dash="dash",
                  line_color="rgba(249,115,22,0.3)",
                  annotation_text="Stage 1",
                  annotation_font_color="#f97316",
                  annotation_font_size=9)
    fig.update_layout(**base_layout("Blood Pressure Trajectory (mmHg)"))
    fig.update_yaxes(title_text="mmHg", title_font_color=C["muted"])
    fig.update_xaxes(title_text="Years from baseline",
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
        (90, "Normal", "rgba(74,222,128,0.4)"),
        (60, "CKD Stage 3", "rgba(249,115,22,0.4)"),
        (30, "CKD Stage 4", "rgba(239,68,68,0.4)"),
    ]:
        fig.add_hline(y=val, line_dash="dash", line_color=col,
                      annotation_text=label, annotation_font_size=9,
                      annotation_font_color=col.replace("0.4", "1"))
    fig.update_layout(**base_layout("eGFR — Kidney Function"))
    fig.update_yaxes(title_text="eGFR", title_font_color=C["muted"])
    fig.update_xaxes(title_text="Years", title_font_color=C["muted"])
    return fig

def build_survival_chart(results):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results["year"], y=results["survival"],
        name="Survival",
        line=dict(color=C["survival"], width=2.5),
        fill="tozeroy", fillcolor="rgba(74,222,128,0.07)",
        mode="lines"))
    fig.add_hline(y=50, line_dash="dash",
                  line_color="rgba(239,68,68,0.3)",
                  annotation_text="50%",
                  annotation_font_color="#ef4444",
                  annotation_font_size=9)
    fig.update_layout(**base_layout("Cumulative Survival (%)"))
    fig.update_yaxes(range=[0, 105], title_text="%",
                     title_font_color=C["muted"])
    fig.update_xaxes(title_text="Years", title_font_color=C["muted"])
    return fig


# ── Report viewer window ──────────────────────────────────────────────────
class ReportViewer(QMainWindow):

    def __init__(self, fsa_path=None):
        super().__init__()
        self.fsa_path = fsa_path
        self.data     = None
        self.setWindowTitle("FlowState AI — Report Viewer")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(STYLE)

        if fsa_path:
            self._load_file(fsa_path)
        else:
            self._show_picker()

    # ── File loading ──────────────────────────────────────────────────────
    def _show_picker(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open FlowState Report",
            "reports/",
            "FlowState Archive (*.fsa);;JSON Files (*.json);;All Files (*)")
        if path:
            self._load_file(path)
        else:
            sys.exit(0)

    def _load_file(self, path):
        try:
            with open(path) as f:
                self.data = json.load(f)
            self.fsa_path = path
            self._build_ui()
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Could not open file:\n{e}")
            sys.exit(1)

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        d = self.data
        p = d.get("patient", {})
        r = d.get("results", {})
        aria_summary = d.get("aria_summary", "No ARIA summary available.")
        notes        = d.get("notes", "")
        created      = d.get("created", "")
        sex_str      = "Male" if p.get("sex") == 1 else "Female"

        self.setWindowTitle(
            f"FlowState Report — {sex_str}, Age {p.get('age')} — "
            f"{created[:10] if created else 'Unknown date'}")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Store for ARIA context injection after UI builds
        self._report_patient = p
        self._report_results = r

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
        version = QLabel("v.2  ·  Report Viewer")
        version.setStyleSheet(
            "color:#334155;font-size:11px;"
            "font-family:'Georgia',serif;letter-spacing:1px;")
        tbl.addWidget(brand)
        tbl.addSpacing(12)
        tbl.addWidget(version)
        tbl.addStretch()

        # File info
        file_lbl = QLabel(
            Path(self.fsa_path).name if self.fsa_path else "")
        file_lbl.setStyleSheet(
            "color:#334155;font-size:10px;"
            "font-family:'Georgia',serif;")
        tbl.addWidget(file_lbl)
        tbl.addSpacing(16)

        resim_btn = QPushButton("⟳  Resimulate")
        resim_btn.setObjectName("primary")
        resim_btn.setFixedHeight(32)
        resim_btn.clicked.connect(self._resimulate)
        tbl.addWidget(resim_btn)

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

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background:#0a0a0f;border:none;")
        content = QWidget()
        content.setStyleSheet("background:#0a0a0f;")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(40, 36, 40, 40)
        cl.setSpacing(24)
        scroll.setWidget(content)

        # Content + ARIA side by side
        from PyQt6.QtWidgets import QSplitter
        from PyQt6.QtGui import QColor

        self.aria_panel = ARIAPanel()
        self.aria_panel.setVisible(False)

        # Inject report context into ARIA
        # Build a params dict from saved patient data
        aria_params = {
            "age":           p.get("age"),
            "sex":           p.get("sex"),
            "sbp":           p.get("sbp"),
            "dbp":           p.get("dbp"),
            "waist_cm":      p.get("waist_cm"),
            "heart_rate":    p.get("heart_rate"),
            "egfr":          p.get("egfr"),
            "pack_years":    p.get("pack_years"),
            "alcohol_g_day": p.get("alcohol_g_day"),
            "sodium_mg_day": p.get("sodium_mg_day"),
        }
        self.aria_panel.set_context(aria_params, r, 0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(4)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setStyleSheet("""
            QSplitter { background: #0a0a0f; }
            QSplitter::handle:horizontal {
                background: #1a1a2e; width: 4px;
            }
            QSplitter::handle:horizontal:hover {
                background: #6366f1;
            }
        """)
        self.splitter.addWidget(scroll)
        self.splitter.addWidget(self.aria_panel)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setSizes([self.width(), 0])
        root.addWidget(self.splitter)

        # ── Patient header ────────────────────────────────────────────────
        hdr_card = card()
        hl = QHBoxLayout(hdr_card)
        hl.setContentsMargins(28, 20, 28, 20)
        hl.setSpacing(40)

        left = QWidget()
        left.setStyleSheet("background:transparent;")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(4)
        name_lbl = QLabel(
            f"{sex_str}  ·  Age {p.get('age')}  ·  "
            f"SBP {p.get('sbp')} / DBP {p.get('dbp')} mmHg")
        name_lbl.setObjectName("heading")
        ll.addWidget(name_lbl)
        date_lbl = QLabel(
            f"Simulated {created[:19].replace('T', ' ') if created else ''}  ·  "
            f"{r['year'][-1] if r.get('year') else '?'}-year horizon")
        date_lbl.setObjectName("subheading")
        ll.addWidget(date_lbl)
        hl.addWidget(left, stretch=1)

        # Summary stats on the right
        stats = [
            ("SBP rise",
             f"+{r['sbp'][-1]-r['sbp'][0]:.1f} mmHg" if r.get('sbp') else "—",
             C["sbp"]),
            ("eGFR decline",
             f"-{r['egfr'][0]-r['egfr'][-1]:.1f} ml/min" if r.get('egfr') else "—",
             C["egfr"]),
            ("Final survival",
             f"{r['survival'][-1]:.1f}%" if r.get('survival') else "—",
             C["survival"]),
        ]
        for stat_lbl, val, color in stats:
            sw = QWidget()
            sw.setStyleSheet("background:transparent;")
            sl = QVBoxLayout(sw)
            sl.setContentsMargins(0, 0, 0, 0)
            sl.setSpacing(2)
            vl = QLabel(val)
            vl.setStyleSheet(
                f"color:{color};font-size:22px;"
                f"font-family:'Georgia',serif;")
            nl = QLabel(stat_lbl)
            nl.setObjectName("field_label")
            sl.addWidget(vl)
            sl.addWidget(nl)
            hl.addWidget(sw)

        cl.addWidget(hdr_card)

        # ── Patient profile grid ──────────────────────────────────────────
        profile_card = card()
        pl = QVBoxLayout(profile_card)
        pl.setContentsMargins(28, 20, 28, 20)
        pl.setSpacing(12)
        pl.addWidget(lbl("PATIENT PROFILE", "field_label"))

        grid = QWidget()
        grid.setStyleSheet("background:transparent;")
        gl = QGridLayout(grid)
        gl.setContentsMargins(0, 8, 0, 0)
        gl.setSpacing(16)
        gl.setHorizontalSpacing(32)

        fields = [
            ("Age",          f"{p.get('age')} years"),
            ("Sex",          sex_str),
            ("Systolic BP",  f"{p.get('sbp')} mmHg"),
            ("Diastolic BP", f"{p.get('dbp')} mmHg"),
            ("Waist",        f"{p.get('waist_cm')} cm"),
            ("Heart Rate",   f"{p.get('heart_rate')} bpm"),
            ("eGFR",         f"{p.get('egfr')} ml/min/1.73m²"),
            ("Pack-Years",   f"{p.get('pack_years')}"),
            ("Alcohol",      f"{p.get('alcohol_g_day')} g/day"),
            ("Sodium",       f"{p.get('sodium_mg_day')} mg/day"),
        ]
        for i, (k, v) in enumerate(fields):
            row, col = divmod(i, 5)
            fw = QWidget()
            fw.setStyleSheet("background:transparent;")
            fwl = QVBoxLayout(fw)
            fwl.setContentsMargins(0, 0, 0, 0)
            fwl.setSpacing(2)
            fwl.addWidget(lbl(k.upper(), "field_label"))
            vl = QLabel(v)
            vl.setObjectName("value")
            fwl.addWidget(vl)
            gl.addWidget(fw, row, col)

        pl.addWidget(grid)
        cl.addWidget(profile_card)

        # ── Charts ────────────────────────────────────────────────────────
        if r:
            charts_row = QWidget()
            charts_row.setStyleSheet("background:transparent;")
            crl = QHBoxLayout(charts_row)
            crl.setContentsMargins(0, 0, 0, 0)
            crl.setSpacing(16)

            from PyQt6.QtGui import QColor
            bg = QColor("#12121a")

            bp_view = QWebEngineView()
            bp_view.setMinimumHeight(300)
            bp_view.page().setBackgroundColor(bg)
            bp_view.setHtml(fig_to_html(build_bp_chart(r)))
            crl.addWidget(bp_view, stretch=2)

            egfr_view = QWebEngineView()
            egfr_view.setMinimumHeight(300)
            egfr_view.page().setBackgroundColor(bg)
            egfr_view.setHtml(fig_to_html(build_egfr_chart(r)))
            crl.addWidget(egfr_view, stretch=1)

            surv_view = QWebEngineView()
            surv_view.setMinimumHeight(300)
            surv_view.page().setBackgroundColor(bg)
            surv_view.setHtml(fig_to_html(build_survival_chart(r)))
            crl.addWidget(surv_view, stretch=1)

            cl.addWidget(charts_row)

        # ── ARIA summary ──────────────────────────────────────────────────
        aria_card = card()
        al = QVBoxLayout(aria_card)
        al.setContentsMargins(28, 20, 28, 20)
        al.setSpacing(10)

        aria_hdr = QWidget()
        aria_hdr.setStyleSheet("background:transparent;")
        ahl = QHBoxLayout(aria_hdr)
        ahl.setContentsMargins(0, 0, 0, 0)
        aria_title = QLabel("ARIA")
        aria_title.setStyleSheet(
            "color:#6366f1;font-size:13px;"
            "font-family:'Georgia',serif;"
            "letter-spacing:2px;font-weight:bold;")
        aria_sub = QLabel("AI-generated clinical summary")
        aria_sub.setObjectName("subheading")
        ahl.addWidget(aria_title)
        ahl.addSpacing(8)
        ahl.addWidget(aria_sub)
        ahl.addStretch()
        al.addWidget(aria_hdr)

        aria_text = QLabel(aria_summary)
        aria_text.setWordWrap(True)
        aria_text.setStyleSheet(
            "color:#94a3b8;font-size:12px;"
            "font-family:'Segoe UI',Arial,sans-serif;"
            "line-height:1.6;background:transparent;border:none;")
        aria_text.setFont(QFont("Segoe UI", 10))
        aria_text.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        al.addWidget(aria_text)
        cl.addWidget(aria_card)

        # ── Notes ─────────────────────────────────────────────────────────
        notes_card = card()
        nl = QVBoxLayout(notes_card)
        nl.setContentsMargins(28, 20, 28, 20)
        nl.setSpacing(10)
        nl.addWidget(lbl("NOTES", "field_label"))

        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText(
            "Add clinical notes here…")
        self.notes_edit.setPlainText(notes)
        self.notes_edit.setFixedHeight(120)
        nl.addWidget(self.notes_edit)

        save_notes_btn = QPushButton("Save Notes")
        save_notes_btn.setObjectName("secondary")
        save_notes_btn.setFixedWidth(120)
        save_notes_btn.clicked.connect(self._save_notes)
        nl.addWidget(save_notes_btn)
        cl.addWidget(notes_card)

        cl.addStretch()

    # ── Actions ───────────────────────────────────────────────────────────
    def _save_notes(self):
        if not self.fsa_path or not self.data:
            return
        self.data["notes"] = self.notes_edit.toPlainText()
        try:
            with open(self.fsa_path, "w") as f:
                json.dump(self.data, f, indent=2)
            # Brief visual confirmation
            self.statusBar().showMessage("Notes saved.", 2000)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _resimulate(self):
        """
        Write a temp .fsa file and launch a fresh main.py instance
        with the patient pre-loaded. The temp file is picked up by
        main.py on startup via a --load argument.
        """
        if not self.data:
            return
        try:
            tmp = Path(tempfile.gettempdir()) / "_fsa_resim.json"
            with open(tmp, "w") as f:
                json.dump(self.data["patient"], f, indent=2)

            subprocess.Popen(
                [sys.executable, "main.py", "--load", str(tmp)],
                cwd=Path(__file__).parent
            )
            self.close()
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Could not launch main.py:\n{e}")
            

    def _toggle_aria(self):
        visible = not self.aria_panel.isVisible()
        self.aria_panel.setVisible(visible)
        if visible:
            total = self.splitter.width()
            self.splitter.setSizes([total - 420, 420])
        else:
            self.splitter.setSizes([self.splitter.width(), 0])


# ── Entry point ───────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))

    path = sys.argv[1] if len(sys.argv) > 1 else None
    win  = ReportViewer(path)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()