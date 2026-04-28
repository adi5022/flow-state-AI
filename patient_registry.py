# patient_registry.py
#
# FlowState AI v.2 — Patient Registry
# Standalone PyQt6 desktop application.
# Can be launched independently or embedded into main UI.
#
# Run standalone: python patient_registry.py

import sys
import json
from pathlib import Path
from datetime import date

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton, QFrame, QScrollArea,
    QComboBox, QTextEdit, QMessageBox, QFileDialog, QSizePolicy,
    QStackedWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

# ── Storage ───────────────────────────────────────────────────────────────
REGISTRY_PATH = Path("patients/registry.json")
REGISTRY_PATH.parent.mkdir(exist_ok=True)

def load_registry():
    if not REGISTRY_PATH.exists():
        return {}
    try:
        with open(REGISTRY_PATH, "r") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except json.JSONDecodeError:
        return {}
def save_registry(registry):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

def next_id(registry):
    if not registry:
        return "P001"
    nums = [int(k[1:]) for k in registry.keys() if k[1:].isdigit()]
    return f"P{max(nums)+1:03d}" if nums else "P001"

# ── Style ─────────────────────────────────────────────────────────────────
STYLE = """
QMainWindow, QWidget {
    background-color: #0a0a0f;
    color: #e2e8f0;
    font-family: 'Georgia', serif;
}
QFrame#card {
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
}
QFrame#divider {
    background: #1e1e2e;
    max-height: 1px;
}
QLineEdit, QTextEdit, QComboBox {
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-radius: 6px;
    padding: 8px 12px;
    color: #e2e8f0;
    font-size: 13px;
    font-family: 'Georgia', serif;
}
QLineEdit:focus, QTextEdit:focus {
    border: 1px solid #6366f1;
}
QComboBox::drop-down { border: none; padding-right: 8px; }
QComboBox QAbstractItemView {
    background: #12121a;
    border: 1px solid #1e1e2e;
    color: #e2e8f0;
    selection-background-color: #6366f1;
}
QPushButton#primary {
    background: #6366f1;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 13px;
    font-family: 'Georgia', serif;
}
QPushButton#primary:hover { background: #818cf8; }
QPushButton#primary:pressed { background: #4f46e5; }
QPushButton#danger {
    background: #ef4444;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 13px;
    font-family: 'Georgia', serif;
}
QPushButton#danger:hover { background: #f87171; }
QPushButton#secondary {
    background: #1e1e2e;
    color: #94a3b8;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 13px;
    font-family: 'Georgia', serif;
}
QPushButton#secondary:hover { background: #2d2d3f; color: #e2e8f0; }
QPushButton#row_btn {
    background: transparent;
    color: #6366f1;
    border: none;
    font-size: 12px;
    font-family: 'Georgia', serif;
    padding: 4px 8px;
}
QPushButton#row_btn:hover { color: #818cf8; }
QTableWidget {
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    gridline-color: #1e1e2e;
    color: #e2e8f0;
    font-family: 'Georgia', serif;
    font-size: 12px;
}
QTableWidget::item { padding: 8px; }
QTableWidget::item:selected {
    background: #1e1e3e;
    color: #e2e8f0;
}
QHeaderView::section {
    background: #0f0f1a;
    color: #6366f1;
    border: none;
    border-bottom: 1px solid #1e1e2e;
    padding: 8px;
    font-family: 'Georgia', serif;
    font-size: 11px;
    letter-spacing: 1px;
}
QScrollBar:vertical {
    background: #0a0a0f;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #1e1e2e;
    border-radius: 4px;
}
QLabel#heading {
    font-size: 22px;
    color: #e2e8f0;
    font-family: 'Georgia', serif;
}
QLabel#subheading {
    font-size: 12px;
    color: #64748b;
    font-family: 'Georgia', serif;
}
QLabel#field_label {
    font-size: 11px;
    color: #94a3b8;
    font-family: 'Georgia', serif;
    letter-spacing: 0.5px;
}
QLabel#pid_badge {
    font-size: 13px;
    color: #6366f1;
    font-family: 'Georgia', serif;
    font-weight: bold;
}
QLabel#stat_value {
    font-size: 26px;
    color: #e2e8f0;
    font-family: 'Georgia', serif;
}
QLabel#stat_label {
    font-size: 10px;
    color: #64748b;
    font-family: 'Georgia', serif;
    letter-spacing: 1px;
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

def field(label_text, placeholder="", parent=None):
    """Returns (container_widget, QLineEdit)"""
    w = QWidget(parent)
    v = QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(4)
    v.addWidget(lbl(label_text, "field_label"))
    inp = QLineEdit()
    inp.setPlaceholderText(placeholder)
    v.addWidget(inp)
    return w, inp

# ── Patient Form ──────────────────────────────────────────────────────────
class PatientForm(QWidget):
    """Reusable form for adding and editing patients."""
    saved = pyqtSignal(dict)
    cancelled = pyqtSignal()

    def __init__(self, patient=None, parent=None):
        super().__init__(parent)
        self.patient = patient  # None = new, dict = edit
        self._build()
        if patient:
            self._populate(patient)

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        # Title
        title = "Edit Patient" if self.patient else "New Patient"
        root.addWidget(lbl(title, "heading"))
        root.addWidget(lbl("Fill in the patient's clinical profile", "subheading"))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        inner = QWidget()
        form  = QVBoxLayout(inner)
        form.setSpacing(16)
        form.setContentsMargins(0, 0, 16, 0)

        # ── Basic info ────────────────────────────────────────────────────
        basic = card()
        bl    = QGridLayout(basic)
        bl.setContentsMargins(24, 20, 24, 20)
        bl.setSpacing(16)
        bl.addWidget(lbl("Basic Information", "field_label"), 0, 0, 1, 4)

        w, self.inp_name = field("Full Name", "e.g. John Doe")
        bl.addWidget(w, 1, 0, 1, 2)
        w, self.inp_dob  = field("Date of Birth", "YYYY-MM-DD")
        bl.addWidget(w, 1, 2, 1, 2)

        sex_w = QWidget()
        sv    = QVBoxLayout(sex_w)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(4)
        sv.addWidget(lbl("Sex", "field_label"))
        self.sex_combo = QComboBox()
        self.sex_combo.addItems(["Male", "Female"])
        sv.addWidget(self.sex_combo)
        bl.addWidget(sex_w, 2, 0)

        w, self.inp_age = field("Age", "35-75")
        bl.addWidget(w, 2, 1)

        form.addWidget(basic)

        # ── Clinical measurements ─────────────────────────────────────────
        clin = card()
        cl   = QGridLayout(clin)
        cl.setContentsMargins(24, 20, 24, 20)
        cl.setSpacing(16)
        cl.addWidget(lbl("Clinical Measurements", "field_label"), 0, 0, 1, 4)

        w, self.inp_sbp   = field("Systolic BP", "mmHg")
        cl.addWidget(w, 1, 0)
        w, self.inp_dbp   = field("Diastolic BP", "mmHg")
        cl.addWidget(w, 1, 1)
        w, self.inp_waist = field("Waist Circumference", "cm")
        cl.addWidget(w, 1, 2)
        w, self.inp_hr    = field("Resting Heart Rate", "bpm")
        cl.addWidget(w, 1, 3)
        w, self.inp_egfr  = field("eGFR", "ml/min/1.73m²")
        cl.addWidget(w, 2, 0)

        form.addWidget(clin)

        # ── Lifestyle ─────────────────────────────────────────────────────
        life = card()
        ll   = QGridLayout(life)
        ll.setContentsMargins(24, 20, 24, 20)
        ll.setSpacing(16)
        ll.addWidget(lbl("Lifestyle Factors", "field_label"), 0, 0, 1, 3)

        w, self.inp_pack    = field("Pack-Years", "0 if never smoked")
        ll.addWidget(w, 1, 0)
        w, self.inp_alcohol = field("Alcohol", "g/day")
        ll.addWidget(w, 1, 1)
        w, self.inp_sodium  = field("Dietary Sodium", "mg/day")
        ll.addWidget(w, 1, 2)

        form.addWidget(life)

        # ── Notes ─────────────────────────────────────────────────────────
        notes_card = card()
        nl = QVBoxLayout(notes_card)
        nl.setContentsMargins(24, 20, 24, 20)
        nl.setSpacing(8)
        nl.addWidget(lbl("Clinical Notes (optional)", "field_label"))
        self.inp_notes = QTextEdit()
        self.inp_notes.setFixedHeight(80)
        self.inp_notes.setPlaceholderText(
            "e.g. Family history of hypertension. No current medications.")
        nl.addWidget(self.inp_notes)
        form.addWidget(notes_card)
        form.addStretch()

        scroll.setWidget(inner)
        root.addWidget(scroll, stretch=1)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_row = QWidget()
        br      = QHBoxLayout(btn_row)
        br.setContentsMargins(0, 0, 0, 0)
        br.setSpacing(12)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("secondary")
        cancel_btn.clicked.connect(self.cancelled.emit)

        save_btn = QPushButton("Save Patient")
        save_btn.setObjectName("primary")
        save_btn.clicked.connect(self._on_save)

        br.addWidget(cancel_btn)
        br.addStretch()
        br.addWidget(save_btn)
        root.addWidget(btn_row)

    def _populate(self, p):
        self.inp_name.setText(p.get("name", ""))
        self.inp_dob.setText(p.get("dob", ""))
        self.sex_combo.setCurrentText(p.get("sex", "Male"))
        self.inp_age.setText(str(p.get("age", "")))
        self.inp_sbp.setText(str(p.get("sbp", "")))
        self.inp_dbp.setText(str(p.get("dbp", "")))
        self.inp_waist.setText(str(p.get("waist_cm", "")))
        self.inp_hr.setText(str(p.get("heart_rate", "")))
        self.inp_egfr.setText(str(p.get("egfr", "")))
        self.inp_pack.setText(str(p.get("pack_years", "")))
        self.inp_alcohol.setText(str(p.get("alcohol_g_day", "")))
        self.inp_sodium.setText(str(p.get("sodium_mg_day", "")))
        self.inp_notes.setPlainText(p.get("notes", ""))

    def _on_save(self):
        errors = []

        name = self.inp_name.text().strip()
        if not name:
            errors.append("Full name is required.")

        dob = self.inp_dob.text().strip()
        if len(dob) != 10 or dob[4] != "-":
            errors.append("Date of birth must be in YYYY-MM-DD format.")

        def get_float(inp, label, lo, hi):
            try:
                v = float(inp.text())
                if not lo <= v <= hi:
                    errors.append(f"{label} must be between {lo} and {hi}.")
                return v
            except ValueError:
                errors.append(f"{label} must be a number.")
                return None

        age     = get_float(self.inp_age,    "Age",              18,  100)
        sbp     = get_float(self.inp_sbp,    "Systolic BP",      70,  220)
        dbp     = get_float(self.inp_dbp,    "Diastolic BP",     40,  130)
        waist   = get_float(self.inp_waist,  "Waist",            40,  200)
        hr      = get_float(self.inp_hr,     "Heart rate",       30,  150)
        egfr    = get_float(self.inp_egfr,   "eGFR",              1,  200)
        pack    = get_float(self.inp_pack,   "Pack-years",        0, 9999)
        alcohol = get_float(self.inp_alcohol,"Alcohol",           0, 9999)
        sodium  = get_float(self.inp_sodium, "Sodium",            0, 15000)

        if sbp and dbp and dbp >= sbp:
            errors.append("Diastolic BP must be less than Systolic BP.")

        if errors:
            QMessageBox.warning(self, "Validation Error",
                                "\n".join(f"• {e}" for e in errors))
            return

        today = str(date.today())
        pid   = self.patient["id"] if self.patient else None

        patient = {
            "id":            pid,
            "name":          name,
            "dob":           dob,
            "sex":           self.sex_combo.currentText(),
            "age":           int(age),
            "sbp":           int(sbp),
            "dbp":           int(dbp),
            "waist_cm":      waist,
            "heart_rate":    int(hr),
            "egfr":          egfr,
            "pack_years":    pack,
            "alcohol_g_day": alcohol,
            "sodium_mg_day": sodium,
            "notes":         self.inp_notes.toPlainText().strip(),
            "created":       self.patient["created"] if self.patient else today,
            "last_updated":  today,
        }

        self.saved.emit(patient)

# ── Patient Detail View ───────────────────────────────────────────────────
class PatientDetail(QWidget):
    edit_requested   = pyqtSignal(dict)
    delete_requested = pyqtSignal(str)
    export_requested = pyqtSignal(str)
    back_requested   = pyqtSignal()

    def __init__(self, patient, parent=None):
        super().__init__(parent)
        self.patient = patient
        self._build()

    def _build(self):
        p   = self.patient
        pp  = p['sbp'] - p['dbp']
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        # Header
        hdr = QWidget()
        hl  = QHBoxLayout(hdr)
        hl.setContentsMargins(0, 0, 0, 0)

        back_btn = QPushButton("← Back")
        back_btn.setObjectName("secondary")
        back_btn.clicked.connect(self.back_requested.emit)
        hl.addWidget(back_btn)
        hl.addStretch()

        pid_lbl = lbl(p['id'], "pid_badge")
        hl.addWidget(pid_lbl)
        root.addWidget(hdr)

        root.addWidget(lbl(p['name'], "heading"))
        root.addWidget(lbl(
            f"{p['sex']}  ·  DOB {p['dob']}  ·  Age {p['age']}",
            "subheading"))

        # Stats row
        stats_w = QWidget()
        sl      = QHBoxLayout(stats_w)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(12)

        stats = [
            (str(p['sbp']),         "SBP mmHg",      "#ef4444"),
            (str(p['dbp']),         "DBP mmHg",      "#f97316"),
            (str(pp),               "PP mmHg",       "#a855f7"),
            (str(p['egfr']),        "eGFR",          "#22d3ee"),
            (str(p['heart_rate']), "Heart Rate bpm", "#4ade80"),
        ]

        for val, label, color in stats:
            c  = card()
            cl = QVBoxLayout(c)
            cl.setContentsMargins(16, 14, 16, 14)
            cl.setSpacing(4)
            vl = QLabel(val)
            vl.setStyleSheet(
                f"font-size:26px;color:{color};"
                f"font-family:'Georgia',serif;")
            ll = lbl(label, "stat_label")
            cl.addWidget(vl)
            cl.addWidget(ll)
            sl.addWidget(c)

        root.addWidget(stats_w)

        # Details card
        det = card()
        dl  = QGridLayout(det)
        dl.setContentsMargins(24, 20, 24, 20)
        dl.setSpacing(12)

        details = [
            ("Waist circumference", f"{p['waist_cm']} cm"),
            ("Pack-years",          f"{p['pack_years']}"),
            ("Alcohol",             f"{p['alcohol_g_day']} g/day"),
            ("Dietary sodium",      f"{p['sodium_mg_day']} mg/day"),
            ("Created",             p['created']),
            ("Last updated",        p['last_updated']),
        ]

        for i, (k, v) in enumerate(details):
            row, col = divmod(i, 2)
            w2 = QWidget()
            wl = QVBoxLayout(w2)
            wl.setContentsMargins(0, 0, 0, 0)
            wl.setSpacing(2)
            wl.addWidget(lbl(k, "field_label"))
            wl.addWidget(lbl(v))
            dl.addWidget(w2, row, col)

        if p.get('notes'):
            nl = QWidget()
            nll = QVBoxLayout(nl)
            nll.setContentsMargins(0, 0, 0, 0)
            nll.setSpacing(2)
            nll.addWidget(lbl("Clinical Notes", "field_label"))
            nll.addWidget(lbl(p['notes']))
            dl.addWidget(nl, len(details)//2 + 1, 0, 1, 2)

        root.addWidget(det)
        root.addStretch()

        # Action buttons
        btn_row = QWidget()
        br      = QHBoxLayout(btn_row)
        br.setContentsMargins(0, 0, 0, 0)
        br.setSpacing(12)

        edit_btn = QPushButton("Edit Patient")
        edit_btn.setObjectName("secondary")
        edit_btn.clicked.connect(lambda: self.edit_requested.emit(self.patient))

        export_btn = QPushButton("Export JSON")
        export_btn.setObjectName("secondary")
        export_btn.clicked.connect(lambda: self.export_requested.emit(p['id']))

        del_btn = QPushButton("Delete Patient")
        del_btn.setObjectName("danger")
        del_btn.clicked.connect(lambda: self.delete_requested.emit(p['id']))

        br.addWidget(edit_btn)
        br.addWidget(export_btn)
        br.addStretch()
        br.addWidget(del_btn)
        root.addWidget(btn_row)

# ── Patient List ──────────────────────────────────────────────────────────
class PatientList(QWidget):
    patient_selected   = pyqtSignal(str)
    add_requested      = pyqtSignal()
    import_requested   = pyqtSignal()
    simulate_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        # Header
        hdr = QWidget()
        hl  = QHBoxLayout(hdr)
        hl.setContentsMargins(0, 0, 0, 0)

        left = QWidget()
        ll   = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(4)
        ll.addWidget(lbl("Patient Registry", "heading"))
        ll.addWidget(lbl("Manage and simulate patient profiles", "subheading"))
        hl.addWidget(left)
        hl.addStretch()

        import_btn = QPushButton("Import JSON")
        import_btn.setObjectName("secondary")
        import_btn.clicked.connect(self.import_requested.emit)
        hl.addWidget(import_btn)

        add_btn = QPushButton("+ New Patient")
        add_btn.setObjectName("primary")
        add_btn.clicked.connect(self.add_requested.emit)
        hl.addWidget(add_btn)

        root.addWidget(hdr)

        # Stats bar
        self.stats_bar = QWidget()
        sb = QHBoxLayout(self.stats_bar)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.setSpacing(12)

        self.total_lbl    = self._stat_card("0", "TOTAL PATIENTS", "#6366f1")
        self.htn_lbl      = self._stat_card("0", "HYPERTENSIVE", "#ef4444")
        self.highrisk_lbl = self._stat_card("0", "HIGH RISK eGFR<60", "#f97316")

        sb.addWidget(self.total_lbl)
        sb.addWidget(self.htn_lbl)
        sb.addWidget(self.highrisk_lbl)
        sb.addStretch()
        root.addWidget(self.stats_bar)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "ID", "Name", "Age", "Sex", "SBP", "DBP", "eGFR", ""
        ])
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(
            7, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(7, 80)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(False)
        self.table.doubleClicked.connect(self._on_double_click)
        root.addWidget(self.table, stretch=1)

    def _stat_card(self, value, label, color):
        c  = card()
        cl = QHBoxLayout(c)
        cl.setContentsMargins(16, 12, 16, 12)
        cl.setSpacing(12)
        vl = QLabel(value)
        vl.setStyleSheet(
            f"font-size:24px;color:{color};"
            f"font-family:'Georgia',serif;min-width:40px;")
        ll = lbl(label, "stat_label")
        cl.addWidget(vl)
        cl.addWidget(ll)
        c._value_lbl = vl
        return c

    def refresh(self, registry):
        self.table.setRowCount(0)

        total    = len(registry)
        htn      = sum(1 for p in registry.values() if p['sbp'] >= 130)
        highrisk = sum(1 for p in registry.values() if p['egfr'] < 60)

        self.total_lbl._value_lbl.setText(str(total))
        self.htn_lbl._value_lbl.setText(str(htn))
        self.highrisk_lbl._value_lbl.setText(str(highrisk))

        for row, (pid, p) in enumerate(registry.items()):
            self.table.insertRow(row)

            sbp_color = (
                "#ef4444" if p['sbp'] >= 140 else
                "#f97316" if p['sbp'] >= 130 else
                "#4ade80"
            )

            for col, (val, align, color) in enumerate([
                (p['id'],         Qt.AlignmentFlag.AlignLeft,   "#6366f1"),
                (p['name'],       Qt.AlignmentFlag.AlignLeft,   "#e2e8f0"),
                (str(p['age']),   Qt.AlignmentFlag.AlignCenter, "#e2e8f0"),
                (p['sex'],        Qt.AlignmentFlag.AlignCenter, "#e2e8f0"),
                (str(p['sbp']),   Qt.AlignmentFlag.AlignCenter, sbp_color),
                (str(p['dbp']),   Qt.AlignmentFlag.AlignCenter, "#e2e8f0"),
                (str(p['egfr']),  Qt.AlignmentFlag.AlignCenter,
                 "#ef4444" if p['egfr'] < 60 else
                 "#f97316" if p['egfr'] < 90 else "#4ade80"),
            ]):
                item = QTableWidgetItem(val)
                item.setTextAlignment(align)
                item.setForeground(QColor(color))
                self.table.setItem(row, col, item)

            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(4, 0, 4, 0)
            btn_layout.setSpacing(4)

            view_btn = QPushButton("View")
            view_btn.setObjectName("row_btn")
            view_btn.clicked.connect(
                lambda checked, p=pid: self.patient_selected.emit(p))

            sim_btn = QPushButton("Simulate →")
            sim_btn.setObjectName("row_btn")
            sim_btn.setStyleSheet("color:#4ade80;")
            sim_btn.clicked.connect(
                lambda checked, p=pid: self.simulate_requested.emit(p))

            btn_layout.addWidget(view_btn)
            btn_layout.addWidget(sim_btn)
            self.table.setCellWidget(row, 7, btn_widget)
            self.table.setColumnWidth(7, 140)
            self.table.setRowHeight(row, 44)

    def _on_double_click(self, index):
        pid_item = self.table.item(index.row(), 0)
        if pid_item:
            self.patient_selected.emit(pid_item.text())

# ── Main Window ───────────────────────────────────────────────────────────
class PatientRegistry(QMainWindow):

    simulate_requested = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.registry = load_registry()
        self._build()

    def _on_simulate_patient(self, pid):
        if pid in self.registry:
            self.simulate_requested.emit(self.registry[pid])
    def _build(self):
        self.setWindowTitle("FlowState AI — Patient Registry")
        self.setMinimumSize(1100, 720)
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
        version = QLabel("v.2  ·  Patient Registry")
        version.setStyleSheet(
            "color:#334155;font-size:11px;"
            "font-family:'Georgia',serif;letter-spacing:1px;")

        tbl.addWidget(brand)
        tbl.addSpacing(12)
        tbl.addWidget(version)
        tbl.addStretch()
        root.addWidget(topbar)

        # Content area
        content = QWidget()
        content.setStyleSheet("padding: 0px;")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(40, 32, 40, 32)

        self.stack = QStackedWidget()
        cl.addWidget(self.stack)
        root.addWidget(content)

        # Pages
        self.list_page = PatientList()
        self.list_page.patient_selected.connect(self._show_detail)
        self.list_page.add_requested.connect(self._show_add)
        self.list_page.import_requested.connect(self._import_json)
        self.list_page.simulate_requested.connect(self._on_simulate_patient)
        self.stack.addWidget(self.list_page)

        self._refresh_list()
        self.stack.setCurrentWidget(self.list_page)

    def _refresh_list(self):
        self.registry = load_registry()
        self.list_page.refresh(self.registry)

    def _show_add(self):
        form = PatientForm()
        form.saved.connect(self._on_save_new)
        form.cancelled.connect(lambda: self.stack.setCurrentWidget(self.list_page))
        self._push(form)

    def _show_detail(self, pid):
        if pid not in self.registry:
            return
        p      = self.registry[pid]
        detail = PatientDetail(p)
        detail.back_requested.connect(
            lambda: self.stack.setCurrentWidget(self.list_page))
        detail.edit_requested.connect(self._show_edit)
        detail.delete_requested.connect(self._on_delete)
        detail.export_requested.connect(self._export_patient)
        self._push(detail)

    def _show_edit(self, patient):
        form = PatientForm(patient=patient)
        form.saved.connect(self._on_save_edit)
        form.cancelled.connect(
            lambda: self._show_detail(patient['id']))
        self._push(form)

    def _push(self, widget):
        # Remove any widget beyond index 0
        while self.stack.count() > 1:
            w = self.stack.widget(1)
            self.stack.removeWidget(w)
            w.deleteLater()
        self.stack.addWidget(widget)
        self.stack.setCurrentWidget(widget)

    def _on_save_new(self, patient):
        pid           = next_id(self.registry)
        patient['id'] = pid
        self.registry[pid] = patient
        save_registry(self.registry)
        self._refresh_list()
        self.stack.setCurrentWidget(self.list_page)

    def _on_save_edit(self, patient):
        self.registry[patient['id']] = patient
        save_registry(self.registry)
        self._refresh_list()
        self._show_detail(patient['id'])

    def _on_delete(self, pid):
        name = self.registry[pid]['name']
        reply = QMessageBox.question(
            self, "Delete Patient",
            f"Delete {name} ({pid})?\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            del self.registry[pid]
            save_registry(self.registry)
            self._refresh_list()
            self.stack.setCurrentWidget(self.list_page)

    def _export_patient(self, pid):
        p        = self.registry[pid]
        filename = f"patients/{pid}_{p['name'].replace(' ', '_')}.json"
        path, _  = QFileDialog.getSaveFileName(
            self, "Export Patient", filename, "JSON Files (*.json)")
        if path:
            with open(path, "w") as f:
                json.dump(p, f, indent=2)
            QMessageBox.information(
                self, "Exported",
                f"Patient exported to:\n{path}")

    def _import_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Patient", "patients/", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "r") as f:
                p = json.load(f)
            required = ["name","dob","sex","age","sbp","dbp",
                        "waist_cm","heart_rate","egfr",
                        "pack_years","alcohol_g_day","sodium_mg_day"]
            missing = [k for k in required if k not in p]
            if missing:
                QMessageBox.warning(
                    self, "Invalid File",
                    f"Missing fields: {', '.join(missing)}")
                return
            pid           = next_id(self.registry)
            p['id']       = pid
            p['created']  = p.get('created', str(date.today()))
            p['last_updated'] = str(date.today())
            self.registry[pid] = p
            save_registry(self.registry)
            self._refresh_list()
            QMessageBox.information(
                self, "Imported",
                f"Patient imported as {pid}.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not import file:\n{e}")

# ── Entry point ───────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Georgia", 10))
    win = PatientRegistry()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()