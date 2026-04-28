# aria.py
#
# ARIA — Adaptive Risk Intelligence Assistant
# FlowState AI v.2 companion chatbot

import json
import re
import time
import requests
import os
from pathlib import Path
from dotenv import load_dotenv

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QFrame, QScrollArea,
    QComboBox, QSizePolicy, QStackedWidget, QTextBrowser
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

# ── Config (Secure: Environment Variables) ───────────────────────────────────
CONFIG_PATH = Path("aria_config.json")

def load_config():
    # Load from .env file if it exists
    load_dotenv()
    
    # Try environment variables first (secure)
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    provider = os.getenv("ARIA_PROVIDER", "gemini")
    
    # Fallback to legacy JSON config for backward compatibility
    if not gemini_key and not groq_key and CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                legacy_cfg = json.load(f)
                gemini_key = legacy_cfg.get("gemini_api_key", "")
                groq_key = legacy_cfg.get("groq_api_key", "")
                provider = legacy_cfg.get("provider", "gemini")
        except Exception:
            pass
    
    return {
        "gemini_api_key": gemini_key,
        "groq_api_key": groq_key,
        "provider": provider
    }

def save_config(cfg):
    # Note: Saving to environment variables is not recommended
    # This function kept for backward compatibility
    # For new setup, use .env file or system environment variables
    
    # Save to JSON as fallback (will be deprecated)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)

# ── Markdown to HTML converter ────────────────────────────────────────────────
def markdown_to_html(text):
    """Markdown to HTML converter with proper formatting for ARIA responses."""
    # Escape HTML first
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    # Code blocks ```text```
    def code_block_replacer(match):
        code = match.group(1)
        return f'<pre style="background:#1e1e2e;padding:12px;border-radius:8px;margin:12px 0;overflow-x:auto;"><code style="font-family:Consolas,Monaco,monospace;font-size:13px;color:#e2e8f0;white-space:pre-wrap;">{code}</code></pre>'
    text = re.sub(r'```([\s\S]*?)```', code_block_replacer, text)
    
    # Inline code `text`
    text = re.sub(r'`([^`]+)`', r'<code style="background:#374151;padding:2px 6px;border-radius:4px;font-family:Consolas,Monaco,monospace;font-size:13px;color:#e2e8f0;">\1</code>', text)
    
    # Bold: **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
    
    # Italic: *text* or _text_
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)
    text = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'<em>\1</em>', text)
    
    # Headers: # ## ###
    text = re.sub(r'^### (.+)$', r'<h3 style="color:#e2e8f0;font-size:15px;font-weight:600;margin:16px 0 8px 0;">\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h2 style="color:#e2e8f0;font-size:17px;font-weight:600;margin:18px 0 10px 0;">\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'<h1 style="color:#e2e8f0;font-size:19px;font-weight:600;margin:20px 0 12px 0;">\1</h1>', text, flags=re.MULTILINE)
    
    # Lists: - item or * item
    def list_replacer(match):
        items = match.group(0).strip().split('\n')
        list_items = []
        for item in items:
            item = re.sub(r'^[-*]\s+', '', item.strip())
            list_items.append(f'<li style="margin:4px 0;">{item}</li>')
        return f'<ul style="margin:12px 0;padding-left:20px;">{"".join(list_items)}</ul>'
    text = re.sub(r'(^[-*]\s+.+\n?)+', list_replacer, text, flags=re.MULTILINE)
    
    # Paragraphs: double newlines create paragraphs
    paragraphs = text.split('\n\n')
    html_paragraphs = []
    for para in paragraphs:
        para = para.strip()
        if para and not para.startswith('<'):
            # Single newlines within paragraphs become <br>
            para = para.replace('\n', '<br>')
            html_paragraphs.append(f'<p style="margin:8px 0;line-height:1.6;">{para}</p>')
        elif para:
            html_paragraphs.append(para)
    
    return ''.join(html_paragraphs)

# ── API worker ────────────────────────────────────────────────────────────
class APIWorker(QThread):
    finished = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, provider, api_key, messages, model):
        super().__init__()
        self.provider = provider
        self.api_key  = api_key
        self.messages = messages
        self.model    = model

    def run(self):
        try:
            if self.provider == "gemini":
                response = self._call_gemini()
            else:
                response = self._call_groq()
            self.finished.emit(response)
        except Exception as e:
            self.error.emit(str(e))

    def _call_gemini(self):
        system_prompt = ""
        conversation  = []
        for msg in self.messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                role = "model" if msg["role"] == "assistant" else "user"
                conversation.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}]
                })
        url = (f"https://generativelanguage.googleapis.com/v1beta/"
               f"models/{self.model}:generateContent?key={self.api_key}")
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": conversation,
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2000}
        }
        for attempt in range(3):
            r = requests.post(url, json=payload, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        raise Exception("Rate limit hit. Please wait a moment and try again.")

    def _call_groq(self):
        url     = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": self.messages,
                   "temperature": 0.3, "max_tokens": 2000}
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

# ── System prompt ─────────────────────────────────────────────────────────
def build_system_prompt(patient, results, current_year, intervention_name=None, baseline_results=None):
    if patient:
        sex_str  = "Male" if patient.get("sex") == 1 else "Female"
        pat_info = f"""
CURRENT PATIENT PROFILE:
  Age: {patient.get('age')} years | Sex: {sex_str}
  SBP: {patient.get('sbp')} mmHg | DBP: {patient.get('dbp')} mmHg
  Waist: {patient.get('waist_cm')} cm | Heart rate: {patient.get('heart_rate')} bpm
  eGFR: {patient.get('egfr')} ml/min/1.73m²
  Pack-years: {patient.get('pack_years')} | Alcohol: {patient.get('alcohol_g_day')} g/day
  Sodium: {patient.get('sodium_mg_day')} mg/day"""
    else:
        pat_info = "No patient loaded yet."

    if intervention_name and baseline_results and results:
        yr = min(current_year, len(results["year"]) - 1)
        traj_info = f"""
INTERVENTION SIMULATION: {intervention_name}
  Comparison of baseline vs treated trajectories

  Year {results['year'][yr]} - Treated trajectory:
    SBP: {results['sbp'][yr]} mmHg (vs baseline {baseline_results['sbp'][yr]} mmHg)
    DBP: {results['dbp'][yr]} mmHg (vs baseline {baseline_results['dbp'][yr]} mmHg)
    eGFR: {results['egfr'][yr]} ml/min (vs baseline {baseline_results['egfr'][yr]} ml/min)
    Survival: {results['survival'][yr]}% (vs baseline {baseline_results['survival'][yr]}%)

  Final outcomes:
    SBP change: {results['sbp'][-1] - baseline_results['sbp'][-1]:.1f} mmHg
    eGFR change: {results['egfr'][-1] - baseline_results['egfr'][-1]:.1f} ml/min
    Survival change: {results['survival'][-1] - baseline_results['survival'][-1]:.1f} percentage points"""
    elif results:
        yr = min(current_year, len(results["year"]) - 1)
        traj_info = f"""
SIMULATION (Year {results['year'][yr]} selected):
  SBP: {results['sbp'][yr]} mmHg | DBP: {results['dbp'][yr]} mmHg
  PP: {results['pp'][yr]} mmHg | eGFR: {results['egfr'][yr]} ml/min
  Survival: {results['survival'][yr]}%
  Baseline SBP: {results['sbp'][0]} → Final: {results['sbp'][-1]} mmHg (+{results['sbp'][-1]-results['sbp'][0]:.1f})
  eGFR decline: -{results['egfr'][0]-results['egfr'][-1]:.1f} ml/min over {results['year'][-1]} years
  Final survival: {results['survival'][-1]}%"""
    else:
        traj_info = "No simulation results yet."

    intervention_note = f"\nNOTE: An {intervention_name} intervention was applied to this patient." if intervention_name else "\nNOTE: This is an untreated baseline trajectory."

    return f"""You are ARIA — Adaptive Risk Intelligence Assistant — the AI companion for FlowState AI v.2, a patient-specific digital twin for hypertension.

IDENTITY: Knowledgeable, warm, precise clinical research assistant. Not a doctor. Educational and analytical only.

KNOWLEDGE:
- MAP = (SBP + 2×DBP)/3 | PP = SBP-DBP | Both z-scored vs NHANES 2017-2018
- MAP progression: Gradient Boosting, CV R²=0.583, trained on 130,576 NHEFS transitions
- PP progression: Franklin et al. (1999) 0.0373 z-units/year
- eGFR: Glassock & Winearls (2009) decade-specific decline rates
- Mortality: D'Agostino (2008) log-hazard — Lewington (2002) SBP, Go (2004) eGFR, Doll (2004) smoking
- Validated age range: 35-75 | Normal BP <120 | Stage 1: 130-139 | Stage 2: ≥140
- CKD Stage 3: eGFR <60 | Stage 4: <30

CURRENT PATIENT:{pat_info}

SIMULATION DATA:{traj_info}{intervention_note}

RULES:
- Never recommend medications or treatments
- Your domain: cardiovascular physiology, hypertension, kidney function, clinical research, medical statistics, and health science
- You can answer broader medical questions (e.g., "What is ACE inhibitor mechanism?", "How does eGFR relate to CKD?")
- If asked completely off-topic questions (politics, entertainment, non-medical topics), respond with exactly: "I'm designed to assist with medical and cardiovascular health questions. Please ask me something about physiology, clinical research, or this patient's simulation."
- Provide detailed, thorough answers to specific questions
- Use plain language, explain technical terms
- Always note this is an untreated trajectory when relevant
- Format responses using markdown: use **bold** for emphasis, `code` for technical terms, ### for section headers, and - for lists."""

# ── Widgets ───────────────────────────────────────────────────────────────
class MessageBubble(QFrame):
    def __init__(self, text, is_user, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(4)

        if is_user:
            self.setStyleSheet("""
                QFrame {
                    background: #312e81;
                    border-radius: 14px;
                    margin-left: 50px;
                }
            """)
            label = QLabel(text)
            label.setWordWrap(True)
            label.setFont(QFont("Segoe UI", 10))
            label.setStyleSheet(
                "color:#e0e7ff;background:transparent;border:none;")
            label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(label)
        else:
            self.setStyleSheet("""
                QFrame {
                    background: #111827;
                    border-radius: 14px;
                    margin-right: 50px;
                    border: 1px solid #1f2937;
                }
            """)
            hdr = QLabel("ARIA")
            hdr.setFont(QFont("Inter", 8, QFont.Weight.Bold))
            hdr.setStyleSheet(
                "color:#6366f1;background:transparent;"
                "border:none;letter-spacing:2px;")
            layout.addWidget(hdr)
            
            # Use QTextBrowser for markdown rendering
            text_browser = QTextBrowser()
            text_browser.setHtml(markdown_to_html(text))
            text_browser.setFont(QFont("Inter", 11))
            text_browser.setStyleSheet("""
                QTextBrowser {
                    background: transparent;
                    border: none;
                    color: #d1d5db;
                }
                QTextBrowser p {
                    margin: 8px 0;
                    line-height: 1.7;
                }
                QTextBrowser h1, QTextBrowser h2, QTextBrowser h3 {
                    color: #e2e8f0;
                    font-weight: 600;
                }
                QTextBrowser strong {
                    color: #f9fafb;
                    font-weight: 600;
                }
                QTextBrowser pre {
                    background: #1e1e2e;
                    border-radius: 8px;
                }
                QTextBrowser code {
                    background: #374151;
                    color: #e2e8f0;
                }
            """)
            text_browser.setOpenExternalLinks(False)
            text_browser.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Minimum)
            layout.addWidget(text_browser)


class StreamingBubble(QFrame):
    def __init__(self, full_text, parent=None):
        super().__init__(parent)
        self.full_text = full_text
        self.words     = full_text.split()
        self.current   = 0
        self.setStyleSheet("""
            QFrame {
                background: #111827;
                border-radius: 14px;
                margin-right: 50px;
                border: 1px solid #1f2937;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(4)

        hdr = QLabel("ARIA")
        hdr.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        hdr.setStyleSheet(
            "color:#6366f1;background:transparent;"
            "border:none;letter-spacing:2px;")
        layout.addWidget(hdr)

        self.label = QLabel("")
        self.label.setWordWrap(True)
        self.label.setFont(QFont("Inter", 10))
        self.label.setStyleSheet(
            "color:#d1d5db;background:transparent;border:none;")
        self.label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.label)
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(40)

    def _tick(self):
        if self.current >= len(self.words):
            self._timer.stop()
            self.stop_stream()
            return
        self.current += 1
        self.label.setText(" ".join(self.words[:self.current]))

    def stop_stream(self):
        self._timer.stop()
        layout = self.layout()
        layout.removeWidget(self.label)
        self.label.deleteLater()
        
        text_browser = QTextBrowser()
        text_browser.setHtml(markdown_to_html(self.full_text))
        text_browser.setFont(QFont("Inter", 11))
        text_browser.setStyleSheet("""
            QTextBrowser {
                background: transparent;
                border: none;
                color: #d1d5db;
            }
            QTextBrowser p {
                margin: 8px 0;
                line-height: 1.7;
            }
            QTextBrowser h1, QTextBrowser h2, QTextBrowser h3 {
                color: #e2e8f0;
                font-weight: 600;
            }
            QTextBrowser strong {
                color: #f9fafb;
                font-weight: 600;
            }
            QTextBrowser pre {
                background: #1e1e2e;
                border-radius: 8px;
            }
            QTextBrowser code {
                background: #374151;
                color: #e2e8f0;
            }
        """)
        text_browser.setOpenExternalLinks(False)
        text_browser.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum)
        layout.addWidget(text_browser)


class TypingIndicator(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background: #111827;
                border-radius: 14px;
                margin-right: 50px;
                border: 1px solid #1f2937;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        self.label = QLabel("ARIA  ·  ·  ·")
        self.label.setFont(QFont("Segoe UI", 10))
        self.label.setStyleSheet(
            "color:#4b5563;background:transparent;border:none;")
        layout.addWidget(self.label)
        self._dots  = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(500)

    def _tick(self):
        self._dots = (self._dots + 1) % 4
        dots = "  ·" * self._dots + "  ○" * (3 - self._dots)
        self.label.setText(f"ARIA{dots}")

    def stop(self):
        self._timer.stop()


class SetupScreen(QWidget):
    setup_complete = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 32, 28, 28)
        layout.setSpacing(14)

        title = QLabel("Set up ARIA")
        title.setFont(QFont("Segoe UI", 16))
        title.setStyleSheet("color:#e2e8f0;background:transparent;border:none;")
        layout.addWidget(title)

        sub = QLabel(
            "Enter at least one API key to activate ARIA.\n"
            "Both providers are completely free.")
        sub.setWordWrap(True)
        sub.setFont(QFont("Segoe UI", 10))
        sub.setStyleSheet("color:#6b7280;background:transparent;border:none;")
        layout.addWidget(sub)

        for attr, lbl_text, ph, link in [
            ("gem_input", "Google Gemini API Key",
             "AIza...", "aistudio.google.com"),
            ("groq_input", "Groq API Key",
             "gsk_...", "console.groq.com"),
        ]:
            lbl = QLabel(lbl_text)
            lbl.setFont(QFont("Segoe UI", 9))
            lbl.setStyleSheet(
                "color:#9ca3af;background:transparent;border:none;")
            layout.addWidget(lbl)
            inp = QLineEdit()
            inp.setPlaceholderText(ph)
            inp.setEchoMode(QLineEdit.EchoMode.Password)
            inp.setFont(QFont("Segoe UI", 10))
            inp.setStyleSheet(
                "background:#0f172a;border:1px solid #1e293b;"
                "border-radius:8px;padding:9px 12px;color:#e2e8f0;")
            layout.addWidget(inp)
            setattr(self, attr, inp)
            help_lbl = QLabel(f"Free key → {link}")
            help_lbl.setFont(QFont("Segoe UI", 9))
            help_lbl.setStyleSheet(
                "color:#6366f1;background:transparent;border:none;")
            layout.addWidget(help_lbl)

        save_btn = QPushButton("Activate ARIA")
        save_btn.setFont(QFont("Segoe UI", 10))
        save_btn.setStyleSheet(
            "background:#6366f1;color:#fff;border:none;"
            "border-radius:8px;padding:11px;")
        save_btn.clicked.connect(self._save)
        layout.addWidget(save_btn)

        self.err_lbl = QLabel("")
        self.err_lbl.setFont(QFont("Segoe UI", 9))
        self.err_lbl.setStyleSheet(
            "color:#ef4444;background:transparent;border:none;")
        layout.addWidget(self.err_lbl)
        layout.addStretch()

    def _save(self):
        gem  = self.gem_input.text().strip()
        groq = self.groq_input.text().strip()
        if not gem and not groq:
            self.err_lbl.setText("Enter at least one API key.")
            return
        cfg = load_config()
        if gem:  cfg["gemini_api_key"] = gem
        if groq: cfg["groq_api_key"]   = groq
        if "provider" not in cfg:
            cfg["provider"] = "gemini" if gem else "groq"
        save_config(cfg)
        self.setup_complete.emit()


# ── Main panel ────────────────────────────────────────────────────────────
class ARIAPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.patient_params = None
        self.results        = None
        self.current_year   = 0
        self.intervention_name = None
        self.baseline_results = None
        self.chat_history   = []
        self.worker         = None
        self.typing_widget  = None
        self.config         = load_config()

        self.setMinimumWidth(280)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background:#0a0a12;")
        self._build()

    def _build(self):
        self.stack  = QStackedWidget(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.stack)

        self.setup_screen = SetupScreen()
        self.setup_screen.setup_complete.connect(self._on_setup_complete)
        self.stack.addWidget(self.setup_screen)

        self.chat_page = QWidget()
        self.chat_page.setStyleSheet("background:#0a0a12;")
        self._build_chat_page()
        self.stack.addWidget(self.chat_page)

        cfg = self.config
        if cfg.get("gemini_api_key") or cfg.get("groq_api_key"):
            self.stack.setCurrentWidget(self.chat_page)
        else:
            self.stack.setCurrentWidget(self.setup_screen)

    def _build_chat_page(self):
        layout = QVBoxLayout(self.chat_page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setFixedHeight(54)
        header.setStyleSheet(
            "background:#06060d;border-bottom:1px solid #1e1e2e;")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 0, 12, 0)

        name_lbl = QLabel("ARIA")
        name_lbl.setFont(QFont("Segoe UI", 13))
        name_lbl.setStyleSheet(
            "color:#6366f1;background:transparent;"
            "border:none;letter-spacing:3px;font-weight:bold;")
        sub_lbl = QLabel("Adaptive Risk Intelligence")
        sub_lbl.setFont(QFont("Segoe UI", 8))
        sub_lbl.setStyleSheet(
            "color:#374151;background:transparent;border:none;")

        left = QWidget()
        left.setStyleSheet("background:transparent;")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(1)
        ll.addWidget(name_lbl)
        ll.addWidget(sub_lbl)
        hl.addWidget(left)
        hl.addStretch()

        self.model_combo = QComboBox()
        self.model_combo.setFont(QFont("Segoe UI", 9))
        self.model_combo.setStyleSheet("""
            QComboBox {
                background: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 6px;
                padding: 4px 8px;
                color: #6b7280;
                min-width: 130px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #0f172a;
                color: #e2e8f0;
                selection-background-color: #6366f1;
            }
        """)
        self._populate_model_combo()
        self.model_combo.currentIndexChanged.connect(self._on_model_change)
        hl.addWidget(self.model_combo)
        layout.addWidget(header)

        # Chat scroll area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setStyleSheet("background:#0a0a12;border:none;")
        self.scroll.verticalScrollBar().setStyleSheet("""
            QScrollBar:vertical {
                background:#0a0a12;width:5px;border-radius:3px;
            }
            QScrollBar::handle:vertical {
                background:#1f2937;border-radius:3px;
            }
        """)

        self.messages_widget = QWidget()
        self.messages_widget.setStyleSheet("background:#0a0a12;")
        self.messages_layout = QVBoxLayout(self.messages_widget)
        self.messages_layout.setContentsMargins(16, 20, 16, 16)
        self.messages_layout.setSpacing(14)
        self.messages_layout.addStretch()

        self.scroll.setWidget(self.messages_widget)
        layout.addWidget(self.scroll, stretch=1)

        # Welcome message
        self._add_aria_message(
            "Hello. I am ARIA, your clinical research companion for "
            "FlowState AI.\n\n"
            "Build a patient twin and I can explain their trajectory, "
            "compare values to population norms, and answer questions "
            "about how the simulation works.")

        # Quick prompts
        quick = QWidget()
        quick.setStyleSheet("background:#0a0a12;")
        ql = QHBoxLayout(quick)
        ql.setContentsMargins(16, 4, 16, 8)
        ql.setSpacing(6)
        for prompt in ["Summarise", "Trajectory", "vs Population"]:
            full = {"Summarise": "Summarise this patient",
                    "Trajectory": "Explain the trajectory",
                    "vs Population": "How does this patient compare to population norms?"}[prompt]
            btn = QPushButton(prompt)
            btn.setFont(QFont("Segoe UI", 9))
            btn.setStyleSheet("""
                QPushButton {
                    background: #0f172a;
                    color: #6366f1;
                    border: 1px solid #1e293b;
                    border-radius: 10px;
                    padding: 4px 10px;
                }
                QPushButton:hover { background: #1e293b; }
            """)
            btn.clicked.connect(
                lambda checked, p=full: self._send_message(p))
            ql.addWidget(btn)
        layout.addWidget(quick)

        # Input row
        input_row = QWidget()
        input_row.setStyleSheet(
            "background:#06060d;border-top:1px solid #1e1e2e;")
        irl = QHBoxLayout(input_row)
        irl.setContentsMargins(12, 10, 12, 12)
        irl.setSpacing(8)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask ARIA anything…")
        self.input_field.setFont(QFont("Segoe UI", 10))
        self.input_field.setStyleSheet("""
            QLineEdit {
                background: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 18px;
                padding: 9px 16px;
                color: #e2e8f0;
            }
            QLineEdit:focus { border: 1px solid #6366f1; }
        """)
        self.input_field.returnPressed.connect(self._on_send)
        irl.addWidget(self.input_field, stretch=1)

        send_btn = QPushButton("↑")
        send_btn.setFixedSize(36, 36)
        send_btn.setFont(QFont("Segoe UI", 14))
        send_btn.setStyleSheet("""
            QPushButton {
                background: #6366f1;
                color: white;
                border: none;
                border-radius: 18px;
            }
            QPushButton:hover { background: #818cf8; }
            QPushButton:disabled { background: #1e293b; }
        """)
        send_btn.clicked.connect(self._on_send)
        self.send_btn = send_btn
        irl.addWidget(send_btn)
        layout.addWidget(input_row)

    def _populate_model_combo(self):
        cfg = self.config
        self.model_combo.clear()
        self._model_map = {}
        if cfg.get("gemini_api_key"):
            self.model_combo.addItem("Gemini 2.5 Flash")
            self._model_map["Gemini 2.5 Flash"] = ("gemini", "gemini-2.5-flash")
        if cfg.get("groq_api_key"):
            self.model_combo.addItem("Llama 3.3 70B")
            self._model_map["Llama 3.3 70B"] = ("groq", "llama-3.3-70b-versatile")
        saved = cfg.get("provider", "groq")
        for i in range(self.model_combo.count()):
            provider, _ = self._model_map.get(
                self.model_combo.itemText(i), ("", ""))
            if provider == saved:
                self.model_combo.setCurrentIndex(i)
                break

    def _on_model_change(self, idx):
        name = self.model_combo.currentText()
        if name in self._model_map:
            provider, _ = self._model_map[name]
            cfg = load_config()
            cfg["provider"] = provider
            save_config(cfg)
            self.config = cfg

    def _on_setup_complete(self):
        self.config = load_config()
        self._populate_model_combo()
        self.stack.setCurrentWidget(self.chat_page)

    def set_context(self, patient_params, results, current_year=0, intervention_name=None, baseline_results=None):
        self.patient_params = patient_params
        self.results        = results
        self.current_year   = current_year
        self.intervention_name = intervention_name
        self.baseline_results = baseline_results

    def _on_send(self):
        text = self.input_field.text().strip()
        if not text or not self.input_field.isEnabled():
            return
        self.input_field.clear()
        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)
        self._send_message(text)

    def _send_message(self, text):
        self._add_user_message(text)
        self.chat_history.append({"role": "user", "content": text})

        system_prompt = build_system_prompt(
            self.patient_params, self.results, self.current_year,
            self.intervention_name, self.baseline_results)
        messages = [{"role": "system", "content": system_prompt}]
        messages += self.chat_history[-20:]

        self._show_typing()

        name = self.model_combo.currentText()
        if name not in self._model_map:
            self._hide_typing()
            self._add_aria_message(
                "No API key configured. Please check aria_config.json.")
            self.input_field.setEnabled(True)
            self.send_btn.setEnabled(True)
            return

        provider, model_id = self._model_map[name]
        api_key = (self.config.get("gemini_api_key")
                   if provider == "gemini"
                   else self.config.get("groq_api_key"))

        self.worker = APIWorker(provider, api_key, messages, model_id)
        self.worker.finished.connect(self._on_response)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_response(self, text):
        self._hide_typing()
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self._add_aria_message(text)
        self.chat_history.append({"role": "assistant", "content": text})

    def _on_error(self, error):
        self._hide_typing()
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self._add_aria_message(
            f"Error: {error}\n\nCheck your API key and internet connection.")

    def _add_user_message(self, text):
        bubble = MessageBubble(text, is_user=True)
        self.messages_layout.insertWidget(
            self.messages_layout.count() - 1, bubble)
        self._scroll_to_bottom()

    def _add_aria_message(self, text):
        bubble = MessageBubble(text, is_user=False)
        self.messages_layout.insertWidget(
            self.messages_layout.count() - 1, bubble)
        self._scroll_to_bottom()

    def _show_typing(self):
        self.typing_widget = TypingIndicator()
        self.messages_layout.insertWidget(
            self.messages_layout.count() - 1, self.typing_widget)
        self._scroll_to_bottom()

    def _hide_typing(self):
        if self.typing_widget:
            self.typing_widget.stop()
            self.messages_layout.removeWidget(self.typing_widget)
            self.typing_widget.deleteLater()
            self.typing_widget = None

    def _scroll_to_bottom(self):
        QTimer.singleShot(
            50,
            lambda: self.scroll.verticalScrollBar().setValue(
                self.scroll.verticalScrollBar().maximum()))