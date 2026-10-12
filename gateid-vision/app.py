"""
GateID Vision — Streamlit Dashboard Application
AI-Based Hostel Gate ID Card Compliance Monitoring System
HackZen 2.0 Hackathon Project — Premium AI Surveillance Operations UI
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import os
import shutil
import sys
import tempfile
import time
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config
from src.cv_compat import HAS_OPENCV
from src.event_capture import CapturedEvent
from src.metrics import calculate_ground_truth_metrics
from src.report_generator import ReportGenerator
from src.video_processor import ProcessingResult, VideoProcessor
from src.dashboard_utils import check_ultralytics_available, get_file_content_hash

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GATEID VISION | Campus Ingress Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Design Theme: Premium Midnight Navy & Observability Palette
# -----------------------------------------------------------------------------
DARK_THEME_CSS = """
<style>
    /* Google Fonts & Root Configuration */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

    :root {
        --bg-primary: #0B1020;
        --bg-sidebar: #0D1425;
        --bg-card: #131C2F;
        --bg-card-hover: #18233C;
        --border-subtle: #26344B;
        --border-focus: #4F8CFF;
        --accent-blue: #4F8CFF;
        --accent-cyan: #27D7E7;
        --status-success: #35D39A;
        --status-warning: #FFBE55;
        --status-error: #FF6B78;
        --status-purple: #C099FF;
        --text-primary: #F1F5FC;
        --text-secondary: #9AAAC2;
        --text-muted: #6C7D93;
    }

    /* Global App Background & Base Typography */
    .stApp {
        background-color: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Sidebar Base Styling */
    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar);
        border-right: 1px solid var(--border-subtle);
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }
    div[data-testid="stSidebarNav"] {
        display: none;
    }

    /* Top Brand Header in Sidebar */
    .brand-container {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 4px 6px 16px 6px;
        border-bottom: 1px solid var(--border-subtle);
        margin-bottom: 16px;
    }
    .brand-icon {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #4F8CFF 0%, #27D7E7 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        box-shadow: 0 4px 14px rgba(79, 140, 255, 0.35);
        flex-shrink: 0;
    }
    .brand-title {
        font-size: 17px;
        font-weight: 800;
        letter-spacing: 0.8px;
        color: var(--text-primary);
        margin: 0;
        line-height: 1.15;
    }
    .brand-subtitle {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: var(--accent-cyan);
        margin: 2px 0 0 0;
        font-weight: 600;
    }

    /* Section Labels in Sidebar */
    .sidebar-section-label {
        font-size: 10.5px;
        font-weight: 700;
        letter-spacing: 1.2px;
        color: var(--text-muted);
        text-transform: uppercase;
        margin: 18px 0 8px 4px;
    }

    /* Sidebar Status Chips */
    .status-badge-chip {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(19, 28, 47, 0.8);
        border: 1px solid var(--border-subtle);
        border-radius: 6px;
        padding: 6px 10px;
        margin-bottom: 6px;
        font-size: 12px;
    }
    .status-badge-chip span.label {
        color: var(--text-secondary);
        font-weight: 500;
    }
    .status-badge-chip span.val {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 11.5px;
    }

    /* Page Header Banner */
    .page-header-card {
        background: linear-gradient(135deg, #131C2F 0%, #172238 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 24px;
        position: relative;
        box-shadow: 0 4px 20px rgba(11, 16, 32, 0.5);
    }
    .page-header-category {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        color: var(--accent-blue);
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .page-header-title {
        font-size: 26px;
        font-weight: 800;
        color: var(--text-primary);
        margin: 0 0 6px 0;
        letter-spacing: -0.4px;
    }
    .page-header-desc {
        font-size: 13.5px;
        color: var(--text-secondary);
        margin: 0;
        max-width: 85%;
        line-height: 1.45;
    }
    .live-indicator-pill {
        position: absolute;
        top: 22px;
        right: 26px;
        background: rgba(53, 211, 154, 0.12);
        border: 1px solid rgba(53, 211, 154, 0.4);
        color: var(--status-success);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.8px;
        padding: 4px 10px;
        border-radius: 20px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .live-dot {
        width: 7px;
        height: 7px;
        background-color: var(--status-success);
        border-radius: 50%;
        box-shadow: 0 0 8px var(--status-success);
    }

    /* System Notice Callout */
    .system-notice-card {
        background-color: rgba(79, 140, 255, 0.06);
        border-left: 3px solid var(--accent-blue);
        border-top: 1px solid rgba(79, 140, 255, 0.15);
        border-right: 1px solid rgba(79, 140, 255, 0.15);
        border-bottom: 1px solid rgba(79, 140, 255, 0.15);
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 12.5px;
        color: #C2D1E5;
        margin-top: 14px;
    }

    /* Warning Callout Box */
    .system-warning-box {
        background-color: rgba(255, 190, 85, 0.08);
        border: 1px solid rgba(255, 190, 85, 0.3);
        border-left: 4px solid var(--status-warning);
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 13px;
        color: #FFE1A8;
        margin-bottom: 22px;
    }

    /* KPI Summary Metric Cards */
    .kpi-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 16px 18px;
        position: relative;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }
    .kpi-card:hover {
        border-color: var(--border-focus);
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(11, 16, 32, 0.7);
    }
    .kpi-top-bar {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        border-top-left-radius: 10px;
        border-top-right-radius: 10px;
    }
    .kpi-title {
        color: var(--text-secondary);
        font-size: 11.5px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .kpi-value {
        font-size: 30px;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.5px;
        line-height: 1.1;
        margin-bottom: 4px;
        font-variant-numeric: tabular-nums;
    }
    .kpi-subtitle {
        font-size: 12px;
        color: var(--text-muted);
    }

    /* Status Pills */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 9px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 11.5px;
        letter-spacing: 0.4px;
        text-transform: uppercase;
    }
    .status-pill-visible {
        background: rgba(53, 211, 154, 0.15);
        color: var(--status-success);
        border: 1px solid rgba(53, 211, 154, 0.4);
    }
    .status-pill-verify {
        background: rgba(255, 190, 85, 0.15);
        color: var(--status-warning);
        border: 1px solid rgba(255, 190, 85, 0.4);
    }
    .status-pill-uncertain {
        background: rgba(192, 153, 255, 0.15);
        color: var(--status-purple);
        border: 1px solid rgba(192, 153, 255, 0.4);
    }

    /* Event & Snapshot Container Cards */
    .event-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
        transition: border-color 0.15s ease;
    }
    .event-card:hover {
        border-color: #384A67;
    }

    .snapshot-frame {
        background: var(--bg-sidebar);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 8px;
        margin-bottom: 12px;
    }

    /* Telemetry Panel Box */
    .telemetry-panel {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 18px;
    }

    /* Streamlit Control Overrides */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4F8CFF 0%, #27D7E7 100%) !important;
        border-radius: 4px;
    }

    /* Standard Interactive Buttons */
    div.stButton > button {
        background-color: #162137 !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 0.45rem 0.85rem !important;
        transition: all 0.15s ease !important;
    }
    div.stButton > button:hover {
        background-color: #1F2E4C !important;
        border-color: var(--accent-blue) !important;
        color: var(--text-primary) !important;
        transform: translateY(-1px);
    }

    /* Primary Buttons (Navigation Active / Primary Action) */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4F8CFF 0%, #296CE3 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #4F8CFF !important;
        box-shadow: 0 2px 10px rgba(79, 140, 255, 0.3) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #6299FF 0%, #3A7BF0 100%) !important;
        border-color: var(--accent-cyan) !important;
    }

    /* Download Buttons (High Contrast Emerald / Cyan Theme) */
    div.stDownloadButton > button {
        background: linear-gradient(135deg, #1C744F 0%, #155B3E 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--status-success) !important;
        font-weight: 700 !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        box-shadow: 0 2px 10px rgba(53, 211, 154, 0.25) !important;
        transition: all 0.15s ease !important;
    }
    div.stDownloadButton > button:hover {
        background: linear-gradient(135deg, #248B5E 0%, #1A6C49 100%) !important;
        border-color: var(--accent-cyan) !important;
        color: #FFFFFF !important;
        transform: translateY(-1px);
    }
    div.stDownloadButton > button:disabled {
        background: #172132 !important;
        color: #6C7D93 !important;
        border-color: var(--border-subtle) !important;
        box-shadow: none !important;
    }

    /* Dataframe Table Container */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        background-color: var(--bg-sidebar);
    }

    /* Custom Input and Select Widgets */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: var(--bg-sidebar) !important;
        border-color: var(--border-subtle) !important;
        color: var(--text-primary) !important;
    }
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within {
        border-color: var(--accent-blue) !important;
    }

    /* File Uploader Container */
    [data-testid="stFileUploader"] {
        background-color: var(--bg-card);
        border: 1px dashed var(--border-subtle);
        border-radius: 10px;
        padding: 12px;
        transition: border-color 0.2s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent-blue);
    }
</style>
"""
st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Resource Caching & Helpers (Backend Preserved)
# -----------------------------------------------------------------------------
@st.cache_resource
def get_cached_config() -> Dict[str, Any]:
    """Loads system configuration from config.yaml."""
    cfg_path = PROJECT_ROOT / "config.yaml"
    return load_config(str(cfg_path))


@st.cache_resource
def get_cached_video_processor() -> VideoProcessor:
    """Initializes and caches the VideoProcessor pipeline."""
    config = get_cached_config()
    return VideoProcessor(config=config)


# -----------------------------------------------------------------------------
# Session State Initialization (All Variables Preserved)
# -----------------------------------------------------------------------------
def init_session_state() -> None:
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"
    if "processed_result" not in st.session_state:
        st.session_state.processed_result = None
    if "last_processed_hash" not in st.session_state:
        st.session_state.last_processed_hash = None
    if "reviewed_cases" not in st.session_state:
        st.session_state.reviewed_cases = {}  # event_id -> review_status
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False


init_session_state()


# -----------------------------------------------------------------------------
# Top Header Banner (Dynamic per Page)
# -----------------------------------------------------------------------------
def render_header(current_page: str) -> None:
    """Renders a modern surveillance operations header matching the active page."""
    page_meta = {
        "Dashboard": {
            "title": "📊 Operational Dashboard",
            "category": "SURVEILLANCE OPERATIONS // REAL-TIME OVERVIEW",
            "desc": "Live ingress telemetry, visitor compliance statistics, and entrance verification overview.",
        },
        "Video Monitoring": {
            "title": "📹 Ingress Video Monitoring",
            "category": "GATE CAMERA INGESTION // CV PIPELINE",
            "desc": "Automated person tracking, anatomical ID badge detection, and real-time frame telemetry.",
        },
        "Verification Queue": {
            "title": "🔍 Security Verification Queue",
            "category": "OFFICER AUDIT TERMINAL // DECISION SUPPORT",
            "desc": "Human-in-the-loop inspection console for flagged unbadged entrants and ambiguous detections.",
        },
        "Event History": {
            "title": "📜 Ingress Event Audit Log",
            "category": "HISTORICAL TELEMETRY // INCIDENT ARCHIVE",
            "desc": "Chronological audit trail of captured ingress events, tracking histories, and visual evidence.",
        },
        "Reports": {
            "title": "📑 Compliance & Governance Reports",
            "category": "OPERATIONAL AUDIT // DATA EXPORT",
            "desc": "Aggregated ingress metrics, ground-truth evaluation status, and downloadable audit reports.",
        },
        "System Status": {
            "title": "⚙️ System Diagnostics & Runtime Health",
            "category": "HARDWARE & INFERENCE PLATFORM // STATUS",
            "desc": "Local compute environment, neural network weights status, and file storage integrity.",
        },
    }

    meta = page_meta.get(current_page, page_meta["Dashboard"])
    has_active_video = st.session_state.processed_result is not None
    status_label = "SESSION ACTIVE" if has_active_video else "MONITORING READY"
    status_color = "#35D39A" if has_active_video else "#4F8CFF"

    st.markdown(
        f"""
        <div class="page-header-card">
            <div class="live-indicator-pill" style="color:{status_color}; border-color:{status_color}66; background:{status_color}1a;">
                <div class="live-dot" style="background-color:{status_color}; box-shadow:0 0 8px {status_color};"></div>
                {status_label}
            </div>
            <div class="page-header-category">
                <span>🛡️ GATEID VISION</span> &nbsp;•&nbsp; <span>{meta['category']}</span>
            </div>
            <h1 class="page-header-title">{meta['title']}</h1>
            <p class="page-header-desc">{meta['desc']}</p>
            <div class="system-notice-card">
                ℹ️ <strong>Decision Support Notice:</strong> GateID Vision assists hostel security personnel by highlighting entrants that require manual verification. It does not perform facial recognition, biometric identity extraction, or card forgery verification.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Model Status Warning Banner
# -----------------------------------------------------------------------------
def render_model_warning_if_needed(processor: VideoProcessor) -> None:
    """Renders a styled notice if operating in computer-vision fallback mode."""
    if not processor.id_detector.is_available:
        has_vision = getattr(processor.id_detector, "has_vision_fallback", False)
        if has_vision:
            st.markdown(
                """
                <div class="system-warning-box">
                    ℹ️ <strong>OpenCV Torso Badge & Lanyard Fallback Active:</strong>
                    Custom YOLO weights were not found at <code>models/id_card_detector.pt</code>.
                    Operating in <strong>High-Precision Computer Vision Torso Mode</strong>: detecting chest badges and lanyards, classifying verified entrants, and flagging unbadged cases for security officer review.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="system-warning-box" style="border-left-color:#FF6B78; color:#FFA8B0;">
                    ⚠️ <strong>ID-Card Detector Offline:</strong>
                    Model weights missing at <code>models/id_card_detector.pt</code> and fallback is disabled.
                </div>
                """,
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------------------
# Sidebar Navigation Panel
# -----------------------------------------------------------------------------
def render_sidebar(processor: VideoProcessor) -> str:
    """Renders the dark enterprise navigation sidebar with system health pills."""
    st.sidebar.markdown(
        """
        <div class="brand-container">
            <div class="brand-icon">🛡️</div>
            <div>
                <div class="brand-title">GATEID VISION</div>
                <div class="brand-subtitle">AI Security Operations</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('<div class="sidebar-section-label">// SURVEILLANCE NAVIGATION</div>', unsafe_allow_html=True)
    pages = [
        ("📊 Operational Dashboard", "Dashboard"),
        ("📹 Video Monitoring", "Video Monitoring"),
        ("🔍 Verification Queue", "Verification Queue"),
        ("📜 Event History", "Event History"),
        ("📑 Compliance Reports", "Reports"),
        ("⚙️ System Diagnostics", "System Status"),
    ]

    for label, key in pages:
        is_active = (st.session_state.current_page == key)
        button_type = "primary" if is_active else "secondary"
        if st.sidebar.button(label, key=f"nav_{key}", use_container_width=True, type=button_type):
            st.session_state.current_page = key
            st.rerun()

    # Active Session Summary (if a video has been analyzed)
    if st.session_state.processed_result is not None:
        res = st.session_state.processed_result
        st.sidebar.markdown('<div class="sidebar-section-label">// ACTIVE SESSION DATA</div>', unsafe_allow_html=True)
        st.sidebar.markdown(
            f"""
            <div style="background:#131C2F; border:1px solid #26344B; border-radius:8px; padding:10px 12px; margin-bottom:10px;">
                <div style="font-size:11px; color:#9AAAC2; text-transform:uppercase;">Ingested Source</div>
                <div style="font-size:12.5px; font-weight:700; color:#F1F5FC; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{Path(res.video_path).name}</div>
                <div style="display:flex; justify-content:space-between; margin-top:8px; font-size:11.5px;">
                    <span style="color:#9AAAC2;">Entrants: <strong style="color:#F1F5FC;">{res.metrics.unique_tracked_people}</strong></span>
                    <span style="color:#9AAAC2;">Events: <strong style="color:#27D7E7;">{res.metrics.total_events_captured}</strong></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.sidebar.button("🗑️ Clear Active Result", key="sidebar_clear_btn", use_container_width=True):
            st.session_state.processed_result = None
            st.session_state.last_processed_hash = None
            st.rerun()

    # System Status Indicators in Sidebar
    st.sidebar.markdown('<div class="sidebar-section-label">// ENGINE TELEMETRY</div>', unsafe_allow_html=True)
    model_ok = processor.id_detector.is_available
    cv_ok = HAS_OPENCV
    ultra_ok = check_ultralytics_available()

    st.sidebar.markdown(
        f"""
        <div class="status-badge-chip">
            <span class="label">OpenCV Engine</span>
            <span class="val" style="color:{'#35D39A' if cv_ok else '#FFBE55'};">{'🟢 Native' if cv_ok else '🟡 Mock Shim'}</span>
        </div>
        <div class="status-badge-chip">
            <span class="label">Ultralytics YOLO</span>
            <span class="val" style="color:{'#35D39A' if ultra_ok else '#FFBE55'};">{'🟢 Loaded' if ultra_ok else '🟡 Fallback'}</span>
        </div>
        <div class="status-badge-chip">
            <span class="label">Card Detector</span>
            <span class="val" style="color:{'#35D39A' if model_ok else '#27D7E7'};">{'🟢 Trained Model' if model_ok else '🔵 CV Torso Fallback'}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <div style="margin-top:24px; padding-top:12px; border-top:1px solid #26344B; text-align:center;">
            <div style="font-size:11px; font-weight:700; color:#6C7D93; letter-spacing:0.5px;">GATEID VISION v1.0.0</div>
            <div style="font-size:10.5px; color:#4E5D73; margin-top:2px;">HackZen 2.0 • Security Operations</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return st.session_state.current_page


# -----------------------------------------------------------------------------
# 1. Dashboard Overview Page
# -----------------------------------------------------------------------------
def render_dashboard_page(processor: VideoProcessor) -> None:
    render_model_warning_if_needed(processor)

    res: Optional[ProcessingResult] = st.session_state.processed_result

    # 6-Column High-Impact KPI Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        val = str(res.metrics.unique_tracked_people) if res else "0"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-top-bar" style="background:#4F8CFF;"></div>
                <div class="kpi-title"><span>Tracked Entrants</span> <span>👥</span></div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-subtitle">Unique Visitors</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        val = str(res.metrics.id_visible_count) if res else "0"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-top-bar" style="background:#35D39A;"></div>
                <div class="kpi-title"><span>ID Visible</span> <span>🛡️</span></div>
                <div class="kpi-value" style="color:#35D39A;">{val}</div>
                <div class="kpi-subtitle">Badges Confirmed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        val = str(res.metrics.verification_required_count) if res else "0"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-top-bar" style="background:#FFBE55;"></div>
                <div class="kpi-title"><span>Needs Verify</span> <span>⚠️</span></div>
                <div class="kpi-value" style="color:#FFBE55;">{val}</div>
                <div class="kpi-subtitle">Security Prompts</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        val = str(res.metrics.uncertain_count) if res else "0"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-top-bar" style="background:#C099FF;"></div>
                <div class="kpi-title"><span>Uncertain Cases</span> <span>🔍</span></div>
                <div class="kpi-value" style="color:#C099FF;">{val}</div>
                <div class="kpi-subtitle">Occluded / Blurred</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c5:
        val = f"{res.elapsed_time_sec:.1f}s" if res else "0.0s"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-top-bar" style="background:#27D7E7;"></div>
                <div class="kpi-title"><span>Process Time</span> <span>⏱️</span></div>
                <div class="kpi-value" style="color:#27D7E7;">{val}</div>
                <div class="kpi-subtitle">Video Processing</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c6:
        val = f"{res.metrics.approx_fps:.1f}" if res and res.metrics.approx_fps > 0 else "0.0"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-top-bar" style="background:#4F8CFF;"></div>
                <div class="kpi-title"><span>Throughput</span> <span>⚡</span></div>
                <div class="kpi-value">{val} <span style="font-size:14px; color:#9AAAC2;">FPS</span></div>
                <div class="kpi-subtitle">Real-Time Speed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if not res:
        st.markdown(
            """
            <div style="background:#131C2F; border:1px dashed #26344B; border-radius:12px; padding:32px 24px; text-align:center; margin-top:8px;">
                <div style="font-size:36px; margin-bottom:12px;">📹</div>
                <h3 style="color:#F1F5FC; margin:0 0 8px 0; font-size:18px;">No Entrance Video Analyzed in Current Session</h3>
                <p style="color:#9AAAC2; font-size:13.5px; max-width:540px; margin:0 auto 20px auto;">
                    Upload a recorded gate entry recording in <strong>Video Monitoring</strong> to activate the computer vision pipeline, detect crowd entrants, and monitor student ID badges.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_c1, col_c2, col_c3 = st.columns([1, 1, 1])
        with col_c2:
            if st.button("🚀 Ingest Video in Monitoring Panel", type="primary", use_container_width=True):
                st.session_state.current_page = "Video Monitoring"
                st.rerun()
        return

    # Charts and Event Breakdown
    col_chart, col_events = st.columns([1, 1])

    with col_chart:
        st.markdown("### 📈 Ingress Classification Breakdown")
        m = res.metrics
        status_data = pd.DataFrame(
            {
                "Category": ["ID Visible", "Verification Required", "Uncertain"],
                "Count": [m.id_visible_count, m.verification_required_count, m.uncertain_count],
            }
        ).set_index("Category")

        if m.total_events_captured > 0:
            st.bar_chart(status_data, color="#4F8CFF", height=280)
        else:
            st.caption("No events recorded during video analysis.")

    with col_events:
        st.markdown("### 🕒 Real-Time Surveillance Event Stream")
        events = res.events
        if events:
            recent_events = events[-5:]
            for e in reversed(recent_events):
                pill_class = (
                    "status-pill-visible"
                    if e.status == "id_visible"
                    else ("status-pill-verify" if e.status == "verification_required" else "status-pill-uncertain")
                )
                st.markdown(
                    f"""
                    <div class="event-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <span class="status-pill {pill_class}">{e.status.upper()}</span>
                                <strong style="margin-left:8px; color:#F1F5FC;">Track {e.track_id}</strong>
                            </div>
                            <span style="color:#9AAAC2; font-size:12px; font-family:'JetBrains Mono', monospace;">t={e.video_timestamp}s (Frame {e.frame_number})</span>
                        </div>
                        <div style="margin-top:6px; font-size:12.5px; color:#C2D1E5;">
                            {e.reason}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No event flags were captured for this video.")

    # Recent Snapshot Previews
    st.markdown("### 📸 Captured Evidence Previews")
    snap_events = [e for e in res.events if e.snapshot_path and Path(e.snapshot_path).exists()]
    if snap_events:
        cols = st.columns(min(4, len(snap_events)))
        for idx, e in enumerate(snap_events[:4]):
            with cols[idx]:
                badge_color = "#35D39A" if e.status == "id_visible" else ("#FFBE55" if e.status == "verification_required" else "#C099FF")
                st.markdown(
                    f"""
                    <div class="snapshot-frame">
                        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                            <span style="background:{badge_color}22; color:{badge_color}; border:1px solid {badge_color}66; border-radius:4px; padding:1px 6px; font-size:11px; font-weight:700;">
                                {e.status.upper()}
                            </span>
                            <span style="font-size:11px; color:#9AAAC2;">Person {e.track_id}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(e.snapshot_path, use_container_width=True)
    else:
        st.caption("No snapshot image files available on disk for preview.")


# -----------------------------------------------------------------------------
# 2. Video Monitoring Page (Centerpiece Feature)
# -----------------------------------------------------------------------------
def render_video_monitoring_page(processor: VideoProcessor) -> None:
    render_model_warning_if_needed(processor)

    st.markdown(
        """
        <div style="background:#131C2F; border:1px solid #26344B; border-radius:10px; padding:16px 20px; margin-bottom:20px;">
            <div style="display:flex; align-items:center; gap:10px; font-weight:700; color:#F1F5FC; font-size:14px;">
                <span>📹</span> INGRESS VIDEO INGESTION CONSOLE
            </div>
            <p style="color:#9AAAC2; font-size:13px; margin:4px 0 0 0;">
                Upload a recorded campus gate or hostel entrance video file. The computer vision pipeline executes automatically upon upload.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose an entrance recording (MP4, AVI, MOV, MKV)",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload recorded video of students entering the gate.",
    )

    if uploaded_file is None:
        if st.session_state.processed_result is not None:
            st.markdown("---")
            st.markdown("### 📋 Active Video Results")
            _render_processed_video_view(st.session_state.processed_result)
        else:
            st.info("👋 Upload a recorded entrance video above to start automated compliance monitoring.")
        return

    # File details & hashing
    file_bytes = uploaded_file.getvalue()
    file_size_mb = len(file_bytes) / (1024 * 1024)
    file_hash = get_file_content_hash(file_bytes)

    # Ingress Meta Strip
    st.markdown(
        f"""
        <div style="display:flex; gap:24px; background:#0D1425; border:1px solid #26344B; border-radius:8px; padding:10px 16px; margin:12px 0 16px 0; font-size:12.5px;">
            <div><span style="color:#6C7D93;">SOURCE FILE:</span> <strong style="color:#F1F5FC; font-family:'JetBrains Mono', monospace;">{uploaded_file.name}</strong></div>
            <div><span style="color:#6C7D93;">SIZE:</span> <strong style="color:#27D7E7; font-family:'JetBrains Mono', monospace;">{file_size_mb:.2f} MB</strong></div>
            <div><span style="color:#6C7D93;">CHECKSUM:</span> <strong style="color:#9AAAC2; font-family:'JetBrains Mono', monospace;">{file_hash[:12]}...</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    already_processed = (
        st.session_state.processed_result is not None
        and st.session_state.last_processed_hash == file_hash
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        reprocess = st.button("🔄 Re-process Video", help="Force re-run the pipeline on this video")

    if not already_processed or reprocess:
        temp_dir = Path("data/staging")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_video_path = temp_dir / f"upload_{file_hash[:8]}_{uploaded_file.name}"

        with open(temp_video_path, "wb") as f:
            f.write(file_bytes)

        st.markdown(
            """
            <div style="background:#131C2F; border:1px solid #4F8CFF; border-radius:8px; padding:14px 18px; margin:16px 0;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <strong style="color:#4F8CFF; font-size:14px;">⚡ AUTOMATED PIPELINE EXECUTION IN PROGRESS</strong>
                    <span style="color:#27D7E7; font-size:12px; font-family:'JetBrains Mono', monospace;">RUNNING INFERENCE</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        progress_bar = st.progress(0.0)
        status_text = st.empty()
        preview_container = st.empty()
        metrics_container = st.empty()

        def on_progress(ratio: float, cur_frame: int, total_frames: int, frame_arr: Any, stats: Dict[str, Any]):
            progress_bar.progress(min(1.0, max(0.0, ratio)))
            status_text.markdown(f"**Processing Frame:** `{cur_frame} / {total_frames}` ({ratio*100:.1f}%)")
            metrics_container.markdown(
                f"**Active People in View:** `{stats.get('active_people', 0)}` | "
                f"**Unique Visitors Tracked:** `{stats.get('unique_people', 0)}` | "
                f"**Events Captured:** `{stats.get('total_events', 0)}`"
            )
            if frame_arr is not None and hasattr(frame_arr, "shape") and frame_arr.size > 0:
                try:
                    preview_container.image(frame_arr, channels="BGR", caption="Live Pipeline View", use_container_width=True)
                except Exception:
                    pass

        try:
            with st.spinner("Executing Computer Vision Pipeline..."):
                result = processor.process_video(
                    video_path=str(temp_video_path),
                    progress_callback=on_progress,
                )

            st.session_state.processed_result = result
            st.session_state.last_processed_hash = file_hash
            st.success("✅ Video processing and compliance monitoring completed successfully!")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Processing failed with an error: {e}")
            return
        finally:
            if temp_video_path.exists():
                try:
                    temp_video_path.unlink()
                except Exception:
                    pass

    else:
        st.markdown("### 📋 Analysis Results")
        _render_processed_video_view(st.session_state.processed_result)


def _render_processed_video_view(res: ProcessingResult) -> None:
    """Helper to render the processed video preview, summary metrics, event table, and snapshot evidence."""
    m = res.metrics

    # Top KPI Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-top-bar" style="background:#4F8CFF;"></div>
                <div class="kpi-title"><span>Tracked Entrants</span> <span>👥</span></div>
                <div class="kpi-value">{m.unique_tracked_people}</div>
                <div class="kpi-subtitle">Unique Visitors</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-top-bar" style="background:#35D39A;"></div>
                <div class="kpi-title"><span>ID Visible</span> <span>🛡️</span></div>
                <div class="kpi-value" style="color:#35D39A;">{m.id_visible_count}</div>
                <div class="kpi-subtitle">Badges Confirmed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-top-bar" style="background:#FFBE55;"></div>
                <div class="kpi-title"><span>Needs Verification</span> <span>⚠️</span></div>
                <div class="kpi-value" style="color:#FFBE55;">{m.verification_required_count}</div>
                <div class="kpi-subtitle">Manual Check Prompt</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-top-bar" style="background:#C099FF;"></div>
                <div class="kpi-title"><span>Uncertain Cases</span> <span>🔍</span></div>
                <div class="kpi-value" style="color:#C099FF;">{m.uncertain_count}</div>
                <div class="kpi-subtitle">Occluded / Blurred</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m5:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-top-bar" style="background:#27D7E7;"></div>
                <div class="kpi-title"><span>Throughput</span> <span>⚡</span></div>
                <div class="kpi-value">{m.approx_fps:.1f} <span style="font-size:14px; color:#9AAAC2;">FPS</span></div>
                <div class="kpi-subtitle">Elapsed: {res.elapsed_time_sec:.1f}s</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Video Player and Summary Row
    col_v, col_s = st.columns([3, 2])

    with col_v:
        st.markdown("#### 🎬 Annotated Video Telemetry Feed")
        if res.output_video_path and Path(res.output_video_path).exists():
            v_path = Path(res.output_video_path)
            try:
                st.video(str(v_path))
            except Exception:
                st.info("Inline video preview not supported by browser codec; download available below.")

            with open(v_path, "rb") as vf:
                st.download_button(
                    label="⬇️ Download Annotated Video (MP4)",
                    data=vf.read(),
                    file_name=v_path.name,
                    mime="video/mp4",
                    use_container_width=True,
                )
        else:
            st.warning("Annotated output video file is not available.")

    with col_s:
        st.markdown("#### 📊 Execution Summary")
        st.markdown(
            f"""
            <div class="telemetry-panel">
                <div style="font-size:12.5px; margin-bottom:8px;"><span style="color:#9AAAC2;">Input Recording:</span> <strong style="color:#F1F5FC; font-family:'JetBrains Mono', monospace;">{Path(res.video_path).name}</strong></div>
                <div style="font-size:12.5px; margin-bottom:8px;"><span style="color:#9AAAC2;">Total Frames:</span> <strong style="color:#F1F5FC;">{res.total_frames}</strong> (Processed: <strong style="color:#27D7E7;">{res.processed_frames}</strong>)</div>
                <div style="font-size:12.5px; margin-bottom:8px;"><span style="color:#9AAAC2;">Processing Throughput:</span> <strong style="color:#35D39A;">{m.approx_fps:.2f} FPS</strong></div>
                <div style="font-size:12.5px; margin-bottom:8px;"><span style="color:#9AAAC2;">Events Captured:</span> <strong style="color:#F1F5FC;">{m.total_events_captured}</strong></div>
                <div style="font-size:12.5px; margin-top:12px; padding-top:10px; border-top:1px solid #26344B; color:#C2D1E5;">
                    Compliance: <strong style="color:#35D39A;">{m.id_visible_count}</strong> confirmed, <strong style="color:#FFBE55;">{m.verification_required_count}</strong> needs review, <strong style="color:#C099FF;">{m.uncertain_count}</strong> uncertain
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if res.report_csv_path and Path(res.report_csv_path).exists():
            with open(res.report_csv_path, "rb") as cf:
                st.download_button(
                    label="📄 Download Event Report (CSV)",
                    data=cf.read(),
                    file_name=Path(res.report_csv_path).name,
                    mime="text/csv",
                    use_container_width=True,
                )

        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.button("🔍 Review Queue", use_container_width=True):
                st.session_state.current_page = "Verification Queue"
                st.rerun()
        with btn_c2:
            if st.button("📑 Detailed Reports", use_container_width=True):
                st.session_state.current_page = "Reports"
                st.rerun()

    # Detailed Event Results Table
    st.markdown("---")
    st.markdown("### 📋 Ingress Event Audit Table")
    if res.events:
        report_gen = ReportGenerator()
        df = report_gen.events_to_dataframe(res.events)

        # Filtering options
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            status_filter = st.multiselect(
                "Filter Events by Status:",
                options=list(df["status"].unique()),
                default=list(df["status"].unique()),
                key="vm_status_filter",
            )
        with col_f2:
            search_txt = st.text_input("Search Events (by Track ID or Reason):", key="vm_search_txt")

        df_view = df[df["status"].isin(status_filter)]
        if search_txt:
            df_view = df_view[
                df_view["track_id"].astype(str).str.contains(search_txt, case=False)
                | df_view["reason"].str.contains(search_txt, case=False)
            ]

        st.caption(f"Showing **{len(df_view)}** of **{len(df)}** captured ingress records")
        st.dataframe(df_view, use_container_width=True, height=280)
    else:
        st.info("No monitoring events recorded for this video.")

    # Captured Evidence Snapshot Gallery
    st.markdown("### 📸 Captured Evidence Gallery")
    snap_events = [e for e in res.events if e.snapshot_path and Path(e.snapshot_path).exists()]
    if snap_events:
        cols = st.columns(min(4, len(snap_events)))
        for idx, e in enumerate(snap_events[:8]):
            with cols[idx % 4]:
                badge_color = "#35D39A" if e.status == "id_visible" else ("#FFBE55" if e.status == "verification_required" else "#C099FF")
                st.markdown(
                    f"""
                    <div class="snapshot-frame">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <span style="background:{badge_color}22; color:{badge_color}; border:1px solid {badge_color}66; border-radius:4px; padding:2px 6px; font-size:11px; font-weight:700;">
                                {e.status.upper()}
                            </span>
                            <span style="color:#9AAAC2; font-size:11.5px; font-family:'JetBrains Mono', monospace;">Person {e.track_id} (t={e.video_timestamp}s)</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(e.snapshot_path, use_container_width=True)
                st.caption(f"_{e.reason[:55] + '...' if len(e.reason) > 55 else e.reason}_")
    else:
        st.caption("No snapshot image files available on disk for preview.")


# -----------------------------------------------------------------------------
# 3. Verification Queue Page (Officer Terminal)
# -----------------------------------------------------------------------------
def render_verification_queue_page(processor: VideoProcessor) -> None:
    render_model_warning_if_needed(processor)

    st.markdown(
        """
        <div style="background:#131C2F; border:1px solid #26344B; border-radius:10px; padding:14px 18px; margin-bottom:18px;">
            <strong style="color:#FFBE55; font-size:13.5px;">👮 HUMAN-IN-THE-LOOP INSPECTION WORKFLOW</strong>
            <p style="color:#9AAAC2; font-size:13px; margin:4px 0 0 0;">
                Observations flagged here are recommendations for physical security officer inspection,
                <strong>not automated accusations of unbadged presence or disciplinary violations</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    res: Optional[ProcessingResult] = st.session_state.processed_result
    if not res or not res.events:
        st.info("No cases currently in the verification queue. Ingest a video in **Video Monitoring** to populate cases.")
        return

    # Filter controls
    f_col1, f_col2 = st.columns([2, 1])
    with f_col1:
        status_filter = st.multiselect(
            "Filter by Status:",
            options=["verification_required", "uncertain", "id_visible"],
            default=["verification_required", "uncertain"],
        )
    with f_col2:
        search_query = st.text_input("Search by Track ID or Reason:", "")

    filtered_events = [
        e for e in res.events
        if e.status in status_filter
        and (search_query.lower() in str(e.track_id) or search_query.lower() in e.reason.lower())
    ]

    st.markdown(f"**Showing {len(filtered_events)} case(s) requiring review**")

    if not filtered_events:
        st.caption("No events matched the chosen filters.")
        return

    for evt in filtered_events:
        pill_class = (
            "status-pill-verify"
            if evt.status == "verification_required"
            else ("status-pill-uncertain" if evt.status == "uncertain" else "status-pill-visible")
        )
        reviewed_status = st.session_state.reviewed_cases.get(evt.event_id, "Pending Review")

        with st.container():
            st.markdown(
                f"""
                <div class="event-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span class="status-pill {pill_class}">{evt.status.upper()}</span> &nbsp;
                            <strong style="color:#F1F5FC; font-size:14px;">Track ID: Person {evt.track_id}</strong> &nbsp;
                            <span style="color:#6C7D93; font-size:11.5px; font-family:'JetBrains Mono', monospace;">Event: {evt.event_id}</span>
                        </div>
                        <div style="font-size:12.5px; color:{'#35D39A' if reviewed_status != 'Pending Review' else '#FFBE55'}; font-weight:600;">
                            Review Outcome: {reviewed_status}
                        </div>
                    </div>
                    <div style="margin-top:8px; font-size:13px; color:#C2D1E5;">
                        <strong>Observation:</strong> {evt.reason} <br>
                        <span style="color:#9AAAC2; font-size:12px;">Timestamp: {evt.video_timestamp}s (Frame {evt.frame_number}) | Person Conf: {evt.person_confidence:.1%} | Association: {evt.association_outcome}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_img, col_act = st.columns([2, 1])
            with col_img:
                if evt.snapshot_path and Path(evt.snapshot_path).exists():
                    st.image(evt.snapshot_path, caption=f"Visual Evidence (Track {evt.track_id})", use_container_width=True)
                else:
                    st.caption("Snapshot preview not stored on disk.")

            with col_act:
                st.markdown("##### Security Officer Action:")
                if st.button("✅ Confirm ID Physically", key=f"rev_ok_{evt.event_id}", use_container_width=True):
                    st.session_state.reviewed_cases[evt.event_id] = "Manually Verified (ID Present)"
                    st.rerun()
                if st.button("⚠️ Flag for Physical Check", key=f"rev_warn_{evt.event_id}", use_container_width=True):
                    st.session_state.reviewed_cases[evt.event_id] = "Flagged for Physical Check"
                    st.rerun()
                if st.button("❌ Dismiss Observation", key=f"rev_dis_{evt.event_id}", use_container_width=True):
                    st.session_state.reviewed_cases[evt.event_id] = "Dismissed by Officer"
                    st.rerun()


# -----------------------------------------------------------------------------
# 4. Event History Page
# -----------------------------------------------------------------------------
def render_event_history_page(processor: VideoProcessor) -> None:
    render_model_warning_if_needed(processor)

    res: Optional[ProcessingResult] = st.session_state.processed_result
    if not res or not res.events:
        st.info("No monitoring history recorded yet. Process a video in **Video Monitoring** to log events.")
        return

    report_gen = ReportGenerator()
    df = report_gen.events_to_dataframe(res.events)

    # Attach human review outcomes if any
    if "manual_review_outcome" in df.columns:
        df["manual_review_outcome"] = df["event_id"].apply(
            lambda eid: st.session_state.reviewed_cases.get(eid, "Pending Review")
        )

    # Controls
    c1, c2 = st.columns([1, 1])
    with c1:
        sel_status = st.multiselect(
            "Filter by Status:",
            options=list(df["status"].unique()),
            default=list(df["status"].unique()),
        )
    with c2:
        sort_by = st.selectbox("Sort Table by:", ["video_timestamp", "track_id", "frame_number"])

    df_filtered = df[df["status"].isin(sel_status)].sort_values(by=sort_by)

    st.markdown(f"**Total Audit Records:** `{len(df_filtered)}`")
    st.dataframe(df_filtered, use_container_width=True)

    # Snapshot inspection
    st.markdown("### 🖼️ Inspect Snapshot by Event ID")
    event_ids = list(df_filtered["event_id"].unique())
    if event_ids:
        selected_eid = st.selectbox("Select Event ID:", event_ids)
        matched_evt = next((e for e in res.events if e.event_id == selected_eid), None)
        if matched_evt and matched_evt.snapshot_path and Path(matched_evt.snapshot_path).exists():
            st.image(
                matched_evt.snapshot_path,
                caption=f"{matched_evt.event_id} | Track {matched_evt.track_id} | {matched_evt.status}",
                width=500,
            )
        else:
            st.caption("No snapshot image file found for this event.")


# -----------------------------------------------------------------------------
# 5. Reports Page
# -----------------------------------------------------------------------------
def render_reports_page(processor: VideoProcessor) -> None:
    render_model_warning_if_needed(processor)

    res: Optional[ProcessingResult] = st.session_state.processed_result
    if not res:
        st.info("No report data available. Analyze a video to generate audit reports.")
        return

    m = res.metrics
    report_gen = ReportGenerator()
    df = report_gen.events_to_dataframe(res.events)

    if "manual_review_outcome" in df.columns:
        df["manual_review_outcome"] = df["event_id"].apply(
            lambda eid: st.session_state.reviewed_cases.get(eid, "Pending Review")
        )

    st.markdown("### 📊 Operational Summary")
    s_col1, s_col2, s_col3 = st.columns(3)
    s_col1.metric("Unique Visitors Tracked", m.unique_tracked_people)
    s_col2.metric("Total Capture Events", m.total_events_captured)
    s_col3.metric("Processing Throughput", f"{m.approx_fps:.1f} FPS")

    # Honest Ground-Truth Notice
    st.markdown("### 🎯 Ground-Truth Evaluation Metrics")
    gt_eval = calculate_ground_truth_metrics(predictions=[], ground_truth=[])
    st.markdown(
        f"""
        <div style="background:#131C2F; border:1px solid #26344B; border-radius:8px; padding:14px; font-size:13px;">
            <strong>Evaluation Status:</strong> <code style="color:#27D7E7;">{gt_eval['status'].upper()}</code><br>
            <span style="color:#9AAAC2;">{gt_eval['message']}</span><br>
            <small style="color:#6C7D93;">Operational detection counts represent live observed flags, not verified test accuracy percentages. Precision, Recall, and F1 require manually annotated ground-truth boxes.</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📥 Export Reports & Video")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    d_col1, d_col2 = st.columns(2)

    with d_col1:
        st.download_button(
            label="⬇️ Download Event Audit Report (CSV)",
            data=csv_bytes,
            file_name=f"gateid_report_{Path(res.video_path).stem}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d_col2:
        if res.output_video_path and Path(res.output_video_path).exists():
            with open(res.output_video_path, "rb") as vf:
                st.download_button(
                    label="⬇️ Download Annotated Video (MP4)",
                    data=vf.read(),
                    file_name=Path(res.output_video_path).name,
                    mime="video/mp4",
                    use_container_width=True,
                )
        else:
            st.button("Annotated Video Unavailable", disabled=True, use_container_width=True)

    st.markdown("### 📋 Event Data Preview")
    st.dataframe(df, use_container_width=True)


# -----------------------------------------------------------------------------
# 6. System Status Page
# -----------------------------------------------------------------------------
def render_system_status_page(processor: VideoProcessor) -> None:
    config = get_cached_config()

    col_env, col_mod = st.columns(2)

    with col_env:
        st.markdown("### 🖥️ Environment & Dependencies")
        cv_active = HAS_OPENCV
        ultra_active = check_ultralytics_available()

        st.markdown(
            f"""
            <div class="telemetry-panel">
                <div style="margin-bottom:8px;"><span style="color:#9AAAC2;">Python Runtime:</span> <strong style="color:#F1F5FC;">{sys.version.split()[0]}</strong></div>
                <div style="margin-bottom:8px;"><span style="color:#9AAAC2;">Streamlit Dashboard:</span> <strong style="color:#F1F5FC;">{st.__version__}</strong></div>
                <div style="margin-bottom:8px;"><span style="color:#9AAAC2;">OpenCV Engine:</span> <span style="color:{'#35D39A' if cv_active else '#FFBE55'}; font-weight:700;">{'🟢 Native cv2 Active' if cv_active else '🟡 Mock Shim Active'}</span></div>
                <div style="margin-bottom:8px;"><span style="color:#9AAAC2;">Ultralytics Library:</span> <span style="color:{'#35D39A' if ultra_active else '#FFBE55'}; font-weight:700;">{'🟢 Ultralytics Loaded' if ultra_active else '🟡 Not Installed'}</span></div>
                <div style="margin-bottom:8px;"><span style="color:#9AAAC2;">Platform Architecture:</span> <strong style="color:#F1F5FC;">{sys.platform}</strong></div>
                <div><span style="color:#9AAAC2;">Configuration:</span> <strong style="color:#27D7E7;">config.yaml (v{config.get('system', {}).get('version', '1.0.0')})</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_mod:
        st.markdown("### 🧠 Neural Network & Model Status")
        person_cfg = config.get("models", {}).get("person_detector", {})
        id_cfg = config.get("models", {}).get("id_card_detector", {})

        id_path = Path(id_cfg.get("model_path", "models/id_card_detector.pt"))
        id_exists = id_path.exists()
        has_vision = getattr(processor.id_detector, "has_vision_fallback", False)

        id_status_html = "🟢 Trained Weights Present" if id_exists else ("🔵 CV Torso Fallback Active" if has_vision else "🟠 Offline (Fallback Disabled)")
        id_status_color = "#35D39A" if id_exists else ("#27D7E7" if has_vision else "#FF6B78")

        st.markdown(
            f"""
            <div class="telemetry-panel">
                <div style="margin-bottom:8px;"><span style="color:#9AAAC2;">Person Detector:</span> <strong style="color:#F1F5FC;">{person_cfg.get('model_name', 'yolov8n.pt')}</strong></div>
                <div style="margin-bottom:8px;"><span style="color:#9AAAC2;">Confidence Threshold:</span> <strong style="color:#F1F5FC;">{person_cfg.get('confidence_threshold', 0.45)}</strong></div>
                <div style="margin-bottom:8px;"><span style="color:#9AAAC2;">Tracker Configuration:</span> <strong style="color:#F1F5FC;">{person_cfg.get('tracker_type', 'bytetrack.yaml')}</strong></div>
                <div style="margin-bottom:8px;"><span style="color:#9AAAC2;">ID Detector Path:</span> <code style="color:#9AAAC2;">{id_path}</code></div>
                <div style="margin-bottom:8px;"><span style="color:#9AAAC2;">ID Detector Status:</span> <span style="color:{id_status_color}; font-weight:700;">{id_status_html}</span></div>
                <div><span style="color:#9AAAC2;">Fallback Mode Enabled:</span> <strong style="color:#35D39A;">{id_cfg.get('enabled_fallback', True)}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 📁 Output Directories Health")
    dirs = [
        ("Snapshots Directory", Path("outputs/snapshots")),
        ("Reports Directory", Path("outputs/reports")),
        ("Processed Videos Directory", Path("outputs/processed_videos")),
    ]
    d_cols = st.columns(3)
    for idx, (label, p) in enumerate(dirs):
        exists = p.exists()
        st_color = "#35D39A" if exists else "#FF6B78"
        d_cols[idx].markdown(
            f"""
            <div style="background:#131C2F; border:1px solid #26344B; border-radius:8px; padding:12px 14px;">
                <div style="color:#9AAAC2; font-size:12px; font-weight:600; text-transform:uppercase;">{label}</div>
                <code style="color:#F1F5FC; font-size:12px;">{p}</code>
                <div style="margin-top:6px; font-size:11.5px; color:{st_color}; font-weight:700;">
                    {'🟢 Accessible & Ready' if exists else '🔴 Missing Directory'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Main Application Router
# -----------------------------------------------------------------------------
def main() -> None:
    processor = get_cached_video_processor()
    current_page = render_sidebar(processor)
    render_header(current_page)

    if current_page == "Dashboard":
        render_dashboard_page(processor)
    elif current_page == "Video Monitoring":
        render_video_monitoring_page(processor)
    elif current_page == "Verification Queue":
        render_verification_queue_page(processor)
    elif current_page == "Event History":
        render_event_history_page(processor)
    elif current_page == "Reports":
        render_reports_page(processor)
    elif current_page == "System Status":
        render_system_status_page(processor)
    else:
        render_dashboard_page(processor)


if __name__ == "__main__":
    main()
