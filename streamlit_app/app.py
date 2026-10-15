"""Aura-themed local Streamlit interface for Synthetic Data Generation & Predictive Analytics.

Authentication is intended for a local college demonstration. It is not a
replacement for a production identity provider or internet-facing security.
"""

from __future__ import annotations

import hashlib
import hmac
from io import StringIO
import os
from contextlib import contextmanager
import re
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from sdv.utils import load_synthesizer as sdv_load_synthesizer
from sdv.single_table import CTGANSynthesizer
from streamlit_app.services.dataset_health import (
    MAX_UPLOAD_BYTES,
    DatasetInputError,
    analyze_dataset,
    parse_csv_bytes,
)


APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"
MODEL_DIR = APP_DIR / "models"
REPORT_DIR = APP_DIR / "reports"
SYNTHETIC_DIR = APP_DIR / "synthetic_data"
AUTH_DB_PATH = Path(os.environ.get("SYNTHETIC_DATA_AUTH_DB", APP_DIR / ".streamlit" / "auth.sqlite3"))
PBKDF2_ITERATIONS = 600_000
SESSION_IDLE_SECONDS = 30 * 60
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def _connect_auth_db(db_path: Path = AUTH_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path), timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def _auth_connection(db_path: Path = AUTH_DB_PATH):
    connection = _connect_auth_db(db_path)
    try:
        yield connection
    except Exception:
        connection.rollback()
        raise
    else:
        connection.commit()
    finally:
        connection.close()


def initialize_auth_db(db_path: Path = AUTH_DB_PATH) -> None:
    with _auth_connection(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_salt BLOB NOT NULL,
                password_hash BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def password_strength(password: str) -> tuple[int, str]:
    checks = [
        len(password) >= 10,
        any(char.islower() for char in password),
        any(char.isupper() for char in password),
        any(char.isdigit() for char in password),
        any(not char.isalnum() for char in password),
    ]
    score = sum(checks)
    labels = ["Add a password", "Very weak", "Weak", "Fair", "Good", "Strong"]
    return score, labels[score]


def validate_registration(
    full_name: str,
    email: str,
    password: str,
    confirmation: str,
    accepted_terms: bool,
) -> str | None:
    name = full_name.strip()
    normalized_email = email.strip()
    if not name:
        return "Enter your full name."
    if len(name) > 100:
        return "Name must be 100 characters or fewer."
    if not normalized_email or len(normalized_email) > 254 or not EMAIL_PATTERN.fullmatch(normalized_email):
        return "Enter a valid email address."
    if len(password) < 10 or len(password) > 128:
        return "Use a password between 10 and 128 characters."
    if password_strength(password)[0] < 4:
        return "Use at least four of these: lowercase, uppercase, number, and symbol, with 10+ characters."
    if password != confirmation:
        return "Passwords do not match."
    if not accepted_terms:
        return "Accept the local demo terms to create an account."
    return None


def create_account(
    full_name: str,
    email: str,
    password: str,
    db_path: Path = AUTH_DB_PATH,
) -> tuple[bool, str]:
    """Create an account; only a salted PBKDF2 hash is persisted."""
    normalized_email = email.strip().casefold()
    salt = secrets.token_bytes(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    try:
        with _auth_connection(db_path) as connection:
            connection.execute(
                "INSERT INTO users (full_name, email, password_salt, password_hash) VALUES (?, ?, ?, ?)",
                (full_name.strip(), normalized_email, salt, password_hash),
            )
    except sqlite3.IntegrityError:
        return False, "An account with that email already exists. Try signing in."
    return True, "Account created. Sign in with your new credentials to continue."


def authenticate_user(email: str, password: str, db_path: Path = AUTH_DB_PATH) -> dict[str, Any] | None:
    normalized_email = email.strip().casefold()
    with _auth_connection(db_path) as connection:
        row = connection.execute(
            "SELECT id, full_name, email, password_salt, password_hash FROM users WHERE email = ?",
            (normalized_email,),
        ).fetchone()
    if row is None:
        return None
    candidate = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), row["password_salt"], PBKDF2_ITERATIONS
    )
    if not hmac.compare_digest(candidate, row["password_hash"]):
        return None
    return {"id": row["id"], "full_name": row["full_name"], "email": row["email"]}


@st.cache_data(show_spinner=False)
def read_csv_artifact(path_string: str) -> pd.DataFrame:
    return pd.read_csv(path_string)


@st.cache_resource(show_spinner="Loading the trained CTGAN model…")
def load_synthesizer(model_path: str) -> CTGANSynthesizer:
    return sdv_load_synthesizer(filepath=Path(model_path))


THEMES = {
    "Dark": {
        "bg": "#08090B", "surface": "#111116", "surface2": "#191720",
        "border": "rgba(255,255,255,.08)", "accent": "#B56CFF", "accent2": "#D38BFF",
        "text": "#F5F3F8", "muted": "#A6A1AE", "soft": "#77717F",
        "surface_glass": "rgba(25,23,32,.88)", "surface_glass2": "rgba(15,15,20,.82)",
        "input": "rgba(8,9,12,.74)", "nav": "rgba(15,14,20,.74)",
        "nav_active": "linear-gradient(180deg,rgba(142,82,183,.28),rgba(93,54,122,.17))",
        "shadow": "rgba(0,0,0,.22)", "ambient1": "rgba(111,53,160,.20)",
        "ambient2": "rgba(93,49,126,.13)", "ambient3": "rgba(118,54,144,.11)",
        "plot_bg": "#111116", "grid": "rgba(245,243,248,.10)",
        "colorway": ["#B56CFF", "#D38BFF", "#8C73FF", "#E0B0FF", "#9B8AA8"],
        "button_text": "#FFFFFF", "table_text": "#F5F3F8",
    },
    "Light": {
        "bg": "#F6F4FA", "surface": "#FFFFFF", "surface2": "#F0ECF7",
        "border": "rgba(35,20,50,.12)", "accent": "#8142C5", "accent2": "#A66CE0",
        "text": "#21172B", "muted": "#6E6478", "soft": "#82788C",
        "surface_glass": "rgba(255,255,255,.88)", "surface_glass2": "rgba(250,248,253,.94)",
        "input": "rgba(255,255,255,.94)", "nav": "rgba(255,255,255,.82)",
        "nav_active": "linear-gradient(180deg,rgba(166,108,224,.16),rgba(129,66,197,.08))",
        "shadow": "rgba(51,31,69,.09)", "ambient1": "rgba(173,126,224,.18)",
        "ambient2": "rgba(214,186,242,.25)", "ambient3": "rgba(189,151,225,.13)",
        "plot_bg": "#FFFFFF", "grid": "rgba(35,20,50,.10)",
        "colorway": ["#8142C5", "#A66CE0", "#5D66B3", "#C184CF", "#627D8A"],
        "button_text": "#FFFFFF", "table_text": "#21172B",
    },
}


def inject_theme(mode: str = "Dark") -> None:
    palette = THEMES.get(mode, THEMES["Dark"])
    css_variables = " ".join(f"--aura-{key.replace('_', '-')}: {value};" for key, value in palette.items() if not isinstance(value, list))
    background = (
        f"radial-gradient(ellipse at 12% 9%,{palette['ambient1']},transparent 34%),"
        f"radial-gradient(ellipse at 86% 21%,{palette['ambient2']},transparent 30%),"
        f"radial-gradient(ellipse at 72% 88%,{palette['ambient3']},transparent 32%)"
    )
    primary_start, primary_end = ("#7743a7", "#75449d") if mode == "Dark" else ("#8142C5", "#A66CE0")
    st.markdown(
        f"""
        <style>
        :root {{ {css_variables} --aura-purple:var(--aura-accent); --aura-lavender:var(--aura-accent-2); }}
        html, body, [data-testid="stAppViewContainer"] {{ background:var(--aura-bg); color:var(--aura-text); font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color-scheme:{'dark' if mode == 'Dark' else 'light'}; }}
        [data-testid="stAppViewContainer"] {{ background-image:{background}; background-attachment:fixed; }}
        [data-testid="stHeader"] {{ background:color-mix(in srgb,var(--aura-bg) 84%,transparent); }}
        [data-testid="stSidebar"] {{ background:linear-gradient(180deg,var(--aura-surface-2),var(--aura-bg)); border-right:1px solid var(--aura-border); }}
        [data-testid="stMainBlockContainer"] {{ max-width:1220px; padding-top:2.2rem; }}
        h1,h2,h3,h4 {{ color:var(--aura-text); letter-spacing:-.035em; }} h1 {{ font-weight:720; }}
        p,label,[data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"] {{ color:var(--aura-muted); }} a {{ color:var(--aura-accent)!important; }}
        [data-testid="stVerticalBlockBorderWrapper"] {{ background:linear-gradient(145deg,var(--aura-surface-glass),var(--aura-surface-glass2)); border:1px solid var(--aura-border); border-radius:20px; box-shadow:0 18px 48px var(--aura-shadow),inset 0 1px rgba(255,255,255,.08); backdrop-filter:blur(16px); }}
        [data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,[data-testid="stTextArea"] textarea {{ color:var(--aura-text); background:var(--aura-input); border:1px solid var(--aura-border); border-radius:12px; min-height:2.9rem; transition:border-color .18s ease,box-shadow .18s ease; }}
        [data-testid="stTextInput"] input:focus,[data-testid="stNumberInput"] input:focus,[data-testid="stTextArea"] textarea:focus {{ border-color:var(--aura-accent); box-shadow:0 0 0 3px color-mix(in srgb,var(--aura-accent) 16%,transparent),0 0 22px color-mix(in srgb,var(--aura-accent) 10%,transparent); }}
        [data-testid="stSelectbox"] div[data-baseweb="select"]>div,[data-testid="stMultiSelect"] div[data-baseweb="select"]>div {{ background:var(--aura-input); border-color:var(--aura-border); border-radius:12px; }}
        [data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"] {{ background:var(--aura-surface); border:1px solid var(--aura-border); }}
        .stButton>button,.stDownloadButton>button,button[kind="primary"] {{ color:var(--aura-button-text); border:1px solid color-mix(in srgb,var(--aura-accent) 40%,transparent); border-radius:12px; background:linear-gradient(115deg,{primary_start},var(--aura-accent) 55%,{primary_end}); box-shadow:0 8px 24px color-mix(in srgb,var(--aura-accent) 20%,transparent); transition:transform .18s ease,box-shadow .18s ease,filter .18s ease; }}
        .stButton>button:hover,.stDownloadButton>button:hover {{ transform:translateY(-1px); filter:brightness(1.06); box-shadow:0 10px 28px color-mix(in srgb,var(--aura-accent) 28%,transparent); }}
        button:focus-visible {{ outline:2px solid var(--aura-lavender)!important; outline-offset:2px; }}
        [data-testid="stMetric"] {{ background:var(--aura-surface-2); border:1px solid var(--aura-border); border-radius:15px; padding:14px 16px; }}
        [data-testid="stMetricValue"],[data-testid="stMetricLabel"] {{ color:var(--aura-text); }}
        [data-testid="stDataFrame"] {{ --gdg-bg-cell:var(--aura-surface)!important; --gdg-bg-cell-medium:var(--aura-surface-2)!important; --gdg-bg-header:var(--aura-surface-2)!important; --gdg-bg-header-hovered:var(--aura-surface-2)!important; --gdg-bg-header-has-focus:var(--aura-surface-2)!important; --gdg-text-dark:var(--aura-text)!important; --gdg-text-medium:var(--aura-muted)!important; --gdg-text-light:var(--aura-soft)!important; --gdg-border-color:var(--aura-border)!important; border:1px solid var(--aura-border); border-radius:14px; overflow:hidden; }}
        [data-testid="stProgressBar"]>div>div {{ background:linear-gradient(90deg,var(--aura-accent),var(--aura-accent-2)); }}
        [data-testid="stAlert"] {{ background:var(--aura-surface-2); border-color:var(--aura-border); color:var(--aura-text); }}
        [data-testid="stCheckbox"] label, [data-testid="stToggle"] label {{ color:var(--aura-muted); }}
        [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{ background:var(--aura-accent); border-color:var(--aura-accent); }}
        .aura-brand {{ display:flex; align-items:center; gap:.7rem; margin:.3rem 0 1.2rem; }}
        .aura-mark {{ width:38px; height:38px; display:grid; place-items:center; border-radius:13px; color:#f4eaff; font-size:19px; background:linear-gradient(145deg,#5c3476,var(--aura-accent)); box-shadow:0 0 24px color-mix(in srgb,var(--aura-accent) 24%,transparent); }}
        .aura-brand-name {{ color:var(--aura-text); font-weight:700; letter-spacing:-.025em; }} .aura-brand-sub {{ color:var(--aura-soft); font-size:.74rem; letter-spacing:.12em; text-transform:uppercase; }}
        .top-nav-divider {{ height:1px; margin:.2rem 0 1.6rem; background:linear-gradient(90deg,color-mix(in srgb,var(--aura-accent) 34%,transparent),var(--aura-border) 45%,transparent); }}
        [data-testid="stSegmentedControl"] {{ width:100%; }}
        [data-testid="stSegmentedControl"] [role="radiogroup"] {{ width:100%; display:flex; flex-wrap:wrap; gap:5px; padding:4px; border:1px solid var(--aura-border); border-radius:14px; background:var(--aura-nav); }}
        [data-testid="stSegmentedControl"] [role="radio"] {{ flex:1 1 auto; min-width:max-content; justify-content:center; color:var(--aura-muted); border:1px solid transparent; border-radius:10px; background:transparent; transition:color .18s ease,background .18s ease,box-shadow .18s ease; }}
        [data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"] {{ color:var(--aura-text); border-color:color-mix(in srgb,var(--aura-accent) 25%,transparent); background:var(--aura-nav-active); box-shadow:0 0 18px color-mix(in srgb,var(--aura-accent) 12%,transparent),inset 0 -2px var(--aura-accent); }}
        [data-testid="stSegmentedControl"] [role="radio"]:hover {{ color:var(--aura-text); background:color-mix(in srgb,var(--aura-accent) 10%,transparent); }}
        [data-testid="stSegmentedControl"] [role="radio"]:focus-visible {{ outline:2px solid var(--aura-lavender); outline-offset:2px; }}
        .eyebrow {{ color:var(--aura-accent-2); text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; }}
        .hero-title {{ font-size:clamp(2rem,4vw,3.35rem); line-height:1.05; font-weight:760; margin:.45rem 0 .8rem; }}
        .hero-copy {{ color:var(--aura-muted); max-width:680px; line-height:1.65; }} .auth-note {{ color:var(--aura-soft); font-size:.82rem; line-height:1.55; }}
        @media(max-width:900px) {{ [data-testid="stSegmentedControl"] [role="radiogroup"] {{ justify-content:flex-start; }} [data-testid="stSegmentedControl"] [role="radio"] {{ flex:1 1 30%; }} }}
        @media(max-width:700px) {{ [data-testid="stMainBlockContainer"] {{ padding:1rem .8rem 2rem; }} [data-testid="stVerticalBlockBorderWrapper"] {{ border-radius:16px; }} [data-testid="stSegmentedControl"] [role="radio"] {{ flex:1 1 42%; }} }}
        @media(prefers-reduced-motion:reduce) {{ *,*::before,*::after {{ transition:none!important; animation:none!important; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def brand(compact: bool = False) -> None:
    subtitle = "Synthetic Data Lab" if compact else "Synthetic Data Generation & Predictive Analytics"
    st.markdown(
        f'<div class="aura-brand"><div class="aura-mark" aria-hidden="true">✦</div>'
        f'<div><div class="aura-brand-name">Synthetic Data Lab</div><div class="aura-brand-sub">{subtitle}</div></div></div>',
        unsafe_allow_html=True,
    )


def show_authentication() -> None:
    if st.session_state.get("auth_page", "Login") == "Login":
        # Clear password widget values after a successful account creation/page switch.
        st.session_state.pop("signup_password", None)
        st.session_state.pop("signup_confirmation", None)
    _, center, _ = st.columns([1, 1.1, 1])
    with center:
        brand()
        with st.container(border=True):
            if st.session_state.get("auth_page", "Login") == "Login":
                created_notice = st.session_state.pop("account_created_notice", None)
                if created_notice:
                    st.success(created_notice)
                st.markdown('<div class="eyebrow">Your private workspace</div>', unsafe_allow_html=True)
                st.markdown("## Welcome back")
                st.caption("Sign in to continue to your synthetic data workspace.")
                show_password = st.checkbox("Show password", key="login_show_password")
                with st.form("login_form"):
                    email = st.text_input("Email", key="login_email", placeholder="you@example.com")
                    password = st.text_input("Password", key="login_password", type="default" if show_password else "password")
                    remember = st.checkbox("Remember me for this browser session", key="remember_session_choice")
                    submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
                if st.button("Forgot password?", key="forgot_password"):
                    st.info("Password recovery is unavailable in this local demo because no verified email service is configured.")
                if submitted:
                    account = authenticate_user(email, password)
                    if account is None:
                        st.error("Email or password is incorrect.")
                    else:
                        st.session_state.authenticated = True
                        st.session_state.user_id = account["id"]
                        st.session_state.user_name = account["full_name"]
                        st.session_state.user_email = account["email"]
                        st.session_state.remember_session = remember
                        st.session_state.last_activity = time.time()
                        st.rerun()
                st.markdown('<div class="auth-note">New to the lab?</div>', unsafe_allow_html=True)
                if st.button("Create an account", key="go_signup", width="stretch"):
                    st.session_state.auth_page = "Sign Up"
                    st.rerun()
            else:
                st.markdown('<div class="eyebrow">Start your workspace</div>', unsafe_allow_html=True)
                st.markdown("## Create your account")
                st.caption("A local demo account gives you access to this workspace on this machine.")
                show_password = st.checkbox("Show passwords", key="signup_show_password")
                full_name = st.text_input("Full name", key="signup_name", placeholder="Your name")
                email = st.text_input("Email", key="signup_email", placeholder="you@example.com")
                password_type = "default" if show_password else "password"
                password = st.text_input("Password", key="signup_password", type=password_type)
                confirmation = st.text_input("Confirm password", key="signup_confirmation", type=password_type)
                score, strength = password_strength(password)
                st.caption(f"Password strength: {strength}")
                st.progress(score / 5)
                with st.expander("Local demo terms"):
                    st.write("Use this account only for the local Synthetic Data Lab demonstration. The account database is stored on this machine. Do not reuse a personal password. This app does not provide email verification, password recovery, or production account security.")
                accepted_terms = st.checkbox("I agree to the local demo terms", key="signup_terms")
                if st.button("Create account", type="primary", key="create_account", width="stretch"):
                    error = validate_registration(full_name, email, password, confirmation, accepted_terms)
                    if error:
                        st.error(error)
                    else:
                        created, message = create_account(full_name, email, password)
                        if created:
                            st.session_state.account_created_notice = message
                            st.session_state.auth_page = "Login"
                            st.session_state.login_email = email.strip().casefold()
                            st.rerun()
                        else:
                            st.error(message)
                st.markdown('<div class="auth-note">Already registered?</div>', unsafe_allow_html=True)
                if st.button("Back to sign in", key="go_login", width="stretch"):
                    st.session_state.auth_page = "Login"
                    st.rerun()
        st.caption("Local SQLite authentication • Not intended for production or internet exposure")


def load_frame(relative_path: str) -> pd.DataFrame | None:
    path = APP_DIR / relative_path
    if not path.is_file():
        return None
    try:
        return read_csv_artifact(str(path))
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.warning(f"Could not read `{relative_path}`: {error}")
        return None


def hero(kicker: str, title: str, copy: str) -> None:
    st.markdown(f'<div class="eyebrow">{kicker}</div><div class="hero-title">{title}</div><div class="hero-copy">{copy}</div><br>', unsafe_allow_html=True)


def plotly_theme(fig: Any) -> Any:
    palette = THEMES.get(st.session_state.get("theme_mode", "Dark"), THEMES["Dark"])
    template = "plotly_dark" if st.session_state.get("theme_mode", "Dark") == "Dark" else "plotly_white"
    fig.update_layout(
        template=template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=palette["text"]),
        colorway=palette["colorway"],
        margin=dict(l=16, r=16, t=34, b=16),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=palette["text"])),
        hoverlabel=dict(bgcolor=palette["surface"], bordercolor=palette["border"], font=dict(color=palette["text"])),
    )
    fig.update_xaxes(gridcolor=palette["grid"], zerolinecolor=palette["grid"], linecolor=palette["border"], tickfont_color=palette["muted"], title_font_color=palette["text"])
    fig.update_yaxes(gridcolor=palette["grid"], zerolinecolor=palette["grid"], linecolor=palette["border"], tickfont_color=palette["muted"], title_font_color=palette["text"])
    return fig


def chart_colors() -> list[str]:
    return THEMES.get(st.session_state.get("theme_mode", "Dark"), THEMES["Dark"])["colorway"]


def page_dashboard() -> None:
    real = load_frame("data/heart_unique.csv")
    synthetic = load_frame("synthetic_data/heart_synthetic.csv")
    summary = load_frame("reports/quality_summary.csv")
    hero("Workspace overview", "Synthetic Data Lab", "A clear view of your datasets, generation workflow, and saved analysis artifacts.")
    a, b, c, d = st.columns(4)
    a.metric("Unique real records", f"{len(real):,}" if real is not None else "Unavailable")
    b.metric("Saved synthetic records", f"{len(synthetic):,}" if synthetic is not None else "Unavailable")
    c.metric("Features", f"{len(real.columns):,}" if real is not None else "Unavailable")
    if summary is not None and not summary.empty and "exact_record_matches" in summary:
        d.metric("Exact record matches", f"{int(summary.iloc[0]['exact_record_matches']):,}")
    else:
        d.metric("Quality summary", "Unavailable")
    left, right = st.columns([1.15, .85])
    with left, st.container(border=True):
        st.subheader("Project workflow")
        st.markdown("**01 · Explore**  Review the original and deduplicated heart datasets.")
        st.markdown("**02 · Generate**  Sample new records from the existing trained CTGAN model.")
        st.markdown("**03 · Evaluate**  Review saved quality, privacy, and model-utility reports.")
    with right, st.container(border=True):
        st.subheader("Available artifacts")
        for path in ["models/heart_ctgan.pkl", "models/heart_metadata.json", "reports/quality_summary.csv", "reports/privacy/privacy_summary.csv", "reports/ml_utility/ml_utility_results.csv"]:
            st.markdown(f"{'●' if (APP_DIR / path).is_file() else '○'}  `{path}`")
    st.caption("Report files are displayed as saved. The app does not rerun report scripts or overwrite existing artifacts.")


def page_explorer() -> None:
    hero("Dataset explorer", "Explore the data", "Inspect the project datasets, feature summaries, and distributions without changing source files.")
    choices = {"Deduplicated real dataset": "data/heart_unique.csv", "Original real dataset": "data/heart.csv", "Saved synthetic dataset": "synthetic_data/heart_synthetic.csv"}
    label = st.selectbox("Dataset", list(choices))
    df = load_frame(choices[label])
    if df is None:
        st.info(f"`{choices[label]}` is not available.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns):,}")
    c3.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    with st.expander("Missing values by feature"):
        missing = df.isna().sum().rename("missing").to_frame()
        missing["percent"] = (missing["missing"] / max(len(df), 1) * 100).round(2)
        st.dataframe(missing, width="stretch")
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("Download this dataset as CSV", df.to_csv(index=False).encode("utf-8"), file_name=Path(choices[label]).name, mime="text/csv", key="download_explorer_csv")
    column = st.selectbox("Feature distribution", list(df.columns), key="distribution_feature")
    if pd.api.types.is_numeric_dtype(df[column]) and df[column].nunique() > 12:
        fig = px.histogram(df, x=column, nbins=24, color_discrete_sequence=chart_colors())
    else:
        counts = df[column].astype(str).value_counts().rename_axis(column).reset_index(name="records")
        fig = px.bar(counts, x=column, y="records", color_discrete_sequence=chart_colors())
    st.plotly_chart(plotly_theme(fig), width="stretch")


def page_generate() -> None:
    hero("CTGAN generation", "Generate synthetic records", "Use the existing trained model. New rows stay in this session until you download them.")
    model_path = MODEL_DIR / "heart_ctgan.pkl"
    if not model_path.is_file():
        st.warning("Generation is unavailable: the trained model file was not found.")
        return
    with st.container(border=True):
        st.markdown("The trained `heart_ctgan.pkl` model is loaded from the project. Generated rows are held in memory and are **not written** to `synthetic_data/`.")
        rows = st.number_input("Rows to generate", min_value=1, max_value=10_000, value=302, step=1)
        if st.button("Generate data", type="primary", key="generate_data"):
            try:
                with st.spinner("Sampling from the trained CTGAN model…"):
                    synthesizer = load_synthesizer(str(model_path))
                    generated = synthesizer.sample(num_rows=int(rows))
                st.session_state.generated_csv = generated.to_csv(index=False)
                st.session_state.generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
                st.success(f"Generated {len(generated):,} rows from the existing model.")
            except Exception as error:
                st.error(f"Generation failed: {error}")
    csv_data = st.session_state.get("generated_csv")
    if csv_data:
        generated = pd.read_csv(StringIO(csv_data))
        st.caption(f"Generated during this session at {st.session_state.get('generated_at', 'unknown time')}.")
        st.dataframe(generated.head(100), width="stretch", hide_index=True)
        st.download_button("Download generated CSV", csv_data.encode("utf-8"), "heart_synthetic_generated.csv", "text/csv", key="download_generated_csv")


def page_quality() -> None:
    hero("Saved evaluations", "Quality & privacy reports", "Browse existing report artifacts without rerunning analysis scripts.")
    summary = load_frame("reports/quality_summary.csv")
    numerical = load_frame("reports/numerical_comparison.csv")
    categorical = load_frame("reports/categorical_comparison.csv")
    privacy = load_frame("reports/privacy/privacy_summary.csv")
    if summary is not None and not summary.empty:
        row = summary.iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Real rows", f"{int(row['real_rows']):,}")
        m2.metric("Synthetic rows", f"{int(row['synthetic_rows']):,}")
        m3.metric("Exact record matches", f"{int(row['exact_record_matches']):,}")
    else:
        st.info("Detailed quality summary is unavailable until its report CSV exists.")
    tab_dist, tab_corr, tab_privacy = st.tabs(["Distributions", "Correlations", "Privacy screening"])
    with tab_dist:
        if numerical is not None:
            st.subheader("Numerical feature comparison")
            st.dataframe(numerical, width="stretch", hide_index=True)
            long = numerical.melt(id_vars="column", value_vars=["real_mean", "synthetic_mean"], var_name="dataset", value_name="mean")
            fig = px.bar(long, x="column", y="mean", color="dataset", barmode="group", color_discrete_sequence=chart_colors())
            st.plotly_chart(plotly_theme(fig), width="stretch")
        if categorical is not None:
            st.subheader("Categorical feature comparison")
            st.dataframe(categorical, width="stretch", hide_index=True)
    with tab_corr:
        difference = load_frame("reports/correlation_difference.csv")
        if difference is None:
            st.info("Correlation comparison is unavailable until its report CSV exists.")
        else:
            difference = difference.set_index(difference.columns[0]).apply(pd.to_numeric, errors="coerce")
            fig = px.imshow(difference, color_continuous_scale="PuOr", zmin=-1, zmax=1, aspect="auto", labels={"color": "Synthetic − real"})
            st.plotly_chart(plotly_theme(fig), width="stretch")
            st.caption("Cells show synthetic correlation minus real correlation for each feature pair.")
    with tab_privacy:
        if privacy is None or privacy.empty:
            st.info("Privacy screening report is unavailable until `reports/privacy/privacy_summary.csv` exists.")
        else:
            p = privacy.iloc[0]
            cols = st.columns(4)
            for col, key, label in zip(cols, ["minimum_distance", "median_distance", "mean_distance", "p05_distance"], ["Minimum", "Median", "Mean", "5th percentile"]):
                col.metric(label, f"{float(p[key]):.3f}")
            distances = load_frame("reports/privacy/nearest_neighbor_distances.csv")
            if distances is not None and "nearest_real_distance" in distances:
                fig = px.histogram(distances, x="nearest_real_distance", nbins=24, color_discrete_sequence=chart_colors())
                st.plotly_chart(plotly_theme(fig), width="stretch")
            st.warning("This is a relative nearest-neighbor screening report, not a formal privacy guarantee.")


def page_predictive() -> None:
    hero("Model utility", "Predictive analytics", "Review the saved Logistic Regression evaluation comparing real-data and synthetic-data training.")
    results = load_frame("reports/ml_utility/ml_utility_results.csv")
    if results is None or results.empty:
        st.info("Saved model evaluation is unavailable. The project does not include a saved prediction model for individual predictions.")
        return
    st.dataframe(results, width="stretch", hide_index=True)
    metrics = [column for column in ["accuracy", "precision", "recall", "f1_score", "roc_auc"] if column in results]
    if metrics and "model" in results:
        long = results.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="score")
        fig = px.bar(long, x="metric", y="score", color="model", barmode="group", range_y=[0, 1], color_discrete_sequence=chart_colors())
        st.plotly_chart(plotly_theme(fig), width="stretch")
    st.info("Interactive predictions are unavailable: the repository contains evaluation results, but no persisted classifier and no prediction input pipeline. No predictions are fabricated.")


def page_dataset_health() -> None:
    hero(
        "Dataset diagnostics",
        "Smart Dataset Health Check",
        "Inspect missing values, repeated rows, schema patterns, and numerical distributions without changing the selected dataset.",
    )
    project_sources = {
        "Deduplicated real dataset": "data/heart_unique.csv",
        "Original real dataset": "data/heart.csv",
        "Saved synthetic dataset": "synthetic_data/heart_synthetic.csv",
    }
    source = st.selectbox(
        "Dataset source",
        ["Upload a CSV", *project_sources],
        key="health_dataset_source",
    )

    if source == "Upload a CSV":
        uploaded = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            max_upload_size=MAX_UPLOAD_BYTES // (1024 * 1024),
            help="UTF-8 CSV only; maximum 20 MB, 250,000 data rows, 500 columns, and 5 million cells.",
            key="health_csv_upload",
        )
        if uploaded is None:
            st.info("Upload a CSV to preview it and run the health checks.")
            return
        content = uploaded.getvalue()
        display_name = "Uploaded CSV"
        source_id = "upload"
    else:
        relative_path = project_sources[source]
        try:
            content = (APP_DIR / relative_path).read_bytes()
        except OSError:
            st.error(f"The selected project dataset is unavailable: {source}.")
            return
        display_name = source
        source_id = relative_path

    try:
        frame = parse_csv_bytes(content)
    except DatasetInputError as error:
        st.error(str(error))
        return

    content_fingerprint = hashlib.sha256(source_id.encode("utf-8") + b"\0" + content).hexdigest()
    st.caption(f"Previewing: {display_name}. The source file is read-only during this analysis.")
    row_metric, column_metric, type_metric = st.columns(3)
    row_metric.metric("Rows", f"{len(frame):,}")
    column_metric.metric("Columns", f"{len(frame.columns):,}")
    type_metric.metric("Detected data types", f"{frame.dtypes.astype(str).nunique():,}")

    with st.expander("Preview rows and inferred types", expanded=True):
        st.dataframe(frame.head(12), width="stretch", hide_index=True)
        types = pd.DataFrame({"column": frame.columns.astype(str), "inferred CSV dtype": frame.dtypes.astype(str).to_numpy()})
        st.dataframe(types, width="stretch", hide_index=True)

    target_options = [
        str(column)
        for column in frame.columns
        if frame[column].nunique(dropna=True) <= 20
    ]
    target_column = st.selectbox(
        "Target column for optional class-balance check",
        options=[None, *target_options],
        format_func=lambda value: "No target selected" if value is None else value,
        help="Leave unselected unless this dataset has a known classification target.",
        key="health_target_column",
    )
    report_fingerprint = hashlib.sha256(
        f"{content_fingerprint}\0{target_column or ''}".encode("utf-8")
    ).hexdigest()

    if st.button("Run health check", type="primary", key="run_health_check"):
        try:
            with st.spinner("Calculating dataset health findings…"):
                report = analyze_dataset(frame, target_column=target_column)
            st.session_state.health_check_report = {
                "fingerprint": report_fingerprint,
                "analysis": report,
            }
            st.success("Health check completed from the selected dataset.")
        except DatasetInputError as error:
            st.error(str(error))
            return
        except (ValueError, TypeError, OverflowError) as error:
            st.error(f"The dataset could not be analyzed: {error}")
            return

    stored_report = st.session_state.get("health_check_report", {})
    if stored_report.get("fingerprint") != report_fingerprint:
        st.info("Review the preview, optionally select a target, then run the health check.")
        return

    report = stored_report["analysis"]
    summary = report["summary"]
    st.subheader("Health check summary")
    metrics = st.columns(5)
    metrics[0].metric("Rows", f"{summary['rows']:,}")
    metrics[1].metric("Columns", f"{summary['columns']:,}")
    metrics[2].metric("Missing cells", f"{summary['missing_cells']:,}", f"{summary['missing_percent']:.2f}%")
    metrics[3].metric("Exact duplicate rows", f"{summary['duplicate_rows']:,}", f"{summary['duplicate_percent']:.2f}%")
    metrics[4].metric("Columns with findings", f"{summary['columns_with_findings']:,}")
    st.caption(f"Possible numerical outliers flagged by the IQR rule: {summary['outlier_values']:,}. These are statistical candidates, not confirmed errors.")
    st.info("No single health score is shown: detected facts and heuristic warnings have different meanings. Findings do not prove that a dataset is safe or suitable for machine learning.")

    st.subheader("Missingness by column")
    missing_chart = report["columns"].sort_values("missing_percent", ascending=False).head(20)
    missing_fig = px.bar(
        missing_chart,
        x="column",
        y="missing_percent",
        labels={"column": "Column", "missing_percent": "Missing values (%)"},
        color_discrete_sequence=chart_colors(),
    )
    st.plotly_chart(plotly_theme(missing_fig), width="stretch")

    if not report["findings"].empty:
        finding_counts = report["findings"].groupby("category", as_index=False).size().sort_values("size", ascending=False)
        finding_fig = px.bar(
            finding_counts,
            x="category",
            y="size",
            labels={"category": "Finding type", "size": "Count"},
            color_discrete_sequence=chart_colors(),
        )
        st.plotly_chart(plotly_theme(finding_fig), width="stretch")
        st.subheader("Findings and recommendations")
        st.dataframe(report["findings"], width="stretch", hide_index=True)
    else:
        st.success("No findings were triggered by the checks in this health check.")

    st.subheader("Column-level report")
    st.dataframe(report["columns"], width="stretch", hide_index=True)
    findings_csv = report["findings"].to_csv(index=False).encode("utf-8")
    columns_csv = report["columns"].to_csv(index=False).encode("utf-8")
    download_findings, download_columns = st.columns(2)
    download_findings.download_button(
        "Download findings CSV",
        findings_csv,
        file_name="dataset_health_findings.csv",
        mime="text/csv",
        key="download_health_findings",
        width="stretch",
    )
    download_columns.download_button(
        "Download column summary CSV",
        columns_csv,
        file_name="dataset_health_columns.csv",
        mime="text/csv",
        key="download_health_columns",
        width="stretch",
    )

    numeric_types = {"numerical", "numeric stored as text"}
    numeric_columns = report["columns"].loc[
        report["columns"]["inferred_type"].isin(numeric_types), "column"
    ].tolist()
    if numeric_columns:
        selected_numeric = st.selectbox("Numerical distribution", numeric_columns, key="health_numeric_column")
        distribution_series = pd.to_numeric(frame[selected_numeric], errors="coerce")
        distribution_series = distribution_series.replace([float("inf"), float("-inf")], pd.NA).dropna().astype(float)
        distribution_values = pd.DataFrame({selected_numeric: distribution_series})
        distribution, box = st.columns(2)
        with distribution:
            histogram = px.histogram(
                distribution_values,
                x=selected_numeric,
                nbins=30,
                color_discrete_sequence=chart_colors(),
            )
            st.plotly_chart(plotly_theme(histogram), width="stretch")
        with box:
            box_plot = px.box(
                distribution_values,
                y=selected_numeric,
                points="outliers",
                color_discrete_sequence=chart_colors(),
            )
            st.plotly_chart(plotly_theme(box_plot), width="stretch")
    else:
        st.info("No continuous numerical columns were detected for histogram or box-plot analysis.")

    target_distribution = report["target_distribution"]
    if target_column and not target_distribution.empty:
        st.subheader(f"Selected target distribution: {target_column}")
        target_fig = px.bar(
            target_distribution,
            x="class",
            y="percent",
            labels={"class": "Class", "percent": "Non-missing labels (%)"},
            color_discrete_sequence=chart_colors(),
        )
        st.plotly_chart(plotly_theme(target_fig), width="stretch")

    st.caption("Duplicate analysis confirms exact repeated rows only. Near-duplicate similarity and domain-specific validity require a known schema and are not asserted by this generic check.")


PAGES = {
    "Overview": page_dashboard,
    "Dataset Explorer": page_explorer,
    "Dataset Health Check": page_dataset_health,
    "Generate Data": page_generate,
    "Quality Reports": page_quality,
    "Predictive Analytics": page_predictive,
}


def main() -> None:
    st.set_page_config(page_title="Synthetic Data Lab", page_icon="✦", layout="wide", initial_sidebar_state="collapsed")
    st.session_state.theme_mode = "Dark"
    inject_theme("Dark")
    try:
        initialize_auth_db()
    except (OSError, sqlite3.Error) as error:
        st.error(f"Could not initialize the local authentication database: {error}")
        st.stop()
    if st.session_state.get("authenticated"):
        # Password inputs are transient widget state; discard the credential after auth.
        st.session_state.pop("login_password", None)
        last_activity = st.session_state.get("last_activity", time.time())
        remembered = st.session_state.get("remember_session", False)
        if not remembered and time.time() - last_activity > SESSION_IDLE_SECONDS:
            st.session_state.clear()
            st.session_state.auth_page = "Login"
            st.info("Your session expired after 30 minutes of inactivity. Please sign in again.")
        else:
            st.session_state.last_activity = time.time()
    if not st.session_state.get("authenticated"):
        show_authentication()
        return
    brand_col, nav_col, user_col = st.columns([1.05, 2.7, 1.55], vertical_alignment="center")
    with brand_col:
        brand(compact=True)
    with nav_col:
        page = st.segmented_control(
            "Workspace",
            options=list(PAGES),
            default="Overview",
            selection_mode="single",
            label_visibility="collapsed",
            key="workspace_page",
            width="stretch",
        ) or "Overview"
    with user_col:
        user_info, logout_col = st.columns([1.45, 1], vertical_alignment="center")
        with user_info:
            st.caption(f"Signed in as {st.session_state.get('user_name', 'User')}")
        with logout_col:
            if st.button("Log out", key="logout", width="stretch"):
                for key in ["authenticated", "user_id", "user_name", "user_email", "remember_session", "last_activity", "generated_csv", "generated_at", "health_check_report"]:
                    st.session_state.pop(key, None)
                st.session_state.auth_page = "Login"
                st.rerun()
    st.markdown('<div class="top-nav-divider"></div>', unsafe_allow_html=True)
    # Dashboard pages and backend operations are dispatched only after the auth gate above.
    PAGES[page]()


if __name__ == "__main__":
    main()
