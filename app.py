"""
app.py — LoanIQ · Premium Loan Intelligence Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoanIQ — Loan Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────── GLOBAL STYLES ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'Outfit', sans-serif !important;
    background: #06080f !important;
    color: #e2e8f0 !important;
}

/* Hide all Streamlit chrome */
#MainMenu, header[data-testid="stHeader"],
footer, [data-testid="stToolbar"],
[data-testid="collapsedControl"],
.stDeployButton, #stDecoration { display: none !important; }

[data-testid="stSidebar"] { display: none !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #06080f; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 2px; }

/* Main content */
.main .block-container {
    max-width: 1280px !important;
    padding: 0 2rem 4rem !important;
    margin: 0 auto !important;
}

/* Inputs */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #0d1117 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem !important;
}

.stSelectbox label, .stNumberInput label {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
}

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #0d1117 !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid #1e2d45 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border-radius: 9px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 8px 18px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
    color: white !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2d45 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: #0d1117 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary { color: #94a3b8 !important; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.03em !important;
    padding: 14px 36px !important;
    cursor: pointer !important;
    box-shadow: 0 4px 24px rgba(14,165,233,0.3) !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(14,165,233,0.45) !important;
}

/* Code blocks */
.stCodeBlock { background: #0d1117 !important; border: 1px solid #1e2d45 !important; }

hr { border-color: #1e2d45 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR    = Path("models")
MODEL_PATH   = MODEL_DIR / "best_model.joblib"
ENCODER_PATH = MODEL_DIR / "ohe.joblib"
REPORT_PATH  = MODEL_DIR / "model_report.joblib"
DATA_PATH    = "loan_approval_dataset.csv"

NUMERIC_COLS = [
    "no_of_dependents", "income_annum", "loan_amount", "loan_term",
    "cibil_score", "residential_assets_value", "commercial_assets_value",
    "luxury_assets_value", "bank_asset_value",
]
CAT_COLS = ["education", "self_employed"]

BG    = "#06080f"
CARD  = "#0d1117"
BORD  = "#1e2d45"
CYAN  = "#0ea5e9"
GREEN = "#10b981"
RED   = "#ef4444"
AMBER = "#f59e0b"
MUTED = "#64748b"

# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()
    return df

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

@st.cache_resource
def load_encoder():
    if ENCODER_PATH.exists():
        return joblib.load(ENCODER_PATH)
    return None

@st.cache_data
def load_report():
    if REPORT_PATH.exists():
        return joblib.load(REPORT_PATH)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────
def stat_card(label, value, delta="", icon="", accent=CYAN):
    delta_html = f'<div style="color:{GREEN if "+" in str(delta) else RED};font-size:0.78rem;font-weight:600;margin-top:6px;">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div style="background:{CARD};border:1px solid {BORD};border-left:3px solid {accent};
                border-radius:16px;padding:24px 26px;height:100%;
                box-shadow:0 2px 20px rgba(0,0,0,0.3);">
        <div style="color:{MUTED};font-size:0.72rem;font-weight:700;letter-spacing:0.1em;
                    text-transform:uppercase;margin-bottom:10px;">{icon} {label}</div>
        <div style="font-size:1.9rem;font-weight:800;color:#f1f5f9;
                    font-family:'JetBrains Mono',monospace;line-height:1;">{value}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)

def section_title(text, sub=""):
    sub_html = f'<p style="color:{MUTED};font-size:0.95rem;margin-top:6px;">{sub}</p>' if sub else ""
    st.markdown(f'<div style="margin:36px 0 20px;"><h2 style="font-size:1.45rem;font-weight:700;color:#f1f5f9;letter-spacing:-0.02em;">{text}</h2>{sub_html}</div>', unsafe_allow_html=True)

def plotly_cfg(fig, title="", height=380):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit", color="#94a3b8", size=12),
        title=dict(text=title, font=dict(size=13, color="#e2e8f0", family="Outfit"), x=0),
        height=height,
        margin=dict(l=8, r=8, t=44 if title else 12, b=8),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        xaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45", zerolinecolor="#1e2d45"),
        yaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45", zerolinecolor="#1e2d45"),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Top nav
# ─────────────────────────────────────────────────────────────────────────────
def topnav():
    model   = load_model()
    encoder = load_encoder()
    report  = load_report()
    dot     = f'<span style="color:{GREEN};">●</span> Live' if (model and encoder) else f'<span style="color:{AMBER};">⚠</span> No Model'
    f1txt  = f"  F1 {report.get('best_f1',0):.3f}" if report else ""

    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
                padding:18px 0 20px;border-bottom:1px solid {BORD};margin-bottom:4px;">
        <div style="display:flex;align-items:center;gap:14px;">
            <div style="width:38px;height:38px;background:linear-gradient(135deg,{CYAN},{CYAN}88);
                        border-radius:10px;display:flex;align-items:center;justify-content:center;
                        font-size:1.25rem;font-weight:800;color:white;">◈</div>
            <div>
                <div style="font-size:1.2rem;font-weight:800;color:#f1f5f9;letter-spacing:-0.03em;">LoanIQ</div>
                <div style="font-size:0.7rem;color:{MUTED};letter-spacing:0.07em;text-transform:uppercase;">Loan Intelligence Platform</div>
            </div>
        </div>
        <div style="font-size:0.8rem;color:{MUTED};">{dot}<span style="font-family:'JetBrains Mono',monospace;color:{CYAN};">{f1txt}</span></div>
    </div>
    """, unsafe_allow_html=True)

    if "page" not in st.session_state:
        st.session_state.page = "Overview"

    pages = [("◉", "Overview"), ("◈", "Explorer"), ("◆", "Predict"), ("◇", "Model")]
    cols  = st.columns([1, 1, 1, 1, 5])
    for i, (ic, p) in enumerate(pages):
        active = st.session_state.page == p
        label  = f"{ic}  {p}"
        if cols[i].button(label, key=f"nav_{p}"):
            st.session_state.page = p
            st.rerun()
        # Active indicator via CSS injection
        if active:
            cols[i].markdown(f"""
            <style>
            div[data-testid="stButton"]:nth-of-type({i+1}) button {{
                background: linear-gradient(135deg, {CYAN}33, {CYAN}18) !important;
                border: 1px solid {CYAN}66 !important;
                color: {CYAN} !important;
                box-shadow: none !important;
            }}
            </style>""", unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    return st.session_state.page

# ─────────────────────────────────────────────────────────────────────────────
# Page: Overview
# ─────────────────────────────────────────────────────────────────────────────
def page_overview():
    df = load_data()

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{CARD} 0%,#0a1628 60%,{BG} 100%);
                border:1px solid {BORD};border-radius:20px;padding:52px 48px;
                margin-bottom:32px;position:relative;overflow:hidden;">
        <div style="position:absolute;top:-60px;right:-60px;width:280px;height:280px;
                    background:radial-gradient(circle,{CYAN}18 0%,transparent 70%);
                    pointer-events:none;"></div>
        <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.14em;
                    text-transform:uppercase;color:{CYAN};margin-bottom:14px;">
            AI-Powered Credit Analysis
        </div>
        <h1 style="font-size:2.8rem;font-weight:800;color:#f1f5f9;
                   letter-spacing:-0.04em;line-height:1.12;margin-bottom:16px;">
            Smarter loan decisions,<br>powered by machine learning.
        </h1>
        <p style="color:{MUTED};font-size:1.05rem;max-width:540px;line-height:1.65;">
            Predict approval outcomes in seconds using CIBIL scores, income profiles,
            and asset data — trained on {len(df):,} real applications.
        </p>
    </div>
    """ if df is not None else f'<div style="background:{CARD};border:1px solid {BORD};border-radius:20px;padding:48px;"><h1 style="color:{RED};">Dataset not found</h1></div>', unsafe_allow_html=True)

    if df is None:
        return

    approved = (df["loan_status"].str.strip().str.lower() == "approved").sum()
    rejected = len(df) - approved

    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_card("Total Applications", f"{len(df):,}",  icon="◈", accent=CYAN)
    with c2: stat_card("Approved", f"{approved:,}", f"+{approved/len(df)*100:.1f}%", icon="✓", accent=GREEN)
    with c3: stat_card("Rejected", f"{rejected:,}", f"-{rejected/len(df)*100:.1f}%", icon="✕", accent=RED)
    with c4: stat_card("Features",  "11", icon="⬡", accent=AMBER)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        fig = go.Figure(go.Pie(
            labels=["Approved", "Rejected"], values=[approved, rejected], hole=0.72,
            marker=dict(colors=[GREEN, RED], line=dict(color=BG, width=3)),
            textinfo="none",
            hovertemplate="%{label}: %{value:,}<extra></extra>",
        ))
        fig.add_annotation(
            text=f"<b>{approved/len(df)*100:.0f}%</b><br><span>Approved</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color="#f1f5f9", family="Outfit"), align="center",
        )
        plotly_cfg(fig, "Approval Split", height=300)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        df_a = df[df["loan_status"].str.strip().str.lower() == "approved"]["cibil_score"].dropna()
        df_r = df[df["loan_status"].str.strip().str.lower() == "rejected"]["cibil_score"].dropna()
        fig  = go.Figure()
        fig.add_trace(go.Histogram(x=df_a, name="Approved", nbinsx=40,
                                   marker_color=GREEN, opacity=0.75))
        fig.add_trace(go.Histogram(x=df_r, name="Rejected", nbinsx=40,
                                   marker_color=RED, opacity=0.75))
        fig.update_layout(barmode="overlay")
        plotly_cfg(fig, "CIBIL Score Distribution by Status", height=300)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    section_title("Dataset Preview", "First 6 rows of the loan dataset")
    st.dataframe(df.head(6), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Explorer
# ─────────────────────────────────────────────────────────────────────────────
def page_explorer():
    df = load_data()
    if df is None:
        st.error("Dataset not found.")
        return

    section_title("Data Explorer", "Interactive visualisation of applicant features")
    tab1, tab2, tab3 = st.tabs(["  Feature Distributions  ", "  CIBIL Deep Dive  ", "  Correlation Matrix  "])

    num_features = [c for c in NUMERIC_COLS if c in df.columns]

    with tab1:
        feat   = st.selectbox("Select a numeric feature", num_features, key="exp_feat")
        df_plt = df.copy()
        df_plt["status"] = df_plt["loan_status"].str.strip().str.lower()

        fig = go.Figure()
        for status, color in [("approved", GREEN), ("rejected", RED)]:
            vals = df_plt[df_plt["status"] == status][feat].dropna()
            fig.add_trace(go.Violin(
                y=vals, name=status.capitalize(), box_visible=True, meanline_visible=True,
                fillcolor="rgba(16,185,129,0.27)" if color == GREEN else "rgba(239,68,68,0.27)", line_color=color, opacity=0.85,
            ))
        plotly_cfg(fig, f"{feat.replace('_',' ').title()} — Violin Plot", height=400)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        c1, c2 = st.columns(2)
        for i, cat in enumerate([c for c in CAT_COLS if c in df.columns]):
            df_grp = df.groupby([cat, "loan_status"]).size().reset_index(name="count")
            df_grp["loan_status"] = df_grp["loan_status"].str.strip()
            fig = px.bar(df_grp, x=cat, y="count", color="loan_status", barmode="group",
                         color_discrete_map={"Approved": GREEN, "Rejected": RED},
                         labels={"count": "Applications", cat: cat.replace("_", " ").title()})
            plotly_cfg(fig, f"{cat.replace('_',' ').title()} Breakdown", height=320)
            (c1 if i % 2 == 0 else c2).plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab2:
        df_c   = df.copy()
        df_c["status"] = df_c["loan_status"].str.strip()
        bins   = [300, 400, 500, 600, 700, 800, 900]
        labels = ["300–400", "400–500", "500–600", "600–700", "700–800", "800–900"]
        df_c["cibil_band"] = pd.cut(df_c["cibil_score"], bins=bins, labels=labels)
        rate = (
            df_c.groupby("cibil_band", observed=False)
            .apply(lambda x: (x["status"].str.lower() == "approved").mean() * 100)
            .reset_index(name="rate")
        )
        fig = go.Figure(go.Bar(
            x=rate["cibil_band"], y=rate["rate"],
            marker=dict(color=rate["rate"],
                        colorscale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#10b981"]],
                        showscale=False, line=dict(width=0)),
            text=rate["rate"].round(1).astype(str) + "%", textposition="outside",
            hovertemplate="Band %{x}<br>Approval Rate: %{y:.1f}%<extra></extra>",
        ))
        plotly_cfg(fig, "Approval Rate by CIBIL Score Band", height=380)
        fig.update_layout(yaxis=dict(range=[0, 115], ticksuffix="%"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        sample = df_c.sample(min(800, len(df_c)), random_state=42)
        fig = px.scatter(sample, x="cibil_score", y="income_annum", color="status",
                         color_discrete_map={"Approved": GREEN, "Rejected": RED}, opacity=0.65,
                         labels={"cibil_score": "CIBIL Score", "income_annum": "Annual Income (₹)"})
        plotly_cfg(fig, "CIBIL Score vs Annual Income (sample)", height=360)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab3:
        num_df = df[[c for c in num_features]].copy()
        corr   = num_df.corr()
        mask   = np.triu(np.ones_like(corr, dtype=bool), k=1)
        corr_m = corr.where(~mask)
        fig = go.Figure(go.Heatmap(
            z=corr_m.values,
            x=[c.replace("_", " ") for c in corr.columns],
            y=[c.replace("_", " ") for c in corr.columns],
            colorscale=[[0.0,"#ef4444"],[0.5,"#1e2d45"],[1.0,"#0ea5e9"]],
            zmid=0,
            text=corr_m.round(2).values, texttemplate="%{text}",
            hovertemplate="%{x} × %{y}<br>r = %{z:.3f}<extra></extra>",
            colorbar=dict(tickfont=dict(color=MUTED), bgcolor="rgba(0,0,0,0)"),
        ))
        plotly_cfg(fig, "Correlation Matrix (lower triangle)", height=500)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# Page: Predict
# ─────────────────────────────────────────────────────────────────────────────
def page_predict():
    model = load_model()

    st.markdown(f"""
    <div style="margin-bottom:32px;">
        <h1 style="font-size:2.2rem;font-weight:800;color:#f1f5f9;letter-spacing:-0.03em;">
            Predict Loan Outcome
        </h1>
        <p style="color:{MUTED};font-size:1rem;margin-top:8px;">
            Fill in the applicant profile below to get an instant AI prediction.
        </p>
    </div>""", unsafe_allow_html=True)

    if model is None:
        st.markdown(f"""
        <div style="background:#1a0f0f;border:1px solid {RED}44;border-left:3px solid {RED};
                    border-radius:14px;padding:24px 28px;">
            <div style="font-weight:700;color:{RED};margin-bottom:8px;">Model not loaded</div>
            <div style="color:{MUTED};">Run <code style="color:{CYAN};background:{CARD};
            padding:2px 8px;border-radius:4px;">python train.py</code> then refresh.</div>
        </div>""", unsafe_allow_html=True)
        return

    def field_group(title, icon):
        st.markdown(f'<div style="font-size:0.72rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{CYAN};margin:28px 0 12px;">{icon} {title}</div>', unsafe_allow_html=True)

    with st.form("predict_form"):
        field_group("Applicant Profile", "◈")
        c1, c2, c3 = st.columns(3)
        dependents    = c1.number_input("Dependents", 0, 10, 2)
        education     = c2.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = c3.selectbox("Self Employed", ["No", "Yes"])

        field_group("Financial Details", "◆")
        c1, c2, c3, c4 = st.columns(4)
        income      = c1.number_input("Annual Income (₹)", 0, 10_000_000, 500_000, step=10_000)
        loan_amount = c2.number_input("Loan Amount (₹)",   0, 50_000_000, 1_000_000, step=50_000)
        loan_term   = c3.number_input("Loan Term (months)", 1, 360, 60)
        cibil       = c4.number_input("CIBIL Score", 300, 900, 650)

        field_group("Asset Valuations (₹)", "◇")
        c1, c2, c3, c4 = st.columns(4)
        res  = c1.number_input("Residential", 0, 50_000_000, 1_000_000, step=100_000)
        comm = c2.number_input("Commercial",  0, 50_000_000, 500_000,   step=100_000)
        lux  = c3.number_input("Luxury",      0, 50_000_000, 200_000,   step=100_000)
        bank = c4.number_input("Bank Assets", 0, 50_000_000, 300_000,   step=100_000)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Run Prediction  →", use_container_width=True)

    if submitted:
        # Build raw dataframe
        input_df = pd.DataFrame([{
            "no_of_dependents": dependents, "education": education,
            "self_employed": self_employed, "income_annum": income,
            "loan_amount": loan_amount, "loan_term": loan_term,
            "cibil_score": cibil,
            "residential_assets_value": res, "commercial_assets_value": comm,
            "luxury_assets_value": lux, "bank_asset_value": bank,
        }])

        # Preprocess with saved encoder (same as train.py — no ColumnTransformer)
        encoder   = load_encoder()
        cat_cols  = ["education", "self_employed"]
        num_cols  = ["no_of_dependents", "income_annum", "loan_amount", "loan_term",
                     "cibil_score", "residential_assets_value", "commercial_assets_value",
                     "luxury_assets_value", "bank_asset_value"]
        X_cat     = encoder.transform(input_df[cat_cols])
        X_num     = input_df[num_cols].values.astype(float)
        X_input   = np.hstack([X_cat, X_num])

        prediction = model.predict(X_input)[0]
        proba      = model.predict_proba(X_input)[0]
        classes    = model.named_steps["clf"].classes_
        prob_dict  = {k.lower(): v for k, v in zip(classes, proba)}
        p_appr     = prob_dict.get("approved", max(proba))
        p_rej      = 1 - p_appr
        approved   = prediction.lower() == "approved"
        accent     = GREEN if approved else RED
        verdict    = "Approved ✓" if approved else "Rejected ✕"
        conf       = p_appr if approved else p_rej

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{accent}18,{accent}08);
                    border:1px solid {accent}55;border-left:4px solid {accent};
                    border-radius:20px;padding:36px 40px;
                    display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:24px;">
            <div>
                <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.12em;
                            text-transform:uppercase;color:{accent};margin-bottom:10px;">Decision</div>
                <div style="font-size:3rem;font-weight:800;color:#f1f5f9;letter-spacing:-0.04em;line-height:1;">
                    {verdict}</div>
                <div style="margin-top:12px;color:{MUTED};font-size:0.95rem;">
                    Confidence: <span style="color:#f1f5f9;font-weight:700;font-family:'JetBrains Mono',monospace;">{conf*100:.1f}%</span>
                </div>
            </div>
            <div style="display:flex;gap:32px;align-items:center;">
                <div style="text-align:center;">
                    <div style="font-size:2rem;font-weight:800;color:{GREEN};font-family:'JetBrains Mono',monospace;">{p_appr*100:.1f}%</div>
                    <div style="font-size:0.75rem;color:{MUTED};margin-top:4px;text-transform:uppercase;letter-spacing:0.06em;">Approved</div>
                </div>
                <div style="width:1px;height:48px;background:{BORD};"></div>
                <div style="text-align:center;">
                    <div style="font-size:2rem;font-weight:800;color:{RED};font-family:'JetBrains Mono',monospace;">{p_rej*100:.1f}%</div>
                    <div style="font-size:0.75rem;color:{MUTED};margin-top:4px;text-transform:uppercase;letter-spacing:0.06em;">Rejected</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p_appr * 100,
            number=dict(suffix="%", font=dict(size=30, family="JetBrains Mono", color="#f1f5f9")),
            gauge=dict(
                axis=dict(range=[0,100], tickfont=dict(color=MUTED, size=11),
                          tickvals=[0,25,50,75,100]),
                bar=dict(color=accent, thickness=0.25),
                bgcolor="rgba(0,0,0,0)", borderwidth=0,
                steps=[
                    dict(range=[0,40],   color="rgba(239,68,68,0.13)"),
                    dict(range=[40,60],  color="rgba(245,158,11,0.13)"),
                    dict(range=[60,100], color="rgba(16,185,129,0.13)"),
                ],
                threshold=dict(line=dict(color=accent, width=3), thickness=0.75, value=p_appr*100),
            ),
            title=dict(text="Approval Probability", font=dict(size=12, color=MUTED, family="Outfit")),
        ))
        plotly_cfg(fig, height=260)
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORD};border-radius:16px;padding:24px 28px;margin-top:8px;">
            <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
                        color:{MUTED};margin-bottom:16px;">Submitted Profile Summary</div>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;">
                {"".join(f'<div><div style="color:{MUTED};font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;">{k}</div><div style="color:#f1f5f9;font-weight:700;font-family:JetBrains Mono,monospace;font-size:0.9rem;margin-top:4px;">{v}</div></div>' for k, v in [("CIBIL Score", cibil), ("Annual Income", f"₹{income:,}"), ("Loan Amount", f"₹{loan_amount:,}"), ("Loan Term", f"{loan_term} mo")])}
            </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Model
# ─────────────────────────────────────────────────────────────────────────────
def page_model():
    report = load_report()
    if report is None:
        st.markdown(f"""
        <div style="background:#1a0f0f;border:1px solid {RED}44;border-left:3px solid {RED};
                    border-radius:14px;padding:24px 28px;">
            <div style="font-weight:700;color:{RED};">No model report found.</div>
            <div style="color:{MUTED};margin-top:6px;">Run <code style="color:{CYAN};">python train.py</code> first.</div>
        </div>""", unsafe_allow_html=True)
        return

    section_title("Model Intelligence", "Training results, metrics and feature analysis")

    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_card("Best Model",  report.get("best_model_name","—").split("(")[0].strip(), icon="◈", accent=CYAN)
    with c2: stat_card("F1 (macro)",  f"{report.get('best_f1',0):.4f}", icon="◆", accent=GREEN)
    with c3: stat_card("CV F1 Mean",  f"{report.get('cv_f1_mean',0):.4f}", icon="◇", accent=AMBER)
    with c4: stat_card("CV Std",      f"±{report.get('cv_f1_std',0):.4f}", icon="≈", accent=MUTED)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["  Confusion Matrices  ", "  Feature Importance  ", "  Learning Curve  "])

    with tab1:
        cms    = report.get("confusion_matrices", {})
        labels = report.get("labels", ["Approved", "Rejected"])
        if cms:
            cols = st.columns(len(cms))
            for i, (name, cm) in enumerate(cms.items()):
                arr = np.array(cm)
                fig = go.Figure(go.Heatmap(
                    z=arr, x=[f"Pred {l}" for l in labels], y=[f"True {l}" for l in labels],
                    colorscale=[[0, CARD], [1, CYAN]],
                    text=arr, texttemplate="%{text}",
                    textfont=dict(size=18, family="JetBrains Mono"),
                    showscale=False,
                ))
                plotly_cfg(fig, name, height=320)
                cols[i].plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        for name, rep in report.get("model_reports", {}).items():
            with st.expander(f"Full Classification Report — {name}"):
                st.code(rep, language=None)

    with tab2:
        fi = report.get("feature_importances")
        if fi is None:
            st.info("Feature importances only available for tree-based models.")
        else:
            fi_s = pd.Series(fi).sort_values(ascending=True).tail(12)
            fig  = go.Figure(go.Bar(
                x=fi_s.values, y=[f.replace("_"," ").title() for f in fi_s.index],
                orientation="h",
                marker=dict(color=fi_s.values,
                            colorscale=[[0,"#1e2d45"],[1,CYAN]],
                            showscale=False, line=dict(width=0)),
                text=fi_s.round(4).values, textposition="outside",
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ))
            plotly_cfg(fig, "Feature Importance (Top 12)", height=440)
            fig.update_layout(xaxis=dict(tickformat=".3f"), margin=dict(l=4, r=60))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab3:
        lc = report.get("learning_curve")
        if lc is None:
            st.info("Learning curve data not available.")
        else:
            sizes = lc["train_sizes"]
            t_m, t_s = np.array(lc["train_mean"]), np.array(lc["train_std"])
            v_m, v_s = np.array(lc["test_mean"]),  np.array(lc["test_std"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(sizes)+list(sizes[::-1]),
                y=list(t_m+t_s)+list((t_m-t_s)[::-1]),
                fill="toself", fillcolor="rgba(14,165,233,0.13)", line=dict(width=0), showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=list(sizes)+list(sizes[::-1]),
                y=list(v_m+v_s)+list((v_m-v_s)[::-1]),
                fill="toself", fillcolor="rgba(16,185,129,0.13)", line=dict(width=0), showlegend=False,
            ))
            fig.add_trace(go.Scatter(x=sizes, y=t_m, name="Training F1",
                                     line=dict(color=CYAN, width=2.5),
                                     mode="lines+markers", marker=dict(size=7)))
            fig.add_trace(go.Scatter(x=sizes, y=v_m, name="Validation F1",
                                     line=dict(color=GREEN, width=2.5, dash="dot"),
                                     mode="lines+markers", marker=dict(size=7)))
            plotly_cfg(fig, "Learning Curve — Best Model", height=400)
            fig.update_layout(
                xaxis=dict(title="Training Set Size"),
                yaxis=dict(title="F1 Score (macro)", range=[max(0, v_m.min()-0.05), 1.02]),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
def main():
    topnav()
    page = st.session_state.get("page", "Overview")
    if   page == "Overview": page_overview()
    elif page == "Explorer": page_explorer()
    elif page == "Predict":  page_predict()
    elif page == "Model":    page_model()

if __name__ == "__main__":
    main()