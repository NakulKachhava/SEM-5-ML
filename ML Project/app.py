import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib

# ---------------------------------------------------------
# Page Configuration & Modern Dark Theme Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="RiskPulse AI - Loan Default Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Sleek Dark Aesthetic (#0e1117 background, #1e222a cards, neon cyan/indigo accents)
st.markdown(
    """
    <style>
    /* Global Styles & Dark Theme */
    .stApp {
        background-color: #0e1117;
        color: #e2e8f0;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Card Container Component */
    .css-card {
        background: #1e222a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Hero Banner Header */
    .hero-banner {
        background: linear-gradient(135deg, #1e222a 0%, #0f172a 100%);
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 16px;
        padding: 30px 35px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 50%, #667eea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
    }

    /* Custom Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .badge-success {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .badge-danger {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .badge-info {
        background-color: rgba(79, 172, 254, 0.15);
        color: #4facfe;
        border: 1px solid rgba(79, 172, 254, 0.3);
    }

    /* Metric Cards */
    .metric-card {
        background: #181c24;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }

    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #00f2fe;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Style Form Submit Button Specifically */
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%) !important;
        color: #0e1117 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.4) !important;
        color: #0e1117 !important;
    }

    /* Streamlit Widget Overrides */
    .stButton>button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: #0e1117;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.4);
        color: #0e1117;
    }

    /* Sidebar Background */
    section[data-testid="stSidebar"] {
        background-color: #131720;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# ML Model & Scaler Pipeline (Cached)
# ---------------------------------------------------------
FEATURE_COLUMNS = [
    'Age',
    'Income',
    'LoanAmount',
    'CreditScore',
    'MonthsEmployed',
    'NumCreditLines',
    'InterestRate',
    'LoanTerm',
    'DTIRatio',
    "Education_High School",
    "Education_Master's",
    'Education_PhD',
    'EmploymentType_Part-time',
    'EmploymentType_Self-employed',
    'EmploymentType_Unemployed',
    'MaritalStatus_Married',
    'MaritalStatus_Single',
    'HasMortgage_Yes',
    'HasDependents_Yes',
    'LoanPurpose_Business',
    'LoanPurpose_Education',
    'LoanPurpose_Home',
    'LoanPurpose_Other',
    'HasCoSigner_Yes',
]

MODEL_FILE = "loan_model.joblib"
SCALER_FILE = "scaler.joblib"
DATA_FILE = "Loan_default.csv"


@st.cache_resource(show_spinner=False)
def load_or_train_model():
    """Loads cached model & scaler or fits them on Loan_default.csv if not present."""
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        return model, scaler

    # Fallback to training on Loan_default.csv
    if not os.path.exists(DATA_FILE):
        st.error(
            f"Dataset '{DATA_FILE}' not found. Please ensure the CSV is in the root directory."
        )
        st.stop()

    df = pd.read_csv(DATA_FILE)
    if "LoanID" in df.columns:
        df = df.drop(columns=["LoanID"])

    X = df.drop(columns=["Default"])
    y = df["Default"]

    X_encoded = pd.get_dummies(X, drop_first=True)
    X_encoded = X_encoded.apply(
        lambda col: col.astype(int) if col.dtype == "bool" else col
    )

    # Ensure all feature columns match expected schema
    for col in FEATURE_COLUMNS:
        if col not in X_encoded.columns:
            X_encoded[col] = 0
    X_encoded = X_encoded[FEATURE_COLUMNS]

    X_train, _, y_train, _ = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    return model, scaler


with st.spinner("Initializing RiskPulse AI Model Engine..."):
    model, scaler = load_or_train_model()

# ---------------------------------------------------------
# Sidebar - Branding, Metadata & Config
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 15px 0;">
            <div style="font-size: 2.2rem;">🏦</div>
            <div style="font-size: 1.4rem; font-weight: 800; color: #00f2fe; letter-spacing: 0.5px;">RiskPulse AI</div>
            <div style="font-size: 0.8rem; color: #94a3b8;">Automated Credit Risk Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="status-badge badge-success" style="width: 100%; text-align: center; margin-bottom: 15px;">● Model Loaded & Ready</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 📊 Model Summary")
    st.markdown(
        """
        <div style="background: #181c24; border-radius: 8px; padding: 12px; font-size: 0.85rem; border: 1px solid rgba(255,255,255,0.05);">
            <div><b>Algorithm:</b> Logistic Regression</div>
            <div><b>Test Accuracy:</b> <span style="color: #10b981; font-weight: bold;">88.58%</span></div>
            <div><b>Training Data:</b> 255,347 Records</div>
            <div><b>Encoded Features:</b> 24 Dimensions</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### ⚙️ Risk Threshold Config")
    risk_threshold = st.slider(
        "Classification Risk Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Adjust the probability threshold above which an applicant is classified as High Risk (Default). Default is 0.50.",
    )

    st.markdown("---")
    st.markdown("### ⚡ Quick Presets")
    st.caption("Load pre-configured sample applicant profiles:")

    col_p1, col_p2 = st.columns(2)
    load_low_risk = col_p1.button("🟢 Low Risk")
    load_high_risk = col_p2.button("🔴 High Risk")

# ---------------------------------------------------------
# State Management for Presets & Evaluation Trigger
# ---------------------------------------------------------
if "preset_data" not in st.session_state:
    st.session_state.preset_data = {}

if "has_evaluated" not in st.session_state:
    st.session_state.has_evaluated = False

if load_low_risk:
    st.session_state.preset_data = {
        "age": 52,
        "education": "Master's",
        "employment_type": "Full-time",
        "months_employed": 96,
        "marital_status": "Married",
        "has_dependents": "No",
        "income": 135000,
        "credit_score": 780,
        "dti_ratio": 0.22,
        "has_mortgage": "No",
        "has_cosigner": "Yes",
        "loan_amount": 35000,
        "interest_rate": 6.5,
        "loan_term": 36,
        "loan_purpose": "Home",
    }
    st.session_state.has_evaluated = True

elif load_high_risk:
    st.session_state.preset_data = {
        "age": 22,
        "education": "High School",
        "employment_type": "Unemployed",
        "months_employed": 4,
        "marital_status": "Single",
        "has_dependents": "Yes",
        "income": 22000,
        "credit_score": 460,
        "dti_ratio": 0.72,
        "has_mortgage": "Yes",
        "has_cosigner": "No",
        "loan_amount": 160000,
        "interest_rate": 22.5,
        "loan_term": 60,
        "loan_purpose": "Other",
    }
    st.session_state.has_evaluated = True

defaults = st.session_state.preset_data

# ---------------------------------------------------------
# Main Canvas - Hero Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">Loan Default Risk Intelligence Dashboard</div>
        <div class="hero-subtitle">Configure applicant parameters and click "Evaluate Loan Risk Profile" to run default risk prediction.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Input Form - Enclosed in st.form to Prevent Auto-Prediction
# ---------------------------------------------------------
st.markdown("### 📝 Applicant & Loan Evaluation Form")

with st.form("loan_evaluation_form"):
    tab1, tab2, tab3 = st.tabs(
        [
            "👤 Applicant Profile",
            "💵 Financial & Credit Health",
            "🏦 Loan Specifications",
        ]
    )

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input(
                "Age (Years)",
                min_value=18,
                max_value=100,
                value=defaults.get("age", 40),
                help="Applicant's current age. Older applicants generally show higher repayment stability.",
            )
            education = st.selectbox(
                "Education Level",
                options=["High School", "Bachelor's", "Master's", "PhD"],
                index=["High School", "Bachelor's", "Master's", "PhD"].index(
                    defaults.get("education", "Bachelor's")
                ),
                help="Highest completed degree.",
            )
        with col2:
            employment_type = st.selectbox(
                "Employment Type",
                options=[
                    "Full-time",
                    "Part-time",
                    "Self-employed",
                    "Unemployed",
                ],
                index=[
                    "Full-time",
                    "Part-time",
                    "Self-employed",
                    "Unemployed",
                ].index(defaults.get("employment_type", "Full-time")),
                help="Current employment status.",
            )
            months_employed = st.number_input(
                "Months Employed",
                min_value=0,
                max_value=120,
                value=defaults.get("months_employed", 48),
                help="Duration at current job or career in months.",
            )
        with col3:
            marital_status = st.selectbox(
                "Marital Status",
                options=["Divorced", "Married", "Single"],
                index=["Divorced", "Married", "Single"].index(
                    defaults.get("marital_status", "Married")
                ),
                help="Marital status of applicant.",
            )
            has_dependents = st.selectbox(
                "Has Dependents?",
                options=["No", "Yes"],
                index=["No", "Yes"].index(defaults.get("has_dependents", "No")),
                help="Indicates if the applicant has financially dependent family members.",
            )

    with tab2:
        col1, col2, col3 = st.columns(3)
        with col1:
            income = st.number_input(
                "Annual Income ($)",
                min_value=10000,
                max_value=500000,
                value=defaults.get("income", 75000),
                step=5000,
                help="Total gross annual income in USD.",
            )
            credit_score = st.slider(
                "Credit Score (FICO)",
                min_value=300,
                max_value=850,
                value=defaults.get("credit_score", 680),
                help="FICO credit score ranging from 300 (Poor) to 850 (Exceptional).",
            )
        with col2:
            dti_ratio = st.slider(
                "Debt-to-Income (DTI) Ratio",
                min_value=0.0,
                max_value=1.0,
                value=float(defaults.get("dti_ratio", 0.35)),
                step=0.01,
                help="Ratio of total monthly debt payments to gross monthly income.",
            )
            has_mortgage = st.selectbox(
                "Has Existing Mortgage?",
                options=["No", "Yes"],
                index=["No", "Yes"].index(defaults.get("has_mortgage", "No")),
                help="Indicates active mortgage obligation.",
            )
        with col3:
            has_cosigner = st.selectbox(
                "Has Co-Signer?",
                options=["No", "Yes"],
                index=["No", "Yes"].index(defaults.get("has_cosigner", "Yes")),
                help="Having a backup co-signer significantly reduces lender risk.",
            )

    with tab3:
        col1, col2, col3 = st.columns(3)
        with col1:
            loan_amount = st.number_input(
                "Requested Loan Amount ($)",
                min_value=1000,
                max_value=500000,
                value=defaults.get("loan_amount", 50000),
                step=2500,
                help="Principal amount requested by borrower.",
            )
            interest_rate = st.number_input(
                "Interest Rate (%)",
                min_value=1.0,
                max_value=30.0,
                value=float(defaults.get("interest_rate", 12.5)),
                step=0.25,
                help="Annual interest rate for the requested loan.",
            )
        with col2:
            loan_term = st.selectbox(
                "Loan Term (Months)",
                options=[12, 24, 36, 48, 60],
                index=[12, 24, 36, 48, 60].index(
                    defaults.get("loan_term", 36)
                ),
                help="Duration of the loan repayment period.",
            )
        with col3:
            loan_purpose = st.selectbox(
                "Loan Purpose",
                options=["Auto", "Business", "Education", "Home", "Other"],
                index=["Auto", "Business", "Education", "Home", "Other"].index(
                    defaults.get("loan_purpose", "Business")
                ),
                help="Intended usage of loan funds.",
            )

    st.markdown("<br>", unsafe_allow_html=True)
    evaluate_submitted = st.form_submit_button("🔍 Evaluate Loan Risk Profile")

# Derived Financial Indicators Bar
lti_ratio = loan_amount / income if income > 0 else 0
monthly_rate = (interest_rate / 100) / 12
est_monthly_payment = (
    (loan_amount * monthly_rate * ((1 + monthly_rate) ** loan_term))
    / (((1 + monthly_rate) ** loan_term) - 1)
    if monthly_rate > 0
    else loan_amount / loan_term
)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("##### 💡 Derived Financial Indicators")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">${est_monthly_payment:,.2f}</div>
            <div class="metric-label">Est. Monthly Payment</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{lti_ratio:.2f}x</div>
            <div class="metric-label">Loan-to-Income (LTI)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m3:
    credit_tier = (
        "Excellent 🌟"
        if credit_score >= 750
        else (
            "Good 👍"
            if credit_score >= 670
            else "Fair ⚠️" if credit_score >= 580 else "Poor 🚨"
        )
    )
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{credit_tier}</div>
            <div class="metric-label">Credit Bracket</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m4:
    dti_status = "Healthy" if dti_ratio <= 0.36 else "Stretched"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{dti_status}</div>
            <div class="metric-label">DTI Assessment</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Feature Construction & Prediction Execution
# ---------------------------------------------------------
if evaluate_submitted:
    st.session_state.has_evaluated = True

if st.session_state.has_evaluated:
    input_dict = {
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": 2,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti_ratio,
        "Education_High School": 1 if education == "High School" else 0,
        "Education_Master's": 1 if education == "Master's" else 0,
        "Education_PhD": 1 if education == "PhD" else 0,
        "EmploymentType_Part-time": 1 if employment_type == "Part-time" else 0,
        "EmploymentType_Self-employed": (
            1 if employment_type == "Self-employed" else 0
        ),
        "EmploymentType_Unemployed": (
            1 if employment_type == "Unemployed" else 0
        ),
        "MaritalStatus_Married": 1 if marital_status == "Married" else 0,
        "MaritalStatus_Single": 1 if marital_status == "Single" else 0,
        "HasMortgage_Yes": 1 if has_mortgage == "Yes" else 0,
        "HasDependents_Yes": 1 if has_dependents == "Yes" else 0,
        "LoanPurpose_Business": 1 if loan_purpose == "Business" else 0,
        "LoanPurpose_Education": 1 if loan_purpose == "Education" else 0,
        "LoanPurpose_Home": 1 if loan_purpose == "Home" else 0,
        "LoanPurpose_Other": 1 if loan_purpose == "Other" else 0,
        "HasCoSigner_Yes": 1 if has_cosigner == "Yes" else 0,
    }

    input_df = pd.DataFrame([input_dict])[FEATURE_COLUMNS]

    input_scaled = scaler.transform(input_df)
    proba = model.predict_proba(input_scaled)[0]
    default_proba = proba[1]
    non_default_proba = proba[0]
    is_high_risk = default_proba >= risk_threshold

    st.markdown("---")
    st.markdown("### 📊 Predictive Intelligence & Risk Analysis")

    # Results Header Card
    res_col1, res_col2 = st.columns([1.2, 1])

    with res_col1:
        if is_high_risk:
            st.markdown(
                f"""
                <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 12px; padding: 24px; text-align: left;">
                    <div style="display: flex; align-items: center; gap: 16px;">
                        <div style="font-size: 2.2rem; line-height: 1;">🚨</div>
                        <div>
                            <div style="font-size: 1.5rem; font-weight: 800; color: #ef4444; line-height: 1.2;">HIGH RISK - POTENTIAL DEFAULT</div>
                            <div style="color: #cbd5e1; font-size: 0.95rem; margin-top: 6px;">
                                Applicant exhibits elevated risk indicators exceeding the {risk_threshold*100:.0f}% classification threshold.
                            </div>
                        </div>
                    </div>
                    <hr style="border: 0; border-top: 1px solid rgba(239, 68, 68, 0.2); margin: 20px 0;">
                    <div style="display: flex; flex-wrap: wrap; gap: 16px; font-size: 0.95rem;">
                        <div style="flex: 1; min-width: 250px;">
                            <span style="color: #cbd5e1;">Estimated Default Probability:</span> 
                            <span style="color: #ef4444; font-weight: bold; margin-left: 4px;">{default_proba*100:.2f}%</span>
                        </div>
                        <div style="flex: 1; min-width: 280px; text-align: left;">
                            <span style="color: #cbd5e1;">Underwriting Recommendation:</span> 
                            <span style="color: #ef4444; font-weight: bold; margin-left: 4px; white-space: nowrap;">REJECT / MANUAL AUDIT</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 12px; padding: 24px; text-align: left;">
                    <div style="display: flex; align-items: center; gap: 16px;">
                        <div style="font-size: 2.2rem; line-height: 1;">✅</div>
                        <div>
                            <div style="font-size: 1.5rem; font-weight: 800; color: #10b981; line-height: 1.2;">LOW RISK - APPROVED FOR PROCESSING</div>
                            <div style="color: #cbd5e1; font-size: 0.95rem; margin-top: 6px;">
                                Applicant profile is financially sound and well within safety tolerance parameters.
                            </div>
                        </div>
                    </div>
                    <hr style="border: 0; border-top: 1px solid rgba(16, 185, 129, 0.2); margin: 20px 0;">
                    <div style="display: flex; flex-wrap: wrap; gap: 16px; font-size: 0.95rem;">
                        <div style="flex: 1; min-width: 250px;">
                            <span style="color: #cbd5e1;">Estimated Repayment Probability:</span> 
                            <span style="color: #10b981; font-weight: bold; margin-left: 4px;">{non_default_proba*100:.2f}%</span>
                        </div>
                        <div style="flex: 1; min-width: 280px; text-align: left;">
                            <span style="color: #cbd5e1;">Underwriting Recommendation:</span> 
                            <span style="color: #10b981; font-weight: bold; margin-left: 4px; white-space: nowrap;">FAST-TRACK APPROVAL</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with res_col2:
        # Gauge Chart for Risk Probability
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=default_proba * 100,
                number={
                    "suffix": "%",
                    "font": {"color": "#00f2fe", "size": 36},
                },
                title={
                    "text": "Default Probability Index",
                    "font": {"color": "#e2e8f0", "size": 16},
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": "#94a3b8",
                    },
                    "bar": {
                        "color": (
                            "#ef4444"
                            if default_proba >= risk_threshold
                            else "#10b981"
                        )
                    },
                    "bgcolor": "#1e222a",
                    "borderwidth": 1,
                    "bordercolor": "rgba(255,255,255,0.1)",
                    "steps": [
                        {
                            "range": [0, risk_threshold * 100],
                            "color": "rgba(16, 185, 129, 0.15)",
                        },
                        {
                            "range": [risk_threshold * 100, 100],
                            "color": "rgba(239, 68, 68, 0.15)",
                        },
                    ],
                    "threshold": {
                        "line": {"color": "#00f2fe", "width": 3},
                        "thickness": 0.8,
                        "value": risk_threshold * 100,
                    },
                },
            )
        )
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20),
            height=220,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ---------------------------------------------------------
    # Feature Contribution Analysis (Model Explainability)
    # ---------------------------------------------------------
    st.markdown("#### 🔍 Applicant Risk Contribution Breakdown")
    st.caption(
        "Relative impact of applicant parameters on the default risk score based on Logistic Regression feature weights:"
    )

    coefs = model.coef_[0]
    scaled_val = input_scaled[0]
    feature_contributions = coefs * scaled_val

    contrib_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Impact": feature_contributions,
            "Direction": [
                "Increases Risk 📈" if x > 0 else "Lowers Risk 📉"
                for x in feature_contributions
            ],
        }
    ).sort_values(by="Impact", key=abs, ascending=False)

    # Select Top 10 most impactful features for visualization
    top_contrib = contrib_df.head(10).sort_values(by="Impact", ascending=True)

    fig_bar = px.bar(
        top_contrib,
        x="Impact",
        y="Feature",
        orientation="h",
        color="Impact",
        color_continuous_scale=[
            [0.0, "#10b981"],
            [0.5, "#4facfe"],
            [1.0, "#ef4444"],
        ],
        title="Top 10 Feature Drivers (Negative = Protects Applicant, Positive = Increases Default Risk)",
    )

    fig_bar.update_layout(
        paper_bgcolor="#181c24",
        plot_bgcolor="#181c24",
        font=dict(color="#e2e8f0"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)", title="Net Impact Score"
        ),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Feature"),
        coloraxis_showscale=False,
        height=360,
        margin=dict(l=10, r=10, t=40, b=20),
    )

    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.markdown("---")
    st.info(
        "👆 Configure applicant details in the form above and click **'🔍 Evaluate Loan Risk Profile'** to compute prediction and risk analytics."
    )

# Footer & Usage Notice
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.8rem; padding: 10px 0;">
        RiskPulse AI • ML Model Version 1.0 (Logistic Regression 88.58% Acc) • Designed for Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
