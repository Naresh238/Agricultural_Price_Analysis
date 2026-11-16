
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.onion_hybrid_predictor import (
    load_hybrid_results,
    get_market_options,
    get_market_predictions,
    get_selected_model,
    get_data_freshness,
)


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AgriSmart | Price Analysis & Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 2. PROJECT PATHS
# =====================================================

DATA_PATH = "data/processed/agricultural_prices_monthly.csv"

MODEL_PATH = "models/improved_linear_regression_model.pkl"

MODEL_REPORT = "reports/final_model_evaluation.csv"

COMPARISON_REPORT = "reports/model_comparison.csv"

# =====================================================
# 3. PREMIUM CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* Main background */

.stApp {
    background:
        radial-gradient(circle at top right, #dbeafe 0%, transparent 35%),
        linear-gradient(135deg, #ecfdf5, #eff6ff, #f5f3ff);
}

/* Main container */

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Headings */

h1 {
    color: #064e3b !important;
    font-weight: 900 !important;
}

h2 {
    color: #047857 !important;
    font-weight: 800 !important;
}

h3 {
    color: #0369a1 !important;
    font-weight: 750 !important;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #052e16,
        #065f46,
        #0f766e,
        #1d4ed8
    );
}

[data-testid="stSidebar"] * {
    color: white !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 8px;
    transition: 0.3s;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.25);
}

/* Metric cards */

[data-testid="stMetric"] {
    background: linear-gradient(
        135deg,
        #ffffff,
        #d1fae5
    );
    border: 1px solid #86efac;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 8px 25px rgba(5,150,105,0.16);
    transition: all 0.3s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-5px);
    box-shadow: 0 14px 32px rgba(5,150,105,0.28);
}

[data-testid="stMetricLabel"] {
    color: #047857 !important;
    font-weight: 750 !important;
}

[data-testid="stMetricValue"] {
    color: #064e3b !important;
    font-weight: 900 !important;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(
        100deg,
        #059669,
        #0d9488,
        #2563eb
    );
    color: white !important;
    border: none;
    border-radius: 13px;
    padding: 12px 24px;
    font-weight: 800;
    box-shadow: 0 6px 18px rgba(5,150,105,0.25);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 25px rgba(37,99,235,0.30);
}

/* Download buttons */

.stDownloadButton > button {
    background: linear-gradient(
        100deg,
        #2563eb,
        #7c3aed
    );
    color: white !important;
    border: none;
    border-radius: 12px;
    font-weight: 750;
}

/* Data tables */

[data-testid="stDataFrame"] {
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 5px 20px rgba(15,23,42,0.08);
}

/* Input controls */

.stSelectbox, .stTextInput {
    border-radius: 12px;
}

/* Alerts */

[data-testid="stAlert"] {
    border-radius: 14px;
}

/* Dividers */

hr {
    border-color: #86efac !important;
}

/* Hero banner */

.hero {
    background: linear-gradient(
        115deg,
        #052e16,
        #059669,
        #0891b2,
        #2563eb
    );
    padding: 38px;
    border-radius: 24px;
    color: white;
    box-shadow: 0 12px 35px rgba(5,150,105,0.25);
    margin-bottom: 25px;
}

.hero h1 {
    color: white !important;
    font-size: 36px;
    margin-bottom: 10px;
}

.hero p {
    color: #ecfeff;
    font-size: 16px;
}

/* Section cards */

.section-card {
    background: linear-gradient(
        135deg,
        #ffffff,
        #ecfdf5
    );
    padding: 22px;
    border: 1px solid #bbf7d0;
    border-radius: 17px;
    box-shadow: 0 6px 20px rgba(15,118,110,0.10);
    margin-bottom: 15px;
}

/* Footer */

.footer {
    background: linear-gradient(
        100deg,
        #052e16,
        #047857,
        #1d4ed8
    );
    color: white;
    text-align: center;
    padding: 25px;
    border-radius: 18px;
    margin-top: 30px;
    box-shadow: 0 8px 25px rgba(5,150,105,0.20);
}

.footer p {
    color: white;
    margin: 5px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# 4. LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    data = pd.read_csv(DATA_PATH)

    data["Arrival_Date"] = pd.to_datetime(
        data["Arrival_Date"],
        errors="coerce"
    )

    for col in ["Commodity", "Market", "District", "State"]:
        data[col] = data[col].astype("string").str.strip()

    return data


@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


# =====================================================
# 5. CHECK DATASET
# =====================================================

if not os.path.exists(DATA_PATH):

    st.error("Agricultural dataset not found!")

    st.code(DATA_PATH)

    st.stop()

try:
    df = load_data()

except Exception as error:

    st.error(f"Unable to load dataset: {error}")

    st.stop()

if df.empty:

    st.error("The agricultural dataset is empty.")

    st.stop()

# =====================================================
# 6. SIDEBAR
# =====================================================

st.sidebar.markdown(
    """
    <div style="text-align:center;padding:15px;">
        <div style="font-size:55px;">🌾</div>
        <h2 style="color:white !important;">
            AgriSmart
        </h2>
        <p style="color:#d1fae5;">
            Smart Agriculture Analytics
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "MAIN NAVIGATION",
    [
        "Dashboard",
        "Dataset Overview",
        "Commodity Analysis",
        "Price Prediction",
        "Model Performance",
        "Onion Hybrid Prediction"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div style="
        background:rgba(255,255,255,0.12);
        padding:15px;
        border-radius:14px;
    ">
        <h4 style="color:white;">Project Information</h4>
        <p style="color:#d1fae5;">
            Machine Learning Based Agricultural Price Analysis
        </p>
        <p style="color:#d1fae5;">
            B.Tech AI & Data Science
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# 7. HERO HEADER
# =====================================================

st.markdown(
    """
    <div class="hero">
        <h1>🌾 Agricultural Price Analysis</h1>
        <h2 style="color:#d1fae5 !important;">
            Smart Agricultural Price Prediction System
        </h2>
        <p>
            Analyze commodity prices | Explore market trends |
            Discover machine learning predictions
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# PAGE 1: DASHBOARD
# =====================================================

if page == "Dashboard":

    st.header("📊 Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "📁 Total Records",
        f"{len(df):,}"
    )

    col2.metric(
        "🌱 Commodities",
        f"{df['Commodity'].nunique():,}"
    )

    col3.metric(
        "🏪 Markets",
        f"{df['Market'].nunique():,}"
    )

    col4.metric(
        "📍 Districts",
        f"{df['District'].nunique():,}"
    )

    st.divider()

    st.subheader("📅 Historical Dataset Period")

    c1, c2 = st.columns(2)

    c1.metric(
        "Starting Date",
        str(df["Arrival_Date"].min().date())
    )

    c2.metric(
        "Ending Date",
        str(df["Arrival_Date"].max().date())
    )

    st.divider()

    st.subheader("🌿 Top 10 Agricultural Commodities")

    top_commodities = (
        df["Commodity"]
        .value_counts()
        .head(10)
    )

    st.bar_chart(
        top_commodities,
        color="#059669"
    )

    st.divider()

    st.subheader("📋 Recent Dataset Records")

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Historical Dataset",
        df.to_csv(index=False),
        file_name="agricultural_prices_monthly.csv",
        mime="text/csv"
    )

# =====================================================
# PAGE 2: DATASET OVERVIEW
# =====================================================

elif page == "Dataset Overview":

    st.header("📁 Dataset Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Rows", f"{df.shape[0]:,}")

    c2.metric("Total Columns", df.shape[1])

    c3.metric(
        "Missing Values",
        f"{df.isnull().sum().sum():,}"
    )

    st.divider()

    st.subheader("📌 Dataset Column Information")

    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(
        column_info,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("🌱 Commodity Distribution")

    commodity_counts = (
        df["Commodity"]
        .value_counts()
        .head(15)
    )

    st.bar_chart(
        commodity_counts,
        color="#0d9488"
    )

    st.divider()

    st.subheader("🔎 Search Agricultural Records")

    search = st.text_input(
        "Enter commodity name"
    )

    if search:

        filtered_df = df[
            df["Commodity"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

    else:

        filtered_df = df

    st.write(
        f"Matching records: {len(filtered_df):,}"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Filtered Dataset",
        filtered_df.to_csv(index=False),
        file_name="filtered_agricultural_data.csv",
        mime="text/csv"
    )

# =====================================================
# PAGE 3: COMMODITY ANALYSIS
# =====================================================

elif page == "Commodity Analysis":

    st.header("📈 Commodity Price Analysis")

    commodities = sorted(
        df["Commodity"].dropna().unique().tolist()
    )

    selected_commodity = st.selectbox(
        "🌱 Select Commodity",
        commodities
    )

    commodity_df = df[
        df["Commodity"] == selected_commodity
    ]

    districts = sorted(
        commodity_df["District"].dropna().unique().tolist()
    )

    if not districts:

        st.warning("No district data available.")

        st.stop()

    selected_district = st.selectbox(
        "📍 Select District",
        districts
    )

    district_df = commodity_df[
        commodity_df["District"] == selected_district
    ]

    markets = sorted(
        district_df["Market"].dropna().unique().tolist()
    )

    if not markets:

        st.warning("No market data available.")

        st.stop()

    selected_market = st.selectbox(
        "🏪 Select Market",
        markets
    )

    result = district_df[
        district_df["Market"] == selected_market
    ].sort_values("Arrival_Date")

    if not result.empty:

        st.divider()

        st.subheader(
            f"🌾 {selected_commodity} | {selected_market}"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Average Price",
            f"₹{result['Modal_Price'].mean():,.2f}"
        )

        c2.metric(
            "Minimum Price",
            f"₹{result['Modal_Price'].min():,.2f}"
        )

        c3.metric(
            "Maximum Price",
            f"₹{result['Modal_Price'].max():,.2f}"
        )

        st.divider()

        st.subheader("📉 Historical Modal Price Trend")

        chart_data = result.set_index(
            "Arrival_Date"
        )["Modal_Price"]

        st.line_chart(
            chart_data,
            color="#059669"
        )

        st.subheader("📊 Price Statistics")

        st.dataframe(
            result["Modal_Price"].describe().to_frame(
                name="Price Statistics"
            ),
            use_container_width=True
        )

        st.subheader("📋 Market Records")

        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Commodity Analysis",
            result.to_csv(index=False),
            file_name="commodity_analysis.csv",
            mime="text/csv"
        )

# =====================================================
# PAGE 4: PRICE PREDICTION
# =====================================================

elif page == "Price Prediction":

    st.header("🤖 Agricultural Price Prediction")

    st.markdown(
        """
        <div class="section-card">
            <h3>🌾 Machine Learning Prediction</h3>
            <p>
                Predict the next monthly modal price using
                your saved Improved Linear Regression model.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "This model is trained specifically for BAJRI "
        "in Kada market, Beed district, Maharashtra."
    )

    if not os.path.exists(MODEL_PATH):

        st.error("Saved prediction model not found!")

        st.stop()

    bajri_data = df[
        (df["Commodity"].str.upper() == "BAJRI") &
        (df["Market"].str.lower() == "kada") &
        (df["District"].str.lower() == "beed")
    ].sort_values(
        "Arrival_Date"
    ).reset_index(drop=True)

    if len(bajri_data) < 2:

        st.error(
            "Insufficient historical data for prediction."
        )

        st.stop()

    latest = bajri_data.iloc[-1]

    previous = bajri_data.iloc[-2]

    next_month = (
        latest["Arrival_Date"] +
        pd.DateOffset(months=1)
    )

    expected_latest_month = (
        previous["Arrival_Date"] +
        pd.DateOffset(months=1)
    )

    if expected_latest_month != latest["Arrival_Date"]:

        st.error(
            "The latest two observations are not consecutive months."
        )

        st.stop()

    st.subheader("📌 Selected Agricultural Information")

    c1, c2, c3 = st.columns(3)

    c1.metric("Commodity", "BAJRI")

    c2.metric("Market", "Kada")

    c3.metric("District", "Beed")

    st.divider()

    c1, c2 = st.columns(2)

    c1.metric(
        "Last Recorded Price",
        f"₹{latest['Modal_Price']:,.2f}"
    )

    c2.metric(
        "Last Recorded Month",
        latest["Arrival_Date"].strftime("%B %Y")
    )

    st.info(
        f"Next prediction period: {next_month.strftime('%B %Y')}"
    )

    if st.button(
        "✨ Predict Agricultural Price",
        type="primary",
        use_container_width=True
    ):

        try:

            model = load_model()

            prediction_input = pd.DataFrame([{
                "Month_Number": next_month.month,
                "Year": next_month.year,
                "Lag_1": latest["Modal_Price"],
                "Lag_2": previous["Modal_Price"]
            }])

            predicted_price = model.predict(
                prediction_input
            )[0]

            if not np.isfinite(predicted_price):

                raise ValueError(
                    "Invalid prediction generated."
                )

            difference = (
                predicted_price -
                latest["Modal_Price"]
            )

            st.divider()

            st.subheader("🎯 Prediction Result")

            c1, c2 = st.columns(2)

            c1.metric(
                "Previous Modal Price",
                f"₹{latest['Modal_Price']:,.2f}"
            )

            c2.metric(
                "Predicted Modal Price",
                f"₹{predicted_price:,.2f}",
                delta=f"₹{difference:,.2f}"
            )

            st.success(
                f"Prediction completed for "
                f"{next_month.strftime('%B %Y')}!"
            )

            prediction_report = pd.DataFrame({
                "Commodity": ["BAJRI"],
                "Market": ["Kada"],
                "District": ["Beed"],
                "Prediction_Month": [
                    next_month.strftime("%B %Y")
                ],
                "Last_Recorded_Price": [
                    latest["Modal_Price"]
                ],
                "Predicted_Modal_Price": [
                    round(predicted_price, 2)
                ],
                "Price_Unit": ["INR per Quintal"]
            })

            st.download_button(
                "⬇️ Download Prediction Report",
                prediction_report.to_csv(index=False),
                file_name="agricultural_price_prediction.csv",
                mime="text/csv"
            )

            st.warning(
                "Historical demonstration only. "
                "The training dataset ends in November 2016. "
                "This is not a current 2026 market forecast."
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )

# =====================================================
# PAGE 5: MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.header("📊 Machine Learning Model Performance")

    st.write(
        "Compare the forecasting models using MAE and RMSE."
    )

    if not os.path.exists(MODEL_REPORT):

        st.error("Final evaluation report not found!")

        st.stop()

    results = pd.read_csv(MODEL_REPORT)

    required_columns = {"Model", "MAE", "RMSE"}

    if not required_columns.issubset(results.columns):

        st.error("Unexpected model evaluation report format.")

        st.stop()

    st.divider()

    st.subheader("🏆 Final Model Evaluation")

    st.dataframe(
        results.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("📉 Mean Absolute Error Comparison")

    mae_chart = results.set_index(
        "Model"
    )[["MAE"]]

    st.bar_chart(
        mae_chart,
        color="#059669"
    )

    st.divider()

    st.subheader("📈 Root Mean Squared Error Comparison")

    rmse_chart = results.set_index(
        "Model"
    )[["RMSE"]]

    st.bar_chart(
        rmse_chart,
        color="#2563eb"
    )

    best_model = results.loc[
        results["MAE"].idxmin()
    ]

    st.success(
        f"Lowest MAE: {best_model['Model']} "
        f"({best_model['MAE']:.2f})"
    )

    st.info(
        "The evaluation uses six chronological test observations. "
        "These results are preliminary and require further validation."
    )

    st.download_button(
        "⬇️ Download Final Evaluation",
        results.to_csv(index=False),
        file_name="final_model_evaluation.csv",
        mime="text/csv"
    )

    if os.path.exists(COMPARISON_REPORT):

        st.divider()

        st.subheader("Original Machine Learning Model Comparison")

        original_results = pd.read_csv(
            COMPARISON_REPORT
        )

        st.dataframe(
            original_results.round(2),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Original Comparison",
            original_results.to_csv(index=False),
            file_name="model_comparison.csv",
            mime="text/csv"
        )


# =====================================================
# PAGE 6: ONION HYBRID PREDICTION EXPLORER
# =====================================================

elif page == "Onion Hybrid Prediction":

    st.header("?? Onion Hybrid Prediction Explorer")

    st.markdown(
        """
        Explore historical one-step-ahead onion price predictions.
        The selected model for each market was chosen using its
        September 2025 validation performance.
        """
    )

    st.warning(
        "Historical experiment only: the test period is "
        "1 October?5 November 2025, and the latest source "
        "observation is 20 April 2026. These results are not "
        "current November 2026 market prices or live forecasts."
    )

    try:
        predictions, selection = load_hybrid_results()
        market_options = get_market_options(predictions)
        latest_source_date = get_data_freshness()

    except Exception as error:
        st.error(f"Could not load hybrid prediction data: {error}")
        st.stop()

    st.subheader("Select an onion market")

    selected_state = st.selectbox(
        "State",
        sorted(market_options["State"].dropna().unique()),
        key="hybrid_state",
    )

    state_options = market_options[
        market_options["State"] == selected_state
    ]

    selected_district = st.selectbox(
        "District",
        sorted(state_options["District"].dropna().unique()),
        key="hybrid_district",
    )

    district_options = state_options[
        state_options["District"] == selected_district
    ]

    selected_market = st.selectbox(
        "Market",
        sorted(district_options["Market"].dropna().unique()),
        key="hybrid_market",
    )

    market_rows = district_options[
        district_options["Market"] == selected_market
    ]

    selected_variety = st.selectbox(
        "Variety",
        sorted(market_rows["Variety"].dropna().unique()),
        key="hybrid_variety",
    )

    variety_rows = market_rows[
        market_rows["Variety"] == selected_variety
    ]

    selected_grade = st.selectbox(
        "Grade",
        sorted(variety_rows["Grade"].dropna().unique()),
        key="hybrid_grade",
    )

    results = get_market_predictions(
        predictions,
        selected_state,
        selected_district,
        selected_market,
        selected_variety,
        selected_grade,
    )

    selected_model = get_selected_model(
        selection,
        selected_state,
        selected_district,
        selected_market,
        selected_variety,
        selected_grade,
    )

    st.divider()

    st.subheader("Market prediction summary")

    st.write(
        f"**Selected market:** {selected_market}, "
        f"{selected_district}, {selected_state}"
    )
    st.write(f"**Variety:** {selected_variety}")
    st.write(f"**Grade:** {selected_grade}")
    st.write(f"**Validation-selected strategy:** {selected_model}")

    if results.empty:
        st.info("No historical test predictions are available for this market.")
        st.stop()

    actual = results["Actual_Price"]
    predicted = results["Predicted_Price"]
    absolute_error = results["Absolute_Error"]

    market_mae = absolute_error.mean()
    market_rmse = np.sqrt(((actual - predicted) ** 2).mean())

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Test observations",
        f"{len(results):,}",
    )

    metric2.metric(
        "Market MAE",
        f"?{market_mae:,.2f}",
    )

    metric3.metric(
        "Market RMSE",
        f"?{market_rmse:,.2f}",
    )

    metric4.metric(
        "Latest test actual price",
        f"?{actual.iloc[-1]:,.2f}",
    )

    st.caption(
        f"Test dates: {results['Arrival_Date'].min().date()} "
        f"to {results['Arrival_Date'].max().date()} | "
        f"Source data latest date: {latest_source_date.date()}"
    )

    st.divider()

    st.subheader("Actual vs predicted price")

    price_chart = (
        results.set_index("Arrival_Date")[
            ["Actual_Price", "Predicted_Price"]
        ]
        .rename(
            columns={
                "Actual_Price": "Actual Price",
                "Predicted_Price": "Predicted Price",
            }
        )
    )

    st.line_chart(
        price_chart,
        use_container_width=True,
    )

    st.divider()

    st.subheader("Prediction error over time")

    error_chart = (
        results.set_index("Arrival_Date")[["Absolute_Error"]]
        .rename(columns={"Absolute_Error": "Absolute Error"})
    )

    st.line_chart(
        error_chart,
        use_container_width=True,
    )

    st.divider()

    st.subheader("Historical prediction records")

    display_results = results[
        [
            "Arrival_Date",
            "Actual_Price",
            "Predicted_Price",
            "Selected_Model",
            "Absolute_Error",
        ]
    ].copy()

    display_results["Arrival_Date"] = (
        display_results["Arrival_Date"].dt.strftime("%Y-%m-%d")
    )

    display_results = display_results.rename(
        columns={
            "Arrival_Date": "Date",
            "Actual_Price": "Actual Price (?)",
            "Predicted_Price": "Predicted Price (?)",
            "Selected_Model": "Selected Model",
            "Absolute_Error": "Absolute Error (?)",
        }
    )

    st.dataframe(
        display_results.round(2),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download selected market results",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="onion_hybrid_market_predictions.csv",
        mime="text/csv",
    )

    st.info(
        "Interpretation: each prediction estimates the next available "
        "observation for the selected market series. The historical "
        "test observations are not necessarily consecutive calendar days."
    )


# =====================================================
# 8. PREMIUM FOOTER
# =====================================================

st.markdown(
    """
    <div class="footer">
        <h3 style="color:white !important;">
            🌾 AgriSmart
        </h3>
        <p>
            Agricultural Price Analysis and Prediction System
        </p>
        <p>
            Machine Learning | Data Analytics | Smart Agriculture
        </p>
        <p>
            B.Tech Artificial Intelligence and Data Science
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
