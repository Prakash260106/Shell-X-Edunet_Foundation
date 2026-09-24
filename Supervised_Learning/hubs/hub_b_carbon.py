import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

def run_co2_emission():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import joblib
    import io
    
    import plotly.express as px
    import plotly.graph_objects as go
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    

    
    FEATURES = ["Energy_Consumption", "Renewable_Percentage", "GDP"]
    TARGET = "CO2_Emissions"
    
    st.title("🌍 CO2 Emissions Predictor — Polynomial Regression Dashboard")
    st.info("🧠 **Algorithm Technique Used:** Polynomial Regression")
    st.write(
        "Predicts **CO2 Emissions** from **Energy Consumption**, "
        "**Renewable Energy %**, and **GDP** using polynomial regression."
    )
    
    # ===========================================================
    # Sidebar — data & model controls
    # ===========================================================
    st.sidebar.header("⚙️ Settings")
    
    uploaded_file = st.sidebar.file_uploader("Upload sustainability_data.csv", type=["csv"])
    DEFAULT_PATH = "CO2-Emission-Predictor/sustainability_data.csv"
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        try:
            data = pd.read_csv(DEFAULT_PATH)
            st.sidebar.info(f"Using bundled `{DEFAULT_PATH}` found alongside the app.")
        except FileNotFoundError:
            st.warning(
                "No dataset loaded yet. Please upload `sustainability_data.csv` "
                "using the sidebar to continue."
            )
            st.stop()
    
    missing_cols = [c for c in FEATURES + [TARGET] if c not in data.columns]
    if missing_cols:
        st.error(f"Dataset is missing required column(s): {missing_cols}")
        st.stop()
    
    missing_before = data.isnull().sum().sum()
    data = data.dropna()
    
    st.sidebar.markdown("---")
    degree = st.sidebar.slider("Polynomial degree", 1, 5, 2)
    test_size = st.sidebar.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random state", value=42, step=1)
    
    # ===========================================================
    # Train model (cached on inputs that affect it)
    # ===========================================================
    @st.cache_resource(show_spinner="Training model...")
    def train_model(df, degree, test_size, random_state):
        X = df[FEATURES]
        y = df[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state)
        )
        poly = PolynomialFeatures(degree=degree)
        X_poly_train = poly.fit_transform(X_train)
        X_poly_test = poly.transform(X_test)
    
        model = LinearRegression()
        model.fit(X_poly_train, y_train)
        y_pred = model.predict(X_poly_test)
    
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    
        return model, poly, X_train, X_test, y_train, y_test, y_pred, mse, r2
    
    
    model, poly, X_train, X_test, y_train, y_test, y_pred, mse, r2 = train_model(
        data, degree, test_size, random_state
    )
    
    # ===========================================================
    # Tabs
    # ===========================================================
    tab_data, tab_eda, tab_perf, tab_predict, tab_download = st.tabs(
        ["📄 Data", "🔎 Explore", "📊 Model Performance", "🎛️ Predict", "💾 Download"]
    )
    
    # -----------------------------------------------------------
    # Tab: Data
    # -----------------------------------------------------------
    with tab_data:
        st.subheader("Dataset preview")
        st.dataframe(data.head(20), use_container_width=True)
        if missing_before > 0:
            st.caption(f"Dropped rows with missing values ({missing_before} missing cells found).")
    
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", len(data))
        c2.metric("Features", len(FEATURES))
        c3.metric("Train rows", len(X_train))
        c4.metric("Test rows", len(X_test))
    
        st.subheader("Summary statistics")
        st.dataframe(data[FEATURES + [TARGET]].describe(), use_container_width=True)
    
    # -----------------------------------------------------------
    # Tab: Explore (EDA)
    # -----------------------------------------------------------
    with tab_eda:
        st.subheader("Correlation heatmap")
        corr = data[FEATURES + [TARGET]].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title="Feature correlation matrix",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
        st.subheader("Feature vs. Target")
        feat_choice = st.selectbox("Choose a feature to plot against CO2 Emissions", FEATURES)
        fig_scatter = px.scatter(
            data,
            x=feat_choice,
            y=TARGET,
            trendline="ols",
            opacity=0.7,
            title=f"{feat_choice} vs {TARGET}",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
        st.subheader("Distributions")
        dist_col = st.selectbox("Choose a column to view its distribution", FEATURES + [TARGET], index=len(FEATURES))
        fig_hist = px.histogram(data, x=dist_col, nbins=30, marginal="box", title=f"Distribution of {dist_col}")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # -----------------------------------------------------------
    # Tab: Model Performance
    # -----------------------------------------------------------
    with tab_perf:
        st.subheader("Performance metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Squared Error", f"{mse:,.2f}")
        c2.metric("R² Score", f"{r2:.4f}")
        c3.metric("Polynomial Degree", degree)
    
        st.subheader("Actual vs. Predicted")
        perf_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
        fig_pred = px.scatter(
            perf_df,
            x="Actual",
            y="Predicted",
            opacity=0.7,
            title="Polynomial Regression: Predictions vs. Actual Values",
        )
        min_v, max_v = perf_df["Actual"].min(), perf_df["Actual"].max()
        fig_pred.add_trace(
            go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="Perfect prediction", line=dict(color="red"))
        )
        st.plotly_chart(fig_pred, use_container_width=True)
    
        st.subheader("Residuals")
        residuals = perf_df["Actual"] - perf_df["Predicted"]
        fig_res = px.scatter(
            x=perf_df["Predicted"],
            y=residuals,
            labels={"x": "Predicted", "y": "Residual"},
            opacity=0.7,
            title="Residual plot",
        )
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_res, use_container_width=True)
    
        st.subheader("Residual distribution")
        fig_res_hist = px.histogram(residuals, nbins=30, title="Residuals distribution")
        fig_res_hist.update_layout(showlegend=False, xaxis_title="Residual")
        st.plotly_chart(fig_res_hist, use_container_width=True)
    
    # -----------------------------------------------------------
    # Tab: Predict
    # -----------------------------------------------------------
    with tab_predict:
        st.subheader("Try a prediction")
        st.write("Adjust the inputs below to predict CO2 Emissions.")
    
        col1, col2, col3 = st.columns(3)
        with col1:
            energy_in = st.slider(
                "Energy Consumption",
                float(data["Energy_Consumption"].min()),
                float(data["Energy_Consumption"].max()),
                float(data["Energy_Consumption"].mean()),
            )
        with col2:
            renew_in = st.slider(
                "Renewable Percentage",
                float(data["Renewable_Percentage"].min()),
                float(data["Renewable_Percentage"].max()),
                float(data["Renewable_Percentage"].mean()),
            )
        with col3:
            gdp_in = st.slider(
                "GDP",
                float(data["GDP"].min()),
                float(data["GDP"].max()),
                float(data["GDP"].mean()),
            )
    
        input_df = pd.DataFrame(
            {"Energy_Consumption": [energy_in], "Renewable_Percentage": [renew_in], "GDP": [gdp_in]}
        )
        input_poly = poly.transform(input_df)
        prediction = model.predict(input_poly)[0]
    
        st.success(f"Predicted CO2 Emissions: **{prediction:,.2f}**")
    
        st.markdown("---")
        st.subheader("Sensitivity: how prediction changes with one feature")
        sweep_feature = st.selectbox("Feature to sweep", FEATURES, key="sweep")
        sweep_range = np.linspace(data[sweep_feature].min(), data[sweep_feature].max(), 50)
    
        # Build sweep dataframe using current slider values as the baseline for other features
        baseline = {"Energy_Consumption": energy_in, "Renewable_Percentage": renew_in, "GDP": gdp_in}
        sweep_df = pd.DataFrame([baseline] * 50)
        sweep_df[sweep_feature] = sweep_range
        sweep_pred = model.predict(poly.transform(sweep_df))
    
        fig_sweep = px.line(
            x=sweep_range,
            y=sweep_pred,
            labels={"x": sweep_feature, "y": "Predicted CO2 Emissions"},
            title=f"Predicted CO2 Emissions as {sweep_feature} varies (other features held constant)",
        )
        fig_sweep.add_vline(x=baseline[sweep_feature], line_dash="dash", line_color="green")
        st.plotly_chart(fig_sweep, use_container_width=True)
    
    # -----------------------------------------------------------
    # Tab: Download
    # -----------------------------------------------------------
    with tab_download:
        st.subheader("Download trained model")
        model_buffer = io.BytesIO()
        joblib.dump({"model": model, "poly": poly, "features": FEATURES}, model_buffer)
        model_buffer.seek(0)
        st.download_button(
            label="Download model (.pkl)",
            data=model_buffer,
            file_name="polynomialRegModel.pkl",
            mime="application/octet-stream",
        )
    
        st.subheader("Download predictions on test set")
        result_df = X_test.copy()
        result_df["Actual"] = y_test.values
        result_df["Predicted"] = y_pred
        csv_buffer = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download test predictions (.csv)",
            data=csv_buffer,
            file_name="test_predictions.csv",
            mime="text/csv",
        )
    

def run_emission_reduction():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    
    # ────────────────────────────────────────────────────────────────────────────
    # PAGE CONFIG
    # ────────────────────────────────────────────────────────────────────────────
    
    
    # ────────────────────────────────────────────────────────────────────────────
    # CUSTOM CSS — colorful, modern, "climate tech" theme
    # ────────────────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        }
        h1, h2, h3, h4 {
            color: #eafff5 !important;
            font-family: 'Trebuchet MS', sans-serif;
        }
        p, label, span, div {
            color: #e8f7f0;
        }
        .hero-banner {
            background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
            padding: 28px 32px;
            border-radius: 18px;
            margin-bottom: 22px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.35);
        }
        .hero-banner h1 {
            color: #05261f !important;
            margin: 0;
            font-size: 2.3rem;
        }
        .hero-banner p {
            color: #06382d !important;
            font-size: 1.05rem;
            margin-top: 6px;
        }
        .metric-card {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 16px;
            padding: 18px;
            text-align: center;
            backdrop-filter: blur(6px);
        }
        .result-effective {
            background: linear-gradient(120deg, #11998e, #38ef7d);
            color: #06281f;
            padding: 26px;
            border-radius: 18px;
            text-align: center;
            font-size: 1.5rem;
            font-weight: 700;
            box-shadow: 0 10px 30px rgba(56,239,125,0.35);
            animation: pulseGreen 1.8s ease-in-out infinite;
        }
        .result-not-effective {
            background: linear-gradient(120deg, #ff5f6d, #ffc371);
            color: #401515;
            padding: 26px;
            border-radius: 18px;
            text-align: center;
            font-size: 1.5rem;
            font-weight: 700;
            box-shadow: 0 10px 30px rgba(255,95,109,0.35);
        }
        @keyframes pulseGreen {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1e23 0%, #16323a 100%);
        }
        .stButton>button {
            background: linear-gradient(90deg, #11998e, #38ef7d);
            color: #06281f;
            font-weight: 700;
            border-radius: 12px;
            border: none;
            padding: 12px 26px;
            font-size: 1.05rem;
            transition: 0.2s;
            width: 100%;
        }
        .stButton>button:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 18px rgba(56,239,125,0.45);
        }
        div[data-baseweb="tab-list"] {
            gap: 8px;
        }
        button[data-baseweb="tab"] {
            background-color: rgba(255,255,255,0.06);
            border-radius: 10px 10px 0 0;
            color: #e8f7f0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # ────────────────────────────────────────────────────────────────────────────
    # HERO BANNER
    # ────────────────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-banner">
            <h1>🌍 Emission Reduction Predictor</h1>
            <p>Powered by a K-Nearest Neighbors model &nbsp;•&nbsp;
            Estimate whether a proposed energy strategy will effectively reduce emissions</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("🧠 **Algorithm Technique Used:** K-Nearest Neighbors (KNN)")
    
    # ────────────────────────────────────────────────────────────────────────────
    # DATA + MODEL (cached)
    # ────────────────────────────────────────────────────────────────────────────
    base_dir = Path(__file__).resolve().parents[1]
    data_candidates = [
        base_dir / "Emission-Reduction-Predictor-knn" / "emissions_reduction_data (2).csv",
        base_dir / "Emission-Reduction-Predictor-knn" / "emissions_reduction_data.csv",
        base_dir / "emissions_reduction_data.csv",
    ]
    DATA_PATH = next((str(p) for p in data_candidates if p.exists()), str(data_candidates[0]))

    if not Path(DATA_PATH).exists():
        st.error(
            "Dataset not found. Expected one of: "
            + ", ".join(str(p) for p in data_candidates)
        )
        st.stop()

    @st.cache_data
    def load_data(path):
        df = pd.read_csv(path)
        df.fillna(df.mean(numeric_only=True), inplace=True)
        return df
    
    
    @st.cache_resource
    def train_model(df, k):
        X = df[["energy_efficiency", "renewable_ratio", "technology_cost"]]
        y = df["emission_reduction"]
    
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled, y_train)
    
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, target_names=["Not Effective", "Effective"], output_dict=True
        )
    
        return model, scaler, acc, cm, report, X_test, y_test, y_pred
    
    
    data = load_data(DATA_PATH)
    
    # ────────────────────────────────────────────────────────────────────────────
    # SIDEBAR — inputs
    # ────────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Model Settings")
        k = st.slider("Number of Neighbors (k)", min_value=1, max_value=25, value=5, step=1)
    
        st.markdown("---")
        st.markdown("## 🧮 Input Your Scenario")
    
        energy_efficiency = st.slider(
            "⚡ Energy Efficiency (%)",
            float(data.energy_efficiency.min()),
            float(data.energy_efficiency.max()),
            float(data.energy_efficiency.mean()),
            step=0.1,
        )
        renewable_ratio = st.slider(
            "♻️ Renewable Energy Ratio",
            0.0,
            1.0,
            float(data.renewable_ratio.mean()),
            step=0.01,
        )
        technology_cost = st.slider(
            "💰 Technology Cost ($)",
            float(data.technology_cost.min()),
            float(data.technology_cost.max()),
            float(data.technology_cost.mean()),
            step=10.0,
        )
    
        st.markdown("---")
        predict_clicked = st.button("🔮 Predict Emission Reduction")
    
    # Train (cached on k)
    model, scaler, acc, cm, report, X_test, y_test, y_pred = train_model(data, k)
    
    # ────────────────────────────────────────────────────────────────────────────
    # TOP METRIC ROW
    # ────────────────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""<div class="metric-card"><h3>🎯 Accuracy</h3>
            <h2 style="color:#38ef7d !important;">{acc*100:.2f}%</h2></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="metric-card"><h3>📊 Rows</h3>
            <h2 style="color:#38ef7d !important;">{len(data):,}</h2></div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""<div class="metric-card"><h3>✅ Effective Rate</h3>
            <h2 style="color:#38ef7d !important;">{data.emission_reduction.mean()*100:.1f}%</h2></div>""",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""<div class="metric-card"><h3>🧭 k Value</h3>
            <h2 style="color:#38ef7d !important;">{k}</h2></div>""",
            unsafe_allow_html=True,
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ────────────────────────────────────────────────────────────────────────────
    # TABS
    # ────────────────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📈 Model Performance", "🔍 Explore Data"])
    
    # ---- TAB 1: PREDICTION ----
    with tab1:
        left, right = st.columns([1, 1])
    
        with left:
            st.markdown("### Your Scenario")
            input_df = pd.DataFrame(
                {
                    "energy_efficiency": [energy_efficiency],
                    "renewable_ratio": [renewable_ratio],
                    "technology_cost": [technology_cost],
                }
            )
            st.dataframe(input_df.style.format(precision=2), use_container_width=True)
    
            radar_fig = go.Figure()
            radar_fig.add_trace(
                go.Scatterpolar(
                    r=[
                        energy_efficiency / data.energy_efficiency.max() * 100,
                        renewable_ratio * 100,
                        100 - (technology_cost / data.technology_cost.max() * 100),
                    ],
                    theta=["Energy Efficiency", "Renewable Ratio", "Cost Advantage"],
                    fill="toself",
                    line_color="#38ef7d",
                )
            )
            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], color="white"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                margin=dict(t=20, b=20),
                height=350,
            )
            st.plotly_chart(radar_fig, use_container_width=True)
    
        with right:
            st.markdown("### Prediction Result")
            if predict_clicked:
                scaled_input = scaler.transform(input_df)
                pred = model.predict(scaled_input)[0]
                proba = model.predict_proba(scaled_input)[0]
    
                if pred == 1:
                    st.markdown(
                        f"""<div class="result-effective">✅ EFFECTIVE REDUCTION<br>
                        <span style="font-size:1rem;">Confidence: {proba[1]*100:.1f}%</span></div>""",
                        unsafe_allow_html=True,
                    )
                    st.balloons()
                else:
                    st.markdown(
                        f"""<div class="result-not-effective">⚠️ NOT EFFECTIVE<br>
                        <span style="font-size:1rem;">Confidence: {proba[0]*100:.1f}%</span></div>""",
                        unsafe_allow_html=True,
                    )
    
                st.markdown("<br>", unsafe_allow_html=True)
                prob_fig = px.bar(
                    x=["Not Effective", "Effective"],
                    y=proba,
                    color=["Not Effective", "Effective"],
                    color_discrete_map={"Not Effective": "#ff5f6d", "Effective": "#38ef7d"},
                    labels={"x": "Class", "y": "Probability"},
                    text=[f"{p*100:.1f}%" for p in proba],
                )
                prob_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    showlegend=False,
                    height=300,
                )
                st.plotly_chart(prob_fig, use_container_width=True)
            else:
                st.info("👈 Adjust the sliders in the sidebar and click **Predict Emission Reduction** to see results here.")
    
    # ---- TAB 2: MODEL PERFORMANCE ----
    with tab2:
        col1, col2 = st.columns(2)
    
        with col1:
            st.markdown("### Confusion Matrix")
            cm_fig = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale="Tealgrn",
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Not Effective", "Effective"],
                y=["Not Effective", "Effective"],
            )
            cm_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                height=380,
            )
            st.plotly_chart(cm_fig, use_container_width=True)
    
        with col2:
            st.markdown("### Classification Report")
            report_df = pd.DataFrame(report).transpose().round(3)
            st.dataframe(report_df.style.background_gradient(cmap="Greens"), use_container_width=True)
    
        st.markdown("### Accuracy vs. Number of Neighbors (k)")
        k_values = list(range(1, 26))
        acc_scores = []
        X = data[["energy_efficiency", "renewable_ratio", "technology_cost"]]
        y = data["emission_reduction"]
        X_train, X_test_k, y_train, y_test_k = train_test_split(X, y, test_size=0.2, random_state=42)
        sc = StandardScaler()
        X_train_s = sc.fit_transform(X_train)
        X_test_s = sc.transform(X_test_k)
        for kv in k_values:
            m = KNeighborsClassifier(n_neighbors=kv)
            m.fit(X_train_s, y_train)
            acc_scores.append(accuracy_score(y_test_k, m.predict(X_test_s)))
    
        k_fig = px.line(
            x=k_values, y=acc_scores, markers=True,
            labels={"x": "k (Number of Neighbors)", "y": "Accuracy"},
        )
        k_fig.add_vline(x=k, line_dash="dash", line_color="#38ef7d",
                         annotation_text=f"Current k={k}", annotation_font_color="white")
        k_fig.update_traces(line_color="#38ef7d", marker=dict(size=8, color="#11998e"))
        k_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=380,
        )
        st.plotly_chart(k_fig, use_container_width=True)
    
    # ---- TAB 3: EXPLORE DATA ----
    with tab3:
        st.markdown("### Dataset Snapshot")
        st.dataframe(data.head(20), use_container_width=True)
    
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Feature Distributions")
            feature_choice = st.selectbox(
                "Choose a feature", ["energy_efficiency", "renewable_ratio", "technology_cost"]
            )
            hist_fig = px.histogram(
                data, x=feature_choice, color="emission_reduction",
                color_discrete_map={0: "#ff5f6d", 1: "#38ef7d"},
                barmode="overlay", nbins=40,
                labels={"emission_reduction": "Effective"},
            )
            hist_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white", height=380,
            )
            st.plotly_chart(hist_fig, use_container_width=True)
    
        with col2:
            st.markdown("### Feature Relationships")
            scatter_fig = px.scatter(
                data, x="energy_efficiency", y="renewable_ratio",
                color="emission_reduction", size="technology_cost",
                color_discrete_map={0: "#ff5f6d", 1: "#38ef7d"},
                opacity=0.6,
                labels={"emission_reduction": "Effective"},
            )
            scatter_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white", height=380,
            )
            st.plotly_chart(scatter_fig, use_container_width=True)
    
        st.markdown("### Class Balance")
        pie_fig = px.pie(
            data, names=data["emission_reduction"].map({0: "Not Effective", 1: "Effective"}),
            color=data["emission_reduction"].map({0: "Not Effective", 1: "Effective"}),
            color_discrete_map={"Not Effective": "#ff5f6d", "Effective": "#38ef7d"},
            hole=0.45,
        )
        pie_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=380,
        )
        st.plotly_chart(pie_fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; opacity:0.7;'>Built with Streamlit • KNN Classifier • "
        "Data-driven climate insights 🌱</p>",
        unsafe_allow_html=True,
    )
    

