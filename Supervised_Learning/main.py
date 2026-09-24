import streamlit as st
import importlib
from pathlib import Path

st.set_page_config(page_title="Sustainability AI Engine", page_icon="🌍", layout="wide")

# Load CSS for premium aesthetic
css_path = Path(__file__).resolve().with_name("style.css")
with open(css_path, "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize Session State for navigation
if "current_hub" not in st.session_state:
    st.session_state.current_hub = None
if "current_model" not in st.session_state:
    st.session_state.current_model = None

def navigate_to_hub(hub_name):
    st.session_state.current_hub = hub_name
    st.session_state.current_model = None
    st.rerun()

def navigate_to_model(model_name):
    st.session_state.current_model = model_name
    st.rerun()

def go_home():
    st.session_state.current_hub = None
    st.session_state.current_model = None
    st.rerun()

# ----------------- Dashboard View -----------------
if st.session_state.current_hub is None:
    st.title("🌍 Sustainability AI Dashboard")
    st.markdown("Select a hub below to explore its machine learning models.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("assets/hub_a.png", width=220)
        st.markdown("### ⚡ Hub A: Energy & Clean Tech")
        if st.button("Explore Hub A", key="btn_hub_a", use_container_width=True):
            navigate_to_hub("Hub A")
            
        st.markdown("<br>", unsafe_allow_html=True)
            
        st.image("assets/hub_c.png", width=220)
        st.markdown("### ♻️ Hub C: Waste & Circular Economy")
        if st.button("Explore Hub C", key="btn_hub_c", use_container_width=True):
            navigate_to_hub("Hub C")

    with col2:
        st.image("assets/hub_b.png", width=220)
        st.markdown("### ☁️ Hub B: Carbon & Env Impact")
        if st.button("Explore Hub B", key="btn_hub_b", use_container_width=True):
            navigate_to_hub("Hub B")
            
        st.markdown("<br>", unsafe_allow_html=True)
            
        st.image("assets/hub_d.png", width=220)
        st.markdown("### 🌾 Hub D: Land & Agritech")
        if st.button("Explore Hub D", key="btn_hub_d", use_container_width=True):
            navigate_to_hub("Hub D")

# ----------------- Hub View -----------------
elif st.session_state.current_model is None:
    if st.button("← Back to Dashboard"):
        go_home()
        
    hub = st.session_state.current_hub
    st.title(f"{hub} Models")
    st.markdown("---")
    
    if hub == "Hub A":
        models_info = [
            ("Appliance Energy Predictor", "Predicts energy consumption based on temperature.", "Simple Linear Regression"),
            ("Solar Power Output Predictor", "Predicts solar output from weather data.", "Linear Regression"),
            ("Green Tech Sustainability Classifier", "Classifies green tech sustainability levels.", "Logistic Regression")
        ]
    elif hub == "Hub B":
        models_info = [
            ("CO2 Emission Predictor", "Predicts CO2 emissions using energy consumption and GDP.", "Polynomial Regression"),
            ("Emission Reduction Predictor", "Predicts effectiveness of reduction strategies.", "K-Nearest Neighbors (KNN)")
        ]
    elif hub == "Hub C":
        models_info = [
            ("Waste Type Classification for Recycling", "Classifies waste into categories like Plastic, Glass, etc.", "Decision Tree Classifier"),
            ("Waste Management Strategy Predictor", "Recommends best waste management strategies.", "Support Vector Classification (SVC)")
        ]
    elif hub == "Hub D":
        models_info = [
            ("Agricultural Sustainability Predictor", "Predicts agricultural sustainability scores.", "Random Forest")
        ]
        
    for name, desc, algo in models_info:
        icon_img = "assets/hub_a.png"
        if hub == "Hub B": icon_img = "assets/hub_b.png"
        elif hub == "Hub C": icon_img = "assets/hub_c.png"
        elif hub == "Hub D": icon_img = "assets/hub_d.png"
        
        # We use a markdown trick to style the container roughly
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(icon_img, width=160)
        with col2:
            st.markdown(f"### {name}")
            st.write(f"**Description:** {desc}")
            st.write(f"**Algorithm:** `{algo}`")
            if st.button(f"Launch {name}", key=f"launch_{name}", use_container_width=True):
                navigate_to_model(name)
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------- Model View -----------------
else:
    if st.button("← Back to Hub Models"):
        st.session_state.current_model = None
        st.rerun()
        
    model = st.session_state.current_model
    
    # Import and run the respective model based on the selection
    if model == "Appliance Energy Predictor":
        from hubs.hub_a_energy import run_appliance_energy
        run_appliance_energy()
    elif model == "Solar Power Output Predictor":
        from hubs.hub_a_energy import run_solar_power
        run_solar_power()
    elif model == "Green Tech Sustainability Classifier":
        from hubs.hub_a_energy import run_green_tech
        run_green_tech()
    elif model == "CO2 Emission Predictor":
        from hubs.hub_b_carbon import run_co2_emission
        run_co2_emission()
    elif model == "Emission Reduction Predictor":
        from hubs.hub_b_carbon import run_emission_reduction
        run_emission_reduction()
    elif model == "Waste Type Classification for Recycling":
        from hubs.hub_c_waste import run_waste_type
        run_waste_type()
    elif model == "Waste Management Strategy Predictor":
        from hubs.hub_c_waste import run_waste_management
        run_waste_management()
    elif model == "Agricultural Sustainability Predictor":
        from hubs.hub_d_agritech import run_agritech
        run_agritech()
