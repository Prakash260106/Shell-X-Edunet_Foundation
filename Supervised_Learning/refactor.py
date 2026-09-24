import os
import re

hubs_structure = {
    "hub_a_energy": {
        "run_appliance_energy": r"Appliance-Energy-predictor\app.py",
        "run_solar_power": r"Solar-Power-Output-Predictor\app_solar.py",
        "run_green_tech": r"Green-Tech-Sustainability-Classifier lr\app.py"
    },
    "hub_b_carbon": {
        "run_co2_emission": r"CO2-Emission-Predictor\app_polynomial.py",
        "run_emission_reduction": r"Emission-Reduction-Predictor-knn\app.py"
    },
    "hub_c_waste": {
        "run_waste_type": r"Waste-Type-Classification-for-Recycling-DecisionTreeClassifier\app.py",
        "run_waste_management": r"Waste-Management-Strategy-Predictor SVC\app_svc.py"
    },
    "hub_d_agritech": {
        "run_agritech": r"Agricultural-Sustainability-Predictor-RF\app.py"
    }
}

for hub_name, models in hubs_structure.items():
    hub_file = f"hubs/{hub_name}.py"
    with open(hub_file, "w", encoding="utf-8") as out_f:
        out_f.write("import streamlit as st\n")
        out_f.write("import pandas as pd\n")
        out_f.write("import numpy as np\n")
        out_f.write("import matplotlib.pyplot as plt\n")
        out_f.write("import os\n")
        out_f.write("import joblib\n")
        out_f.write("import io\n")
        out_f.write("from sklearn.model_selection import train_test_split\n")
        out_f.write("from sklearn.linear_model import LinearRegression, LogisticRegression\n")
        out_f.write("from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report\n")
        out_f.write("\n")
        
        for func_name, rel_path in models.items():
            if not os.path.exists(rel_path):
                print(f"Path not found: {rel_path}")
                continue
                
            out_f.write(f"def {func_name}():\n")
            with open(rel_path, "r", encoding="utf-8") as in_f:
                content = in_f.read()
                
            # Remove st.set_page_config
            content = re.sub(r"st\.set_page_config\([^)]*\)", "", content)
            
            # Fix relative paths
            dir_name = os.path.dirname(rel_path).replace("\\", "/")
            content = content.replace('pd.read_csv("', f'pd.read_csv("{dir_name}/')
            content = content.replace("pd.read_csv('", f"pd.read_csv('{dir_name}/")
            
            content = content.replace('DEFAULT_PATH = "', f'DEFAULT_PATH = "{dir_name}/')
            content = content.replace("DEFAULT_PATH = '", f"DEFAULT_PATH = '{dir_name}/")
            
            # Indent content
            indented_content = "\n".join("    " + line for line in content.split("\n"))
            out_f.write(indented_content)
            out_f.write("\n\n")

with open("hubs/__init__.py", "w") as f:
    f.write("")

print("Refactoring complete.")
