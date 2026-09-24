# Sustainability AI Engine

A production-ready, modular Streamlit web application that integrates 8 distinct sustainability machine learning models consolidated into 4 functional Hubs.

## Hubs Included

1. **⚡ Hub A: Energy & Clean Tech Engine**
   - Appliance Energy Predictor
   - Solar Power Output Predictor
   - Green Tech Sustainability Classifier
2. **☁️ Hub B: Carbon & Environmental Impact Engine**
   - CO2 Emission Predictor
   - Emission Reduction Predictor
3. **♻️ Hub C: Waste & Circular Economy Engine**
   - Waste Type Classification for Recycling
   - Waste Management Strategy Predictor
4. **🌾 Hub D: Land & Agritech Engine**
   - Agricultural Sustainability Predictor

## Installation

1. Clone this repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

To launch the unified Streamlit application, run:

```bash
streamlit run main.py
```

The app will start a local web server (usually at `http://localhost:8501`) and open automatically in your browser.

## Project Structure

- `main.py`: The central Streamlit application entry point.
- `style.css`: Custom premium CSS styling for the interface.
- `hubs/`: Modular Python package containing the logic for each of the 4 hubs.
- `*/`: Individual model directories containing datasets (`.csv`), standalone legacy scripts, and pre-trained models (`.pkl`).