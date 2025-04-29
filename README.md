# ERCOT Offer Curve Forecasting 🚀

This project forecasts ERCOT generator offer curves using a Transformer deep learning model.

## 📦 Project Structure

- `model_train_script.py` — Code to process ERCOT SCED data, train a Transformer model, and save predictions
- `offer_curve_app.py` — Streamlit app to visualize and export the forecasts
- `Model_Info/` — Folder containing model outputs:
  - `X_sample.npy`
  - `Y_sample.npy`
  - `predictions.npy`
  - `timestamps.npy`
  - `df_sample.csv`
  - `model.keras`

## 🚀 Quick Start

1. Clone the repository:
    ```bash
    git clone https://github.com/jakeosness/ercot-offer-curve-forecast.git
    cd ercot-offer-curve-forecast
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit App:
    ```bash
    streamlit run offer_curve_app.py
    ```

## 📊 About the Model

- Trained on 3 months of ERCOT SCED data (October–December 2024)
- Inputs: generator MW offers, prices, load data, and gas prices
- Forecasts: 72-hour curves, predicting MW and Price points for each generator

## 🔗 External Files

Large model artifacts (`Model_Info/`) are included here, but may need to be updated if retraining.


---
