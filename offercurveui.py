# offer_curve_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import base64
from io import BytesIO
import streamlit.components.v1 as components
import matplotlib.gridspec as gridspec

# === Load Data (LOCAL paths now)
X = np.load("./Model_Info/X_sample.npy")
Y = np.load("./Model_Info/Y_sample.npy")
all_preds = np.load("./Model_Info/predictions.npy")
timestamps = np.load("./Model_Info/timestamps.npy", allow_pickle=True)
df = pd.read_csv("./Model_Info/df_sample.csv")

# === Extract generator names for dropdown
resource_names = [df.loc[i + 2016, 'Resource Name'] for i in range(len(X))]
unique_generators = sorted(set(resource_names))

# === Streamlit UI ===
st.title("ERCOT Generator Offer Curve Forecast")
generator = st.selectbox("Choose Generator:", unique_generators)

# === View mode toggle
view_mode = st.radio(
    "\ud83d\udcca Select Display Mode:",
    ["Scrollable (1 row, wide plots)", "Grid (4x6 layout, all visible)"],
    index=0
)

# === Excel Export Functions (unchanged)
# (your convert_single_prediction_to_excel and convert_all_predictions_to_excel functions remain unchanged)

# === Plotting Function (unchanged)
# (your plot_offer_curves_with_metrics function remains unchanged)

# === Match Generator to First Sample
sample_indices = [i for i in range(len(X)) if df.loc[i + 2016, 'Resource Name'] == generator]

if not sample_indices:
    st.warning(f"No forecast samples found for {generator}.")
else:
    idx = sample_indices[0]
    y_pred = all_preds[idx]
    y_true = Y[idx]
    start_time = pd.to_datetime(timestamps[idx])

    # === NEW: Compute global metrics ===
    points_per_curve = 16
    num_hours = len(y_pred) // points_per_curve

    true_prices = []
    pred_prices = []

    for h in range(num_hours):
        i_start = h * points_per_curve
        mw_true = y_true[i_start:i_start+8]
        price_true = y_true[i_start+8:i_start+16]
        mw_pred = y_pred[i_start:i_start+8]
        price_pred = y_pred[i_start+8:i_start+16]

        sorted_idx = np.argsort(mw_true)
        true_prices.extend(np.array(price_true)[sorted_idx])
        pred_prices.extend(np.array(price_pred)[sorted_idx])

    true_prices = np.array(true_prices)
    pred_prices = np.array(pred_prices)

    avg_r2 = r2_score(true_prices, pred_prices)
    avg_rmse = np.sqrt(mean_squared_error(true_prices, pred_prices))
    within_15 = np.mean(np.abs(pred_prices - true_prices) / np.abs(true_prices) <= 0.15)

    st.subheader(f"\ud83d\udcc8 Forecast Accuracy for {generator}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average R²", f"{avg_r2:.3f}")
    col2.metric("Average RMSE", f"{avg_rmse:.2f}")
    col3.metric("% within ±15%", f"{within_15*100:.1f}%")

    # === NEW: Add spinner during plotting
    with st.spinner('\ud83c\udf00 Generating forecast plots...'):
        plot_offer_curves_with_metrics(
            y_true,
            y_pred,
            start_time,
            scroll_view=(view_mode == "Scrollable (1 row, wide plots)")
        )

    # === Download buttons after plot
    excel_single = convert_single_prediction_to_excel(y_pred, y_true, start_time, generator)
    st.download_button(
        label="\u2b07\ufe0f Download Current Generator Predictions (Actual vs Predicted)",
        data=excel_single,
        file_name=f"{generator}_forecast.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    excel_all = convert_all_predictions_to_excel(all_preds, Y, resource_names, timestamps)
    st.download_button(
        label="\u2b07\ufe0f Download All Generators (Actual vs Predicted, All Tabs + Combined)",
        data=excel_all,
        file_name="all_forecasts.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
