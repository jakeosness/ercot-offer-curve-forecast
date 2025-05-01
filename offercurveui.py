# offer_curve_app.py (with sidebar + right-aligned metrics)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import base64
from io import BytesIO
import streamlit.components.v1 as components
import matplotlib.gridspec as gridspec

# === Load Data
X_parts = [
    np.load("./Model_Info/X_sample_part1.npy"),
    np.load("./Model_Info/X_sample_part2.npy"),
    np.load("./Model_Info/X_sample_part3.npy"),
]
X = np.concatenate(X_parts, axis=0)
Y = np.load("./Model_Info/Y_sample.npy")
all_preds = np.load("./Model_Info/predictions.npy")
timestamps = np.load("./Model_Info/timestamps.npy", allow_pickle=True)
df = pd.read_csv("./Model_Info/df_sample.csv")

# Extract generator names
resource_names = [df.loc[i + 2016, 'Resource Name'] for i in range(len(X))]
unique_generators = sorted(set(resource_names))

# === Sidebar ===
st.sidebar.header("Controls")
generator = st.sidebar.selectbox("Choose Generator:", unique_generators)
view_mode = st.sidebar.radio("Display Mode", ["Scrollable", "Grid"])

# Match Generator to Sample
sample_indices = [i for i in range(len(X)) if df.loc[i + 2016, 'Resource Name'] == generator]

if not sample_indices:
    st.warning(f"No forecast samples found for {generator}.")
else:
    idx = sample_indices[0]
    y_pred = all_preds[idx]
    y_true = Y[idx]
    start_time = pd.to_datetime(timestamps[idx])

    # Compute global metrics
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

    avg_r2 = r2_score(true_prices, pred_prices)
    avg_rmse = np.sqrt(mean_squared_error(true_prices, pred_prices))
    within_15 = np.mean(np.abs(np.array(pred_prices) - np.array(true_prices)) / np.abs(true_prices) <= 0.15)

    # === Main title
    st.markdown("<h3 style='text-align: center;'>ERCOT Generator Offer Curve Forecast</h3>", unsafe_allow_html=True)

    # === Main + Right Layout
    col_main, col_metrics = st.columns([3, 1])

    # === Forecast plots
    with col_main:
        with st.spinner("Generating forecast plots..."):
            def plot_offer_curves_with_metrics(true, pred, start_time, scroll_view=True):
                for day in range(3):
                    if scroll_view:
                        fig, axes = plt.subplots(1, 24, figsize=(6 * 24, 6))
                        for i, h in enumerate(range(day * 24, (day + 1) * 24)):
                            ax = axes[i]
                            i_start = h * 16
                            mw_true = true[i_start:i_start+8]
                            price_true = true[i_start+8:i_start+16]
                            mw_pred = pred[i_start:i_start+8]
                            price_pred = pred[i_start+8:i_start+16]
                            sorted_idx = np.argsort(mw_true)
                            mw_sorted = np.array(mw_true)[sorted_idx]
                            price_true_sorted = np.array(price_true)[sorted_idx]
                            price_pred_sorted = np.array(price_pred)[sorted_idx]
                            cum_mw = np.cumsum(mw_sorted)
                            r2 = r2_score(price_true_sorted, price_pred_sorted)
                            ax.plot(cum_mw, price_true_sorted, 'r--x', label='Actual')
                            ax.plot(cum_mw, price_pred_sorted, 'b-o', label='Predicted')
                            ax.set_title(f"{(start_time + pd.Timedelta(hours=h)).strftime('%m-%d %H:%M')}\nR²: {r2:.2f}", fontsize=8)
                            ax.set_xlabel("MW", fontsize=7)
                            ax.set_ylabel("Price", fontsize=7)
                            ax.tick_params(labelsize=6)
                        fig.tight_layout()
                        buf = BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight")
                        buf.seek(0)
                        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
                        buf.close()
                        components.html(
                            f"""
                            <div style="overflow-x:auto; width:100%">
                                <img src="data:image/png;base64,{img_base64}" style="width:10000px"/>
                            </div>
                            """,
                            height=500,
                            scrolling=True
                        )
                    else:
                        fig = plt.figure(figsize=(28, 18))
                        gs = gridspec.GridSpec(4, 6, figure=fig)
                        for i, h in enumerate(range(day * 24, (day + 1) * 24)):
                            i_start = h * 16
                            mw_true = true[i_start:i_start+8]
                            price_true = true[i_start+8:i_start+16]
                            mw_pred = pred[i_start:i_start+8]
                            price_pred = pred[i_start+8:i_start+16]
                            sorted_idx = np.argsort(mw_true)
                            mw_sorted = np.array(mw_true)[sorted_idx]
                            price_true_sorted = np.array(price_true)[sorted_idx]
                            price_pred_sorted = np.array(price_pred)[sorted_idx]
                            cum_mw = np.cumsum(mw_sorted)
                            r2 = r2_score(price_true_sorted, price_pred_sorted)
                            ax = fig.add_subplot(gs[i])
                            ax.plot(cum_mw, price_true_sorted, 'r--x', label='Actual')
                            ax.plot(cum_mw, price_pred_sorted, 'b-o', label='Predicted')
                            ax.set_title(f"{(start_time + pd.Timedelta(hours=h)).strftime('%m-%d %H:%M')}\nR²: {r2:.2f}", fontsize=10)
                            ax.set_xlabel("Cumulative MW", fontsize=9)
                            ax.set_ylabel("Price", fontsize=9)
                            ax.tick_params(labelsize=8)
                            ax.legend(fontsize=7)
                        plt.tight_layout()
                        st.pyplot(fig)
            plot_offer_curves_with_metrics(y_true, y_pred, start_time, scroll_view=(view_mode == "Scrollable"))

    # === Metrics block (far right)
    with col_metrics:
        st.subheader(f"Forecast Accuracy for {generator}")
        st.metric("Average R²", f"{avg_r2:.3f}")
        st.metric("Average RMSE", f"{avg_rmse:.2f}")
        st.metric("% within ±15%", f"{within_15*100:.1f}%")

    # === Download buttons (in sidebar)
    def convert_single_prediction_to_excel(y_pred, y_true, timestamp, generator):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        points_per_curve = 16
        hours = len(y_pred) // points_per_curve
        timestamps_list = pd.date_range(pd.to_datetime(timestamp), periods=hours, freq='1h')
        rows = []
        for i in range(hours):
            mw_true = y_true[i * points_per_curve : i * points_per_curve + 8]
            price_true = y_true[i * points_per_curve + 8 : (i + 1) * points_per_curve]
            mw_pred = y_pred[i * points_per_curve : i * points_per_curve + 8]
            price_pred = y_pred[i * points_per_curve + 8 : (i + 1) * points_per_curve]
            row = [generator, timestamps_list[i]] + list(mw_true) + list(mw_pred) + list(price_true) + list(price_pred)
            rows.append(row)
        columns = ["Generator Name", "Timestamp"] + [f"MW Point {i+1} (Actual)" for i in range(8)] + [f"MW Point {i+1} (Predicted)" for i in range(8)] + [f"Price Point {i+1} (Actual)" for i in range(8)] + [f"Price Point {i+1} (Predicted)" for i in range(8)]
        df_pred = pd.DataFrame(rows, columns=columns)
        df_pred.to_excel(writer, sheet_name=str(generator)[:31], index=False)
        writer.close()
        output.seek(0)
        return output

    def convert_all_predictions_to_excel(all_preds, all_true, resource_names, timestamps):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        all_rows = []
        for idx in range(len(all_preds)):
            generator = resource_names[idx]
            y_pred = all_preds[idx]
            y_true = all_true[idx]
            timestamp = pd.to_datetime(timestamps[idx])
            if len(y_pred) % 16 != 0 or len(y_true) % 16 != 0:
                continue
            hours = len(y_pred) // 16
            ts_range = pd.date_range(timestamp, periods=hours, freq='1h')
            rows = []
            for i in range(hours):
                mw_true = y_true[i * 16 : i * 16 + 8]
                price_true = y_true[i * 16 + 8 : (i + 1) * 16]
                mw_pred = y_pred[i * 16 : i * 16 + 8]
                price_pred = y_pred[i * 16 + 8 : (i + 1) * 16]
                row = [generator, ts_range[i]] + list(mw_true) + list(mw_pred) + list(price_true) + list(price_pred)
                rows.append(row)
            columns = ["Generator Name", "Timestamp"] + [f"MW Point {i+1} (Actual)" for i in range(8)] + [f"MW Point {i+1} (Predicted)" for i in range(8)] + [f"Price Point {i+1} (Actual)" for i in range(8)] + [f"Price Point {i+1} (Predicted)" for i in range(8)]
            df_pred = pd.DataFrame(rows, columns=columns)
            df_pred.to_excel(writer, sheet_name=str(generator)[:31], index=False)
            all_rows.extend(rows)
        if all_rows:
            df_all = pd.DataFrame(all_rows, columns=columns)
            df_all.to_excel(writer, sheet_name="All_Generators", index=False)
        writer.close()
        output.seek(0)
        return output

    excel_single = convert_single_prediction_to_excel(y_pred, y_true, start_time, generator)
    st.sidebar.download_button("Download Current Generator Predictions", data=excel_single, file_name=f"{generator}_forecast.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    excel_all = convert_all_predictions_to_excel(all_preds, Y, resource_names, timestamps)
    st.sidebar.download_button("Download All Generators Forecasts", data=excel_all, file_name="all_forecasts.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
