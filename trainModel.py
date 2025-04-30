# train_model_full_year.py

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# === STEP 1: Load full-year 2024 data ===
print("[INFO] Step 1: Loading full-year 2024 data...")
df = pd.read_csv("./Model_Info/SCED_FullYear_2024.csv")
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['SCED Time Stamp'])
df = df.sort_values('Datetime').reset_index(drop=True)

# === STEP 2: Feature engineering ===
print("[INFO] Step 2: Feature engineering...")
# Drop columns with high nulls
cols_to_drop = ['Submitted TPO-MW9', 'Submitted TPO-MW10', 'Submitted TPO-Price9', 'Submitted TPO-Price10']
df = df.drop(columns=cols_to_drop)

# Time-based features
df['Hour'] = df['Datetime'].dt.hour
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

# Rolling features
for col in ['Submitted TPO-Price8', 'Submitted TPO-MW8']:
    df[f'{col}_ma3'] = df[col].rolling(window=3, min_periods=1).mean()
    df[f'{col}_diff1'] = df[col].diff().fillna(0)

# Define targets (MW1–MW8, Price1–Price8)
mw_cols = [f'Submitted TPO-MW{i}' for i in range(1, 9)]
price_cols = [f'Submitted TPO-Price{i}' for i in range(1, 9)]
target_cols = mw_cols + price_cols

# Define features
exclude_cols = ['Date', 'SCED Time Stamp', 'Datetime', 'Resource Name'] + target_cols
feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype != 'object']
df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=feature_cols + target_cols)

print("[INFO] Saving cleaned snapshot...")
df.to_csv("./Model_Info/df_sample.csv", index=False)

# === STEP 3: Sequence creation across entire 2024 ===
print("[INFO] Step 3: Creating sequences for full-year training...")
output_dir = "./Model_Info/sequences"
os.makedirs(output_dir, exist_ok=True)

input_hours = 2016
forecast_hours = 72

def create_XY(df, feature_cols, mw_cols, price_cols, input_hours, forecast_hours):
    X, Y, timestamps = [], [], []
    for i in range(len(df) - input_hours - forecast_hours):
        x_block = df.iloc[i:i+input_hours][feature_cols].values
        y_block = []
        for j in range(forecast_hours):
            row = df.iloc[i + input_hours + j]
            y_block.extend(row[mw_cols])
            y_block.extend(row[price_cols])
        X.append(x_block)
        Y.append(y_block)
        timestamps.append(df.iloc[i + input_hours]['Datetime'])
    return np.array(X), np.array(Y), timestamps

num_saved = 0
for gen_name, group in tqdm(df.groupby("Resource Name"), desc="Generators"):
    group = group.sort_values("Datetime")
    if len(group) >= input_hours + forecast_hours:
        X, Y, timestamps = create_XY(group, feature_cols, mw_cols, price_cols, input_hours, forecast_hours)
        np.save(f"{output_dir}/{gen_name}_X_train.npy", X)
        np.save(f"{output_dir}/{gen_name}_Y_train.npy", Y)
        np.save(f"{output_dir}/{gen_name}_timestamps_train.npy", np.array(timestamps))
        num_saved += 1

print(f"[INFO] ✅ Sequence creation complete. {num_saved} generators saved to disk.")
