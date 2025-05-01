# trainModel.py — shared Transformer model with time-based validation split

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf
from ercotModel import build_model, make_predictions
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

print("[INFO] Available GPUs:", tf.config.list_physical_devices('GPU'))

# === Step 1: Load and Filter Data ===
print("[INFO] Loading and filtering Q3 data...")
df = pd.read_csv("./Model_Info/SCED_FullYear_2024.csv")
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['SCED Time Stamp'])
df = df.sort_values('Datetime').reset_index(drop=True)
df = df[(df['Datetime'] >= '2024-07-01') & (df['Datetime'] < '2024-10-01')]

# === Step 2: Feature Engineering ===
print("[INFO] Engineering features...")
cols_to_drop = ['Submitted TPO-MW9', 'Submitted TPO-MW10', 'Submitted TPO-Price9', 'Submitted TPO-Price10']
df = df.drop(columns=cols_to_drop)
df['Hour'] = df['Datetime'].dt.hour
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

for col in ['Submitted TPO-Price8', 'Submitted TPO-MW8']:
    df[f'{col}_ma3'] = df[col].rolling(window=3, min_periods=1).mean()
    df[f'{col}_diff1'] = df[col].diff().fillna(0)

mw_cols = [f'Submitted TPO-MW{i}' for i in range(1, 9)]
price_cols = [f'Submitted TPO-Price{i}' for i in range(1, 9)]
target_cols = mw_cols + price_cols
exclude_cols = ['Date', 'SCED Time Stamp', 'Datetime', 'Resource Name'] + target_cols
feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype != 'object']
df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=feature_cols + target_cols)

df.to_csv("./Model_Info/df_sample_q3.csv", index=False)

# === Step 3: Sequence Creation ===
print("[INFO] Creating sequences for all generators...")
input_hours = 672
forecast_hours = 72
min_records_required = 1000

generator_list = sorted(df['Resource Name'].unique())
gen_to_id = {name: idx for idx, name in enumerate(generator_list)}
num_generators = len(generator_list)

def create_XY(df, feature_cols, mw_cols, price_cols, input_hours, forecast_hours):
    X, Y, timestamps, names, ids = [], [], [], [], []
    for gen_name, group in df.groupby("Resource Name"):
        group = group.sort_values("Datetime")
        if len(group) < max(input_hours + forecast_hours, min_records_required):
            continue
        gen_id = gen_to_id[gen_name]
        for i in range(len(group) - input_hours - forecast_hours):
            x_block = group.iloc[i:i+input_hours][feature_cols].values
            y_block = []
            for j in range(forecast_hours):
                row = group.iloc[i + input_hours + j]
                y_block.extend(row[mw_cols])
                y_block.extend(row[price_cols])
            X.append(x_block)
            Y.append(y_block)
            timestamps.append(group.iloc[i + input_hours]['Datetime'])
            names.append(gen_name)
            ids.append(gen_id)
    return np.array(X), np.array(Y), np.array(timestamps), np.array(names), np.array(ids)

# === Step 4: Time-based Train/Val Split ===
def time_based_split(X, Y, IDs, val_frac=0.2):
    X_train, Y_train, ID_train = [], [], []
    X_val, Y_val, ID_val = [], [], []

    gen_ids = np.unique(IDs)
    for gen_id in gen_ids:
        gen_mask = (IDs == gen_id)
        X_gen = X[gen_mask]
        Y_gen = Y[gen_mask]
        ID_gen = IDs[gen_mask]

        n_val = int(len(X_gen) * val_frac)
        if n_val == 0:
            continue

        X_train.append(X_gen[:-n_val])
        Y_train.append(Y_gen[:-n_val])
        ID_train.append(ID_gen[:-n_val])

        X_val.append(X_gen[-n_val:])
        Y_val.append(Y_gen[-n_val:])
        ID_val.append(ID_gen[-n_val:])

    return (
        np.concatenate(X_train), np.concatenate(Y_train), np.concatenate(ID_train),
        np.concatenate(X_val), np.concatenate(Y_val), np.concatenate(ID_val)
    )

X_all, Y_all, TS_all, GEN_all, ID_all = create_XY(df, feature_cols, mw_cols, price_cols, input_hours, forecast_hours)
print(f"[INFO] ✅ Created {len(X_all)} sequences across {len(np.unique(GEN_all))} generators.")

X_train, Y_train, ID_train, X_val, Y_val, ID_val = time_based_split(X_all, Y_all, ID_all)

# === Step 5: Train Shared Transformer Model ===
print("[INFO] Training shared Transformer model...")
input_shape = X_all.shape[1:]
output_dim = Y_all.shape[1]
model = build_model(input_shape, output_dim, num_generators=num_generators)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath="./Model_Info/best_shared_model.keras",
    monitor='val_loss',
    save_best_only=True,
    verbose=1,
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    [X_train, ID_train], Y_train,
    validation_data=([X_val, ID_val], Y_val),
    epochs=20,
    batch_size=64,
    callbacks=[early_stop, checkpoint, lr_scheduler],
    verbose=1
)

# === Step 6: Make Predictions and Save Outputs ===
print("[INFO] Saving predictions and model...")
preds = make_predictions(model, X_all, ID_all)

np.save("./Model_Info/X_sample.npy", X_all)
np.save("./Model_Info/Y_sample.npy", Y_all)
np.save("./Model_Info/predictions.npy", preds)
np.save("./Model_Info/timestamps.npy", TS_all)
np.save("./Model_Info/resource_names.npy", GEN_all)
np.save("./Model_Info/generator_ids.npy", ID_all)
model.save("./Model_Info/ercot_shared_model.keras")

print("[INFO] ✅ Shared model training complete and saved.")
