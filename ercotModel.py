# ✅ STEP 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model


# Read your data file
input_path = "./Model_Info/SCED_With_Gas_And_Load-2024OctToDec.csv"
df = pd.read_csv(input_path)

# Condition 1: 'Submitted TPO-Price8' is not null
df_valid = df[df['Submitted TPO-Price8'].notna()]

# Condition 2: Each resource must have at least 2000 records
resource_counts = df_valid['Resource Name'].value_counts()
qualified_resources = resource_counts[resource_counts >= 2000].reset_index()
qualified_resources.columns = ['Resource Name', 'Valid Record Count']

# Show results
print(qualified_resources)

# ✅ STEP 2: Load data
df = pd.read_csv(input_path)
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['SCED Time Stamp'])
df = df.sort_values('Datetime').reset_index(drop=True)

# ✅ STEP 3: Filter + Clean
df = df.drop(columns=['Submitted TPO-MW9', 'Submitted TPO-MW10',
                      'Submitted TPO-Price9', 'Submitted TPO-Price10'])

# ✅ STEP 4: Add features
df['Hour'] = df['Datetime'].dt.hour
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
for col in ['Submitted TPO-Price8', 'Submitted TPO-MW8']:
    df[f'{col}_ma3'] = df[col].rolling(window=3, min_periods=1).mean()
    df[f'{col}_diff1'] = df[col].diff().fillna(0)

# ✅ STEP 5: Setup columns
mw_cols = [f'Submitted TPO-MW{i}' for i in range(1, 9)]
price_cols = [f'Submitted TPO-Price{i}' for i in range(1, 9)]
target_cols = mw_cols + price_cols
exclude_cols = ['Date', 'SCED Time Stamp', 'Datetime', 'Resource Name'] + target_cols
feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype != 'object']
df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=feature_cols + target_cols)

# ✅ STEP 6: Create sequences for 2016 input hours and 72 output
def create_XY(df, feature_cols, mw_cols, price_cols, input_hours=2016, forecast_hours=72):
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

# ✅ NEW: Sequence creation per generator
X_all, Y_all, timestamps_all = [], [], []

for gen_name, group in df.groupby("Resource Name"):
    group = group.sort_values("Datetime")

    if len(group) < 2016 + 72:
        continue  # skip generators with not enough data

    X, Y, timestamps = create_XY(group, feature_cols, mw_cols, price_cols)
    X_all.append(X)
    Y_all.append(Y)
    timestamps_all.extend(timestamps)

# Concatenate across generators
X_all = np.concatenate(X_all)
Y_all = np.concatenate(Y_all)

# Save outputs
np.save("./Model_Info/X_sample.npy", X_all)
np.save("./Model_Info/Y_sample.npy", Y_all)
np.save("./Model_Info/timestamps.npy", timestamps_all)

# Optional: Save aligned df as well
df.to_csv("./Model_Info/df_sample.csv", index=False)

# ✅ STEP 7: Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_all.reshape(-1, X_all.shape[-1])).reshape(X_all.shape)

# ✅ STEP 8: Build Transformer model
def transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.1):
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization()(x)
    x_ff = Dense(ff_dim, activation="relu")(x)
    x_ff = Dropout(dropout)(x_ff)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x = Add()([x, x_ff])
    return LayerNormalization()(x)

input_layer = Input(shape=(X_scaled.shape[1], X_scaled.shape[2]))
x = transformer_encoder(input_layer)
x = transformer_encoder(x)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
output_layer = Dense(Y_all.shape[1])(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.summary()

# ✅ STEP 9: Train model
model.fit(X_scaled, Y_all, epochs=50, batch_size=8, validation_split=0.1,
          callbacks=[EarlyStopping(patience=8, restore_best_weights=True)])

# ✅ STEP 10: Predict last 72 hours
sample_x = X_scaled[-1].reshape(1, X_scaled.shape[1], X_scaled.shape[2])
pred = model.predict(sample_x)[0]
true = Y_all[-1]
start_time = timestamps_all[-1] + pd.Timedelta(hours=1)

# ✅ STEP 10.5: Predict for all generators
all_preds = model.predict(X_scaled, verbose=1)
np.save("./Model_Info/predictions.npy", all_preds)

# ✅ STEP 11: Plot results
from pandas import DataFrame
results = []
for h in range(72):
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
    rmse = np.sqrt(mean_squared_error(price_true_sorted, price_pred_sorted))

    for p in range(8):
        results.append({
            'Datetime': start_time + pd.Timedelta(hours=h),
            'Hour': h + 1,
            'Point': p + 1,
            'Cumulative MW': cum_mw[p],
            'Price Actual': price_true_sorted[p],
            'Price Predicted': price_pred_sorted[p],
            'R2': r2,
            'RMSE': rmse
        })

    # Plot
    plt.figure(figsize=(5, 4))
    plt.plot(cum_mw, price_true_sorted, 'r--x', label='Actual')
    plt.plot(cum_mw, price_pred_sorted, 'b-o', label='Predicted')
    plt.title(f"{(start_time + pd.Timedelta(hours=h)).strftime('%Y-%m-%d %H:%M')}\nR²: {r2:.3f}")
    plt.xlabel("Cumulative MW")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ✅ STEP 12: Accuracy Summary for 72-hour prediction
from sklearn.metrics import mean_absolute_error

# Collect all true and predicted prices over the 72-hour horizon
all_true_prices = []
all_pred_prices = []

for h in range(72):
    i_start = h * 16
    price_true = true[i_start+8:i_start+16]
    price_pred = pred[i_start+8:i_start+16]

    # ➕ Sort prices based on corresponding MW
    sorted_idx = np.argsort(true[i_start:i_start+8])
    price_true_sorted = np.array(price_true)[sorted_idx]
    price_pred_sorted = np.array(price_pred)[sorted_idx]

    all_true_prices.extend(price_true_sorted)
    all_pred_prices.extend(price_pred_sorted)

# ✅ Compute overall accuracy metrics
avg_r2 = r2_score(all_true_prices, all_pred_prices)
avg_rmse = np.sqrt(mean_squared_error(all_true_prices, all_pred_prices))
avg_mae = mean_absolute_error(all_true_prices, all_pred_prices)
avg_mape = np.mean(np.abs((np.array(all_true_prices) - np.array(all_pred_prices)) / np.array(all_true_prices))) * 100

# ✅ Display results
print("\nTransformer Prediction Model: 72-Hour Overall Accuracy Summary")
print(f"Resource Name : {df['Resource Name'].iloc[0]}")
print(f"Average R²        : {avg_r2:.4f}")
print(f"Average RMSE     : {avg_rmse:.2f}")
print(f"Average MAE      : {avg_mae:.2f}")
print(f"Average MAPE (%) : {avg_mape:.2f}%")

# Confirm everything before saving
print("X_scaled:", X_scaled.shape)
print("df:", df.shape)
model.summary()

# Now save again
output_dir = "./Model_Info"

np.save(f"{output_dir}/X_sample.npy", X_scaled)
np.save(f"{output_dir}/Y_sample.npy", Y_all)
df.to_csv(f"{output_dir}/df_sample.csv", index=False)
model.save(f"{output_dir}/model.keras")
# Save timestamps used for each sample
np.save(f"{output_dir}/timestamps.npy", np.array(timestamps_all))  # where timestamps[idx] = forecast start time

print("✅ Saved everything to disk.")