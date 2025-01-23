import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import tensorflow as tf



# ------------------------------------------
# A) Paths + Constants
# ------------------------------------------
INVOICE_FOLDER = "ocr_output"   # e.g. invoice_1_output.txt
LABEL_FOLDER   = "txt_labels"           # e.g. label_1.txt
NUM_SAMPLES    = 399

# Serial No. range: [2,000,000..2,999,999] => scale to [0..1]
SERIAL_MIN   = 2_000_000
SERIAL_RANGE = 1_000_000  # => max 2,999,999

# For day/month/year classification:
# day => [1..31], month => [1..12], year => [2010..2024] => 15 classes

# ------------------------------------------
# B) Parse Label File
# ------------------------------------------
def parse_label_file(label_path):
    """
    Example label:
        Serial No.: 2476649
        date: 22-11-2020
        Sales Tax: 3601.0
        Net Tax Inclusive Value: 23601.0

    Returns dict with:
      serial_no (int), day (int), month (int), year (int),
      sales_tax (float), net_tax (float)
    """
    data = {
        'serial_no': None,
        'day': None,
        'month': None,
        'year': None,
        'sales_tax': None,
        'net_tax': None
    }
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for line in lines:
        low = line.lower()
        if low.startswith("serial no.:"):
            val = line.split(":", 1)[1].strip()
            data['serial_no'] = int(val)
        elif low.startswith("date:"):
            val = line.split(":", 1)[1].strip()
            # Expect dd-mm-yyyy
            dt = datetime.strptime(val, "%d-%m-%Y")
            data['day']   = dt.day
            data['month'] = dt.month
            data['year']  = dt.year
        elif low.startswith("sales tax:"):
            val = line.split(":", 1)[1].strip()
            data['sales_tax'] = float(val)
        elif low.startswith("net tax inclusive value:"):
            val = line.split(":", 1)[1].strip()
            data['net_tax'] = float(val)

    return data

# ------------------------------------------
# C) Read All Data (X_text, label fields)
# ------------------------------------------
X_texts = []
serial_nums = []
day_vals    = []
month_vals  = []
year_vals   = []
sales_vals  = []
net_vals    = []

for i in range(1, NUM_SAMPLES + 1):
    invoice_path = os.path.join(INVOICE_FOLDER, f"invoice_{i}_output.txt")
    label_path   = os.path.join(LABEL_FOLDER,   f"label_{i}.txt")

    if not os.path.isfile(invoice_path) or not os.path.isfile(label_path):
        continue

    # OCR text
    with open(invoice_path, "r", encoding="utf-8") as f:
        text = f.read()

    labels = parse_label_file(label_path)
    # If any field is None, skip
    if any(labels[k] is None for k in labels):
        continue

    s_no = labels['serial_no']
    d    = labels['day']
    m    = labels['month']
    y    = labels['year']
    st   = labels['sales_tax']
    nt   = labels['net_tax']

    # Filter out-of-range
    if not (2_000_000 <= s_no <= 2_999_999):
        continue
    if not (1 <= d <= 31 and 1 <= m <= 12 and 2010 <= y <= 2024):
        continue

    X_texts.append(text)
    serial_nums.append(s_no)
    day_vals.append(d)
    month_vals.append(m)
    year_vals.append(y)
    sales_vals.append(st)
    net_vals.append(nt)

X_texts   = np.array(X_texts)
serial_np = np.array(serial_nums, dtype=np.int32)
day_np    = np.array(day_vals,    dtype=np.int32)
month_np  = np.array(month_vals,  dtype=np.int32)
year_np   = np.array(year_vals,   dtype=np.int32)
sales_np  = np.array(sales_vals,  dtype=np.float32)
net_np    = np.array(net_vals,    dtype=np.float32)

print("Loaded samples:", len(X_texts))

# ------------------------------------------
# D) Train/Val/Test Split
# ------------------------------------------
X_tv, X_test, serial_tv, serial_test, \
day_tv, day_test, \
mon_tv, mon_test, \
year_tv, year_test, \
sales_tv, sales_test, \
net_tv, net_test = train_test_split(
    X_texts, serial_np, day_np, month_np, year_np, sales_np, net_np,
    test_size=30, random_state=42
)

X_train, X_val, serial_train, serial_val, \
day_train, day_val, \
mon_train, mon_val, \
year_train, year_val, \
sales_train, sales_val, \
net_train, net_val = train_test_split(
    X_tv, serial_tv, day_tv, mon_tv, year_tv, sales_tv, net_tv,
    test_size=49, random_state=42
)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ------------------------------------------
# E) Scale Large Numeric Outputs [0..1]
# ------------------------------------------
def scale_serial(x):
    # [2e6..3e6] => [0..1]
    return (x - 2_000_000) / 1_000_000

def inv_scale_serial(x):
    return (x * 1_000_000) + 2_000_000

serial_train_scaled = scale_serial(serial_train)
serial_val_scaled   = scale_serial(serial_val)
serial_test_scaled  = scale_serial(serial_test)

# For Sales/Net, use train min/max
sales_min = sales_train.min()
sales_max = sales_train.max()
net_min   = net_train.min()
net_max   = net_train.max()

def scale_sales(x):
    return (x - sales_min) / (sales_max - sales_min + 1e-8)

def inv_scale_sales(x):
    return x * (sales_max - sales_min + 1e-8) + sales_min

def scale_net(x):
    return (x - net_min) / (net_max - net_min + 1e-8)

def inv_scale_net(x):
    return x * (net_max - net_min + 1e-8) + net_min

sales_train_scaled = scale_sales(sales_train)
sales_val_scaled   = scale_sales(sales_val)
sales_test_scaled  = scale_sales(sales_test)

net_train_scaled = scale_net(net_train)
net_val_scaled   = scale_net(net_val)
net_test_scaled  = scale_net(net_test)

# Convert day, month, year => classification indices
day_train_idx   = day_train - 1
month_train_idx = mon_train - 1
year_train_idx  = year_train - 2010

day_val_idx   = day_val - 1
month_val_idx = mon_val - 1
year_val_idx  = year_val - 2010

day_test_idx   = day_test - 1
month_test_idx = mon_test - 1
year_test_idx  = year_test - 2010

# ------------------------------------------
# F) TF–IDF Vectorization
# ------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

# Try a smaller max_features to reduce dimensionality & help generalization
tfidf = TfidfVectorizer(max_features=1000,  # was 2000
                        stop_words='english',
                        lowercase=True,
                        ngram_range=(1, 3))
tfidf.fit(X_train)

X_train_tfidf = tfidf.transform(X_train).toarray()
X_val_tfidf   = tfidf.transform(X_val).toarray()
X_test_tfidf  = tfidf.transform(X_test).toarray()

# ------------------------------------------
# G) Build Multi-Output Model (with Dropout)
# ------------------------------------------
from tensorflow.keras import layers, models, regularizers

inp = tf.keras.Input(shape=(X_train_tfidf.shape[1],))

# Let's reduce hidden layer sizes & add dropout
x = layers.Dense(32, activation='relu')(inp)
x = layers.Dropout(0.3)(x)  # 30% dropout
x = layers.Dense(16, activation='relu')(x)
x = layers.Dropout(0.3)(x)

out_serial = layers.Dense(1, name='serial_no')(x)  # MSE

out_day   = layers.Dense(31, activation='softmax', name='day')(x)
out_month = layers.Dense(12, activation='softmax', name='month')(x)
out_year  = layers.Dense(15, activation='softmax', name='year')(x)

out_sales = layers.Dense(1, name='sales_tax')(x)  # MSE
out_net   = layers.Dense(1, name='net_tax')(x)    # MSE

model = tf.keras.Model(
    inputs=inp,
    outputs=[out_serial, out_day, out_month, out_year, out_sales, out_net]
)

model.compile(
    loss={
      'serial_no': 'mse',
      'day': 'sparse_categorical_crossentropy',
      'month': 'sparse_categorical_crossentropy',
      'year': 'sparse_categorical_crossentropy',
      'sales_tax': 'mse',
      'net_tax': 'mse'
    },
    loss_weights={
      'serial_no': 1.0,
      'day': 1.0,
      'month': 1.0,
      'year': 1.0,
      'sales_tax': 1.0,
      'net_tax': 1.0
    },
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics={
      'serial_no': ['mae'],
      'day':       ['accuracy'],
      'month':     ['accuracy'],
      'year':      ['accuracy'],
      'sales_tax': ['mae'],
      'net_tax':   ['mae']
    }
)

model.summary()

# ------------------------------------------
# H) Train with Early Stopping
# ------------------------------------------
EPOCHS = 200
BATCH_SIZE = 16

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,        # Stop if val_loss doesn't improve for 10 epochs
    restore_best_weights=True
)

history = model.fit(
    X_train_tfidf,
    {
      'serial_no': serial_train_scaled,
      'day': day_train_idx,
      'month': month_train_idx,
      'year': year_train_idx,
      'sales_tax': sales_train_scaled,
      'net_tax': net_train_scaled
    },
    validation_data=(
        X_val_tfidf,
        {
          'serial_no': serial_val_scaled,
          'day': day_val_idx,
          'month': month_val_idx,
          'year': year_val_idx,
          'sales_tax': sales_val_scaled,
          'net_tax': net_val_scaled
        }
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop]  # attach early stopping
)


# ------------------------------------------
# I) Evaluate on Test + Predictions
# ------------------------------------------
test_results = model.evaluate(
    X_test_tfidf,
    {
      'serial_no': serial_test_scaled,
      'day': day_test_idx,
      'month': month_test_idx,
      'year': year_test_idx,
      'sales_tax': sales_test_scaled,
      'net_tax': net_test_scaled
    },
    return_dict=True
)
print("\nTest Results:", test_results)

preds = model.predict(X_test_tfidf)
pred_serial_scaled = preds[0].reshape(-1)  # [0..1]
pred_day_probs   = preds[1]
pred_month_probs = preds[2]
pred_year_probs  = preds[3]
pred_sales_scaled = preds[4].reshape(-1)
pred_net_scaled   = preds[5].reshape(-1)

# ------------------------------------------
# J) Postprocess + Show Table
# ------------------------------------------
final_rows = []
for i in range(len(pred_serial_scaled)):
    # 1) inverse-scale serial
    s_val = inv_scale_serial(pred_serial_scaled[i])
    s_int = int(round(s_val))
    if s_int < 2000000: s_int = 2000000
    if s_int > 2999999: s_int = 2999999

    # 2) day/month/year
    d_idx = np.argmax(pred_day_probs[i])
    day_pred = d_idx + 1
    m_idx = np.argmax(pred_month_probs[i])
    month_pred = m_idx + 1
    y_idx = np.argmax(pred_year_probs[i])
    year_pred = y_idx + 2010

    # 3) sales/net
    st_val = inv_scale_sales(pred_sales_scaled[i])
    nt_val = inv_scale_net(pred_net_scaled[i])
    st_rounded = int(round(st_val))
    nt_rounded = int(round(nt_val))

    # True references
    t_s_val = inv_scale_serial(serial_test_scaled[i])
    t_s_int = int(round(t_s_val))
    if t_s_int < 2000000: t_s_int = 2000000
    if t_s_int > 2999999: t_s_int = 2999999

    true_d = (day_test_idx[i] + 1)
    true_m = (month_test_idx[i] + 1)
    true_y = (year_test_idx[i] + 2010)

    t_st_val = inv_scale_sales(sales_test_scaled[i])
    t_st_rounded = int(round(t_st_val))
    t_nt_val = inv_scale_net(net_test_scaled[i])
    t_nt_rounded = int(round(t_nt_val))

    row = {
        'pred_serial_no': str(s_int),
        'true_serial_no': str(t_s_int),
        'pred_day': day_pred, 'true_day': true_d,
        'pred_month': month_pred, 'true_month': true_m,
        'pred_year': year_pred, 'true_year': true_y,
        'pred_sales_tax': f"{st_rounded}.0",
        'true_sales_tax': f"{t_st_rounded}.0",
        'pred_net_tax': f"{nt_rounded}.0",
        'true_net_tax': f"{t_nt_rounded}.0"
    }
    final_rows.append(row)

df_test = pd.DataFrame(final_rows)
print("\nTest Predictions (sample):")
print(df_test.head(5))

df_test.to_csv("test_predictions_scaled.csv", index=False)
print("Saved: test_predictions_scaled.csv")

# Print training/validation loss
hist_df = pd.DataFrame(history.history)

print("\nEpoch | Train Loss   | Val Loss")
print("------+--------------+---------")
for epoch, (t_loss, v_loss) in enumerate(zip(hist_df['loss'], hist_df['val_loss']), start=1):
    print(f"{epoch:5d} | {t_loss:12.6f} | {v_loss:12.6f}")

# Save to text
output_path = 'epoch_losses_scaled.txt'
with open(output_path, 'w') as f:
    f.write("Epoch | Train Loss   | Val Loss\n")
    f.write("------+--------------+---------\n")
    for epoch, (t_loss, v_loss) in enumerate(zip(hist_df['loss'], hist_df['val_loss']), start=1):
        f.write(f"{epoch:5d} | {t_loss:12.6f} | {v_loss:12.6f}\n")

print(f"Data exported to {output_path}.")
