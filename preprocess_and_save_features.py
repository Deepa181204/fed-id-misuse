# # preprocess_and_save_features.py
# import os, numpy as np, pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from datetime import datetime

# ROOT = r"C:\Users\MANGIPUDI DEEPA\Desktop\fed-id-misuse"
# DATA_DIR = os.path.join(ROOT, "sample_data")
# OUT_DIR = os.path.join(ROOT, "features")
# os.makedirs(OUT_DIR, exist_ok=True)

# CLIENTS = ["aadhaar_office","bank","insurance","pension","telecom","death_registrar","tax"]

# # Fit a state label encoder from aadhaar_office (if available)
# if "aadhaar_office.csv" in os.listdir(DATA_DIR):
#     adf = pd.read_csv(os.path.join(DATA_DIR, "aadhaar_office.csv"))
#     le_state = LabelEncoder()
#     if 'addr_state' in adf.columns:
#         le_state.fit(adf['addr_state'].fillna('NA').astype(str))
#     elif 'state' in adf.columns:
#         le_state.fit(adf['state'].fillna('NA').astype(str))
#     else:
#         le_state.fit(["NA"])
# else:
#     le_state = LabelEncoder()
#     le_state.fit(["NA"])

# def build_vector(row):
#     # canonical fields:
#     # age, state_code, name_len, phone_prefix,
#     # deposit_log, declared_income_log, claim_amount,
#     # sim_flag(0/1), last_txn_delta_days,
#     # has_claims, has_sim, has_death
#     vec = np.zeros(12, dtype=np.float32)
#     # age
#     if 'dob' in row and pd.notnull(row.get('dob')):
#         try:
#             age = (pd.Timestamp('today') - pd.to_datetime(row['dob'])).days // 365
#             vec[0] = max(0, age)
#         except: pass
#     # state code
#     state_val = row.get('addr_state') or row.get('state') or 'NA'
#     try:
#         vec[1] = float(le_state.transform([str(state_val)])[0])
#     except:
#         vec[1] = 0.0
#     # name length
#     if 'name' in row and pd.notnull(row.get('name')):
#         vec[2] = len(str(row['name']))
#     # phone prefix
#     if 'phone' in row and pd.notnull(row.get('phone')):
#         s = ''.join([c for c in str(row['phone']) if c.isdigit()])
#         if len(s) >= 3:
#             vec[3] = float(s[:3])
#     # deposit
#     if 'initial_deposit' in row and pd.notnull(row.get('initial_deposit')):
#         vec[4] = np.log1p(float(row['initial_deposit']))
#     # declared_income
#     if 'declared_income' in row and pd.notnull(row.get('declared_income')):
#         vec[5] = np.log1p(float(row['declared_income']))
#     # claim_amount
#     if 'claim_amount' in row and pd.notnull(row.get('claim_amount')):
#         vec[6] = float(row['claim_amount'])
#     # sim flag
#     if 'sim_id' in row and pd.notnull(row.get('sim_id')):
#         vec[7] = 1.0
#     # last_txn_delta_days
#     if 'last_txn_date' in row and pd.notnull(row.get('last_txn_date')):
#         try:
#             delta = (pd.Timestamp('today') - pd.to_datetime(row['last_txn_date'])).days
#             vec[8] = float(delta)
#         except:
#             vec[8] = 0.0
#     # masks
#     vec[9] = 1.0 if 'claim_amount' in row and pd.notnull(row.get('claim_amount')) else 0.0
#     vec[10] = 1.0 if 'sim_id' in row and pd.notnull(row.get('sim_id')) else 0.0
#     vec[11] = 1.0 if 'death_date' in row and pd.notnull(row.get('death_date')) else 0.0
#     return vec

# for c in CLIENTS:
#     csv_path = os.path.join(DATA_DIR, f"{c}.csv")
#     if not os.path.exists(csv_path):
#         print("Missing", csv_path, "- skipping")
#         continue
#     df = pd.read_csv(csv_path)
#     # build matrix
#     X = np.stack([build_vector(row) for _, row in df.iterrows()])
#     # train/val split: train on anomaly_type == 'none' if column present
#     if 'anomaly_type' in df.columns:
#         normal_mask = df['anomaly_type'] == 'none'
#         if normal_mask.sum() < 10:
#             X_train = X
#         else:
#             X_train = X[normal_mask.values]
#     else:
#         X_train = X
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_val_scaled = scaler.transform(X)
#     # save arrays & scaler
#     np.save(os.path.join(OUT_DIR, f"{c}_X_train.npy"), X_train_scaled)
#     np.save(os.path.join(OUT_DIR, f"{c}_X_val.npy"), X_val_scaled)
#     # save labels if present
#     if 'anomaly_type' in df.columns:
#         np.save(os.path.join(OUT_DIR, f"{c}_labels.npy"), df['anomaly_type'].astype(str).values)
#     # save scaler (joblib)
#     try:
#         import joblib
#         joblib.dump(scaler, os.path.join(OUT_DIR, f"{c}_scaler.joblib"))
#     except Exception as e:
#         print("joblib save failed:", e)
#     print("Saved features for", c, "->", X_train_scaled.shape, X_val_scaled.shape)



import os, numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
from sklearn.model_selection import train_test_split
import joblib

ROOT = r"C:\Users\MANGIPUDI DEEPA\Desktop\fed-id-misuse"
DATA_DIR = os.path.join(ROOT, "sample_data")
OUT_DIR = os.path.join(ROOT, "features")
os.makedirs(OUT_DIR, exist_ok=True)

CLIENTS = ["aadhaar_office","bank","insurance","pension","telecom","death_registrar","tax"]

# Fit a state label encoder from aadhaar_office if possible
if "aadhaar_office.csv" in os.listdir(DATA_DIR):
    adf = pd.read_csv(os.path.join(DATA_DIR, "aadhaar_office.csv"))
    le_state = LabelEncoder()
    if 'addr_state' in adf.columns:
        le_state.fit(adf['addr_state'].fillna('NA').astype(str))
    elif 'state' in adf.columns:
        le_state.fit(adf['state'].fillna('NA').astype(str))
    else:
        le_state.fit(["NA"])
else:
    le_state = LabelEncoder()
    le_state.fit(["NA"])

def build_vector(row):
    vec = np.zeros(12, dtype=np.float32)
    # age
    if 'dob' in row and pd.notnull(row.get('dob')):
        try:
            age = (pd.Timestamp('today') - pd.to_datetime(row['dob'])).days // 365
            vec[0] = max(0, age)
        except: pass
    # state code
    state_val = row.get('addr_state') or row.get('state') or 'NA'
    try:
        vec[1] = float(le_state.transform([str(state_val)])[0])
    except:
        vec[1] = 0.0
    # name length
    if 'name' in row and pd.notnull(row.get('name')):
        vec[2] = len(str(row['name']))
    # phone prefix
    if 'phone' in row and pd.notnull(row.get('phone')):
        s = ''.join([c for c in str(row['phone']) if c.isdigit()])
        if len(s) >= 3:
            vec[3] = float(s[:3])
    # deposit
    if 'initial_deposit' in row and pd.notnull(row.get('initial_deposit')):
        vec[4] = np.log1p(float(row['initial_deposit']))
    # declared_income
    if 'declared_income' in row and pd.notnull(row.get('declared_income')):
        vec[5] = np.log1p(float(row['declared_income']))
    # claim_amount
    if 'claim_amount' in row and pd.notnull(row.get('claim_amount')):
        vec[6] = float(row['claim_amount'])
    # sim flag
    if 'sim_id' in row and pd.notnull(row.get('sim_id')):
        vec[7] = 1.0
    # last_txn_delta_days
    if 'last_txn_date' in row and pd.notnull(row.get('last_txn_date')):
        try:
            delta = (pd.Timestamp('today') - pd.to_datetime(row['last_txn_date'])).days
            vec[8] = float(delta)
        except:
            vec[8] = 0.0
    # masks
    vec[9] = 1.0 if 'claim_amount' in row and pd.notnull(row.get('claim_amount')) else 0.0
    vec[10] = 1.0 if 'sim_id' in row and pd.notnull(row.get('sim_id')) else 0.0
    vec[11] = 1.0 if 'death_date' in row and pd.notnull(row.get('death_date')) else 0.0
    return vec

for c in CLIENTS:
    csv_path = os.path.join(DATA_DIR, f"{c}.csv")
    if not os.path.exists(csv_path):
        print("Missing", csv_path, "- skipping")
        continue

    df = pd.read_csv(csv_path)
    X = np.stack([build_vector(row) for _, row in df.iterrows()])

    # Numeric labels: 0 = normal, 1 = anomaly
    if 'anomaly_type' in df.columns:
        y = (df['anomaly_type'] != 'none').astype(int).values
    else:
        y = np.zeros(len(df), dtype=int)  # assume all normal

    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    # Scale features (fit only on training)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save files
    np.save(os.path.join(OUT_DIR, f"{c}_X_train.npy"), X_train_scaled)
    np.save(os.path.join(OUT_DIR, f"{c}_y_train.npy"), y_train)
    np.save(os.path.join(OUT_DIR, f"{c}_X_test.npy"), X_test_scaled)
    np.save(os.path.join(OUT_DIR, f"{c}_y_test.npy"), y_test)
    joblib.dump(scaler, os.path.join(OUT_DIR, f"{c}_scaler.joblib"))

    print(f"Saved {c}: train={X_train_scaled.shape}, test={X_test_scaled.shape}, anomalies in train={y_train.sum()}, test={y_test.sum()}")
