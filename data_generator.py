# data_generator.py
import argparse, os, random
from faker import Faker
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from sqlalchemy import create_engine
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------- CONFIG ----------
SEED = 42
CORE_AAD_COUNT = 20000
CLIENTS = ['aadhaar_office','bank','insurance','pension','telecom','death_registrar','tax']
STATES = ['Delhi','Maharashtra','Karnataka','Tamil Nadu','Gujarat','West Bengal','Uttar Pradesh','Rajasthan','Kerala','Punjab']

fake = Faker()
Faker.seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ---------- HELPERS ----------
def make_aadhaar_pool(n):
    return [f"SYN_AAD_{i:06d}" for i in range(n)]

def sample_person(aadhaar_id):
    return {
        'aadhaar_id': aadhaar_id,
        'name': fake.name(),
        'dob': fake.date_of_birth(minimum_age=18, maximum_age=95),
        'gender': random.choice(['M','F','O']),
        'addr_state': random.choice(STATES),
        'addr_district': fake.city(),
        'phone': fake.msisdn()[:10],
        'biom_hash': fake.sha1()
    }

def sample_bank_row(aid=None):
    aid = aid or random.choice(aad_pool)
    deposit = round(max(0, np.random.exponential(scale=20000)),2)
    income = int(np.random.choice([50000,100000,250000,500000,1000000], p=[0.4,0.3,0.15,0.1,0.05]))
    return {
        'aadhaar_id': aid,
        'account_id': f"ACC{random.randint(10**6,10**8)}",
        'name': fake.name(),
        'dob': fake.date_of_birth(minimum_age=18, maximum_age=90),
        'addr': fake.address().replace("\n"," "),
        'phone': fake.msisdn()[:10],
        'income_proof_flag': random.random() < 0.7,
        'account_open_time': fake.date_time_between(start_date='-5y', end_date='now'),
        'initial_deposit': deposit,
        'last_txn_date': fake.date_time_between(start_date='-365d', end_date='now'),
        'anomaly_type': 'none'
    }

def sample_insurance_row(aid=None):
    aid = aid or random.choice(aad_pool)
    start = fake.date_between(start_date='-3y', end_date='today')
    claim = fake.date_between(start_date=start, end_date='today')
    return {
        'aadhaar_id': aid,
        'policy_id': f"POL{random.randint(100000,999999)}",
        'policy_start': start,
        'claim_id': f"CLM{random.randint(1000000,9999999)}",
        'claim_date': claim,
        'hospital_id': f"HOSP_{random.randint(1,1000)}",
        'claim_amount': round(max(0, np.random.normal(20000,5000)),2),
        'diagnosis_code': random.choice(['A01','B02','C03','D04']),
        'claim_description': random.choice(['fever','fracture','surgery','covid','maternity']),
        'anomaly_type': 'none'
    }

def sample_pension_row(aid=None):
    aid = aid or random.choice(aad_pool)
    enroll = fake.date_between(start_date='-10y', end_date='today')
    disb = float(random.choice([2000,3000,5000]))
    return {
        'aadhaar_id': aid,
        'benefit_type': random.choice(['old_age','disability','widow']),
        'enroll_date': enroll,
        'disbursement_amt': disb,
        'bank_account': f"BNK{random.randint(10**6,10**8)}",
        'last_disb_date': enroll + timedelta(days=365),
        'anomaly_type': 'none'
    }

def sample_telecom_row(aid=None):
    aid = aid or random.choice(aad_pool)
    return {
        'aadhaar_id': aid,
        'sim_id': f"SIM{random.randint(100000,999999)}",
        'sim_reg_date': fake.date_between(start_date='-3y', end_date='today'),
        'mobile_no': fake.msisdn()[:10],
        'imei': fake.md5()[:15],
        'location_cell': f"cell_{random.randint(1,500)}",
        'sim_vendor': random.choice(['VendorA','VendorB','VendorC']),
        'anomaly_type': 'none'
    }

def sample_death_row(aid=None):
    aid = aid or random.choice(aad_pool)
    return {
        'aadhaar_id': aid,
        'name': fake.name(),
        'death_date': fake.date_between(start_date='-2y', end_date='today'),
        'death_place': random.choice(['Hospital','Home','Road']),
        'cause_code': random.choice(['natural','accident','unknown']),
        'reporting_officer': fake.name(),
        'anomaly_type': 'none'
    }

def sample_tax_row(aid=None):
    aid = aid or random.choice(aad_pool)
    income = int(np.random.choice([300000,600000,1000000,2000000], p=[0.5,0.3,0.15,0.05]))
    return {
        'aadhaar_id': aid,
        'declared_income': income,
        'employer_id': f"EMP_{random.randint(1,1000):04d}",
        'filing_date': fake.date_between(start_date='-3y', end_date='today'),
        'tax_paid': round(income * 0.05, 2),
        'anomaly_type': 'none'
    }

def inject_cross_client_anomalies(records, aad_pool, dup_pct=0.03, death_after_pct=0.02, inflated_claim_pct=0.01):
    num_dup = max(1, int(dup_pct * len(aad_pool)))
    dup_ids = random.sample(aad_pool, num_dup)
    for aid in dup_ids:
        c1, c2 = random.sample(CLIENTS, 2)
        if records[c1]:
            rec = random.choice(records[c1])
            rec['aadhaar_id'] = aid
        if records[c2]:
            rec2 = random.choice(records[c2])
            rec2['aadhaar_id'] = aid
            rec2['name'] = fake.name()
            rec2['anomaly_type'] = 'duplicate_aadhaar_diff_name'

    num_death_after = max(1, int(death_after_pct * len(aad_pool)))
    death_ids = random.sample(aad_pool, num_death_after)
    for aid in death_ids:
        drec = sample_death_row(aid)
        drec['death_date'] = fake.date_between(start_date='-2y', end_date='-30d')
        drec['anomaly_type'] = 'death_after_activity'
        records['death_registrar'].append(drec)
        bankrec = sample_bank_row(aid)
        bankrec['last_txn_date'] = pd.Timestamp(drec['death_date']) + pd.Timedelta(days=random.randint(1,30))
        bankrec['anomaly_type'] = 'death_after_activity'
        records['bank'].append(bankrec)

    num_infl = max(1, int(inflated_claim_pct * len(aad_pool)))
    infl_ids = random.sample(aad_pool, num_infl)
    for aid in infl_ids:
        if records['insurance']:
            r = random.choice(records['insurance'])
            r['claim_amount'] = r.get('claim_amount',1000) * random.uniform(5,12)
            r['anomaly_type'] = 'inflated_claim'

def generate_and_insert(db_url, out_dir, rows_per_client=2000, seed=42, dup_pct=0.03, death_after_pct=0.02, inflated_claim_pct=0.01):
    Faker.seed(seed); random.seed(seed); np.random.seed(seed)
    global aad_pool
    aad_pool = make_aadhaar_pool(CORE_AAD_COUNT)

    engine = create_engine(db_url, pool_pre_ping=True)

    records = {c: [] for c in CLIENTS}

    for c in CLIENTS:
        for _ in range(rows_per_client):
            if c == 'aadhaar_office':
                aid = random.choice(aad_pool)
                rec = sample_person(aid)
                rec['enroll_time'] = fake.date_time_between(start_date='-5y', end_date='now')
                rec['enroll_center_id'] = f"CTR_{random.randint(1,200)}"
                rec['anomaly_type'] = 'none'
                records[c].append(rec)
            elif c == 'bank':
                records[c].append(sample_bank_row())
            elif c == 'insurance':
                records[c].append(sample_insurance_row())
            elif c == 'pension':
                records[c].append(sample_pension_row())
            elif c == 'telecom':
                records[c].append(sample_telecom_row())
            elif c == 'death_registrar':
                records[c].append(sample_death_row())
            elif c == 'tax':
                records[c].append(sample_tax_row())

    inject_cross_client_anomalies(records, aad_pool, dup_pct, death_after_pct, inflated_claim_pct)

    os.makedirs(out_dir, exist_ok=True)
    for c, rows in records.items():
        df = pd.DataFrame(rows)
        csv_path = os.path.join(out_dir, f"{c}.csv")
        df.to_csv(csv_path, index=False)
        print("Wrote", csv_path, "rows:", len(df))
        try:
            df.to_sql(c, engine, if_exists='append', index=False, method='multi', chunksize=1000)
            print(f"Inserted {len(df)} rows into table '{c}'")
        except Exception as e:
            print("DB insert failed for", c, e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='sample_data')
    parser.add_argument('--rows_per_client', type=int, default=2000)
    parser.add_argument('--db_url', default='postgresql://deepasri:2004@localhost:5432/fedid_misuse')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--dup_pct', type=float, default=0.03)
    parser.add_argument('--death_after_pct', type=float, default=0.02)
    parser.add_argument('--inflated_claim_pct', type=float, default=0.01)
    args = parser.parse_args()

    generate_and_insert(args.db_url, args.out_dir, args.rows_per_client, args.seed, args.dup_pct, args.death_after_pct, args.inflated_claim_pct)
