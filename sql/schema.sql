-- sql/schema.sql
-- WARNING: uses synthetic aadhaar ids only (SYN_AAD_...)

CREATE TABLE aadhaar_office (
  id SERIAL PRIMARY KEY,
  aadhaar_id VARCHAR(32) NOT NULL,
  name TEXT,
  dob DATE,
  gender CHAR(1),
  addr_state TEXT,
  addr_district TEXT,
  biom_hash TEXT,
  enroll_time TIMESTAMP,
  enroll_center_id TEXT,
  phone VARCHAR(20),
  anomaly_type TEXT,
  created_at TIMESTAMP DEFAULT now()
);

CREATE INDEX idx_aadhaar_office_aadhaar ON aadhaar_office(aadhaar_id);

CREATE TABLE bank (
  id SERIAL PRIMARY KEY,
  aadhaar_id VARCHAR(32) NOT NULL,
  account_id TEXT,
  name TEXT,
  dob DATE,
  addr TEXT,
  phone VARCHAR(20),
  income_proof_flag BOOLEAN,
  account_open_time TIMESTAMP,
  initial_deposit NUMERIC,
  last_txn_date TIMESTAMP,
  anomaly_type TEXT,
  created_at TIMESTAMP DEFAULT now()
);
CREATE INDEX idx_bank_aadhaar ON bank(aadhaar_id);

CREATE TABLE insurance (
  id SERIAL PRIMARY KEY,
  aadhaar_id VARCHAR(32) NOT NULL,
  policy_id TEXT,
  policy_start DATE,
  claim_id TEXT,
  claim_date DATE,
  hospital_id TEXT,
  claim_amount NUMERIC,
  diagnosis_code TEXT,
  claim_description TEXT,
  anomaly_type TEXT,
  created_at TIMESTAMP DEFAULT now()
);
CREATE INDEX idx_insurance_aadhaar ON insurance(aadhaar_id);

CREATE TABLE pension (
  id SERIAL PRIMARY KEY,
  aadhaar_id VARCHAR(32) NOT NULL,
  benefit_type TEXT,
  enroll_date DATE,
  disbursement_amt NUMERIC,
  bank_account TEXT,
  last_disb_date DATE,
  anomaly_type TEXT,
  created_at TIMESTAMP DEFAULT now()
);
CREATE INDEX idx_pension_aadhaar ON pension(aadhaar_id);

CREATE TABLE telecom (
  id SERIAL PRIMARY KEY,
  aadhaar_id VARCHAR(32) NOT NULL,
  sim_id TEXT,
  sim_reg_date DATE,
  mobile_no VARCHAR(20),
  imei VARCHAR(32),
  location_cell TEXT,
  sim_vendor TEXT,
  anomaly_type TEXT,
  created_at TIMESTAMP DEFAULT now()
);
CREATE INDEX idx_telecom_aadhaar ON telecom(aadhaar_id);
CREATE INDEX idx_telecom_imei ON telecom(imei);

CREATE TABLE death_registrar (
  id SERIAL PRIMARY KEY,
  aadhaar_id VARCHAR(32) NOT NULL,
  name TEXT,
  death_date DATE,
  death_place TEXT,
  cause_code TEXT,
  reporting_officer TEXT,
  anomaly_type TEXT,
  created_at TIMESTAMP DEFAULT now()
);
CREATE INDEX idx_death_aadhaar ON death_registrar(aadhaar_id);

CREATE TABLE tax_records (
  id SERIAL PRIMARY KEY,
  aadhaar_id VARCHAR(32) NOT NULL,
  declared_income NUMERIC,
  employer_id TEXT,
  filing_date DATE,
  tax_paid NUMERIC,
  anomaly_type TEXT,
  created_at TIMESTAMP DEFAULT now()
);
CREATE INDEX idx_tax_aadhaar ON tax_records(aadhaar_id);

-- Optional property table
CREATE TABLE property (
  id SERIAL PRIMARY KEY,
  property_id TEXT,
  aadhaar_id VARCHAR(32),          -- owner
  txn_date DATE,
  txn_amount NUMERIC,
  seller_aadhaar VARCHAR(32),
  anomaly_type TEXT,
  created_at TIMESTAMP DEFAULT now()
);
CREATE INDEX idx_property_aadhaar ON property(aadhaar_id);
