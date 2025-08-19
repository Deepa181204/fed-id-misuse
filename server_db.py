# server_db.py
import sqlite3
import datetime

DB_NAME = "server_updates.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS updates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        client_id TEXT,
        accuracy REAL,
        timestamp TEXT,
        weights BLOB
    )
    """)
    conn.commit()
    conn.close()

def log_update(client_id, accuracy, weights=None):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO updates (client_id, accuracy, timestamp, weights) VALUES (?, ?, ?, ?)",
        (client_id, accuracy, datetime.datetime.now().isoformat(), weights)
    )
    conn.commit()
    conn.close()

def fetch_all_updates():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT * FROM updates")
    rows = cur.fetchall()
    conn.close()
    return rows
