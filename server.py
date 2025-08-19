# import socket
# import threading
# import torch
# import os
# import pickle
# import numpy as np
# from datetime import datetime

# # ------------------ CONFIG ------------------
# HOST = "127.0.0.1"  # Server address
# PORT = 8080         # Server port
# MODEL_DIR = "models" # Folder to store per-client models
# os.makedirs(MODEL_DIR, exist_ok=True)

# # ------------------ HELPERS ------------------
# def log(msg):
#     """Timestamped print for easier debugging"""
#     print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# def save_client_model(client_id, model_state, threshold):
#     """Save model state_dict and threshold for a client"""
#     model_path = os.path.join(MODEL_DIR, f"{client_id}_ae.pt")
#     thresh_path = os.path.join(MODEL_DIR, f"{client_id}_thresh.npy")
#     torch.save(model_state, model_path)
#     np.save(thresh_path, threshold)
#     log(f"✅ Saved model for '{client_id}' -> {model_path}, {thresh_path}")

# def load_client_model(client_id):
#     """Load model state_dict and threshold for a client if exists"""
#     model_path = os.path.join(MODEL_DIR, f"{client_id}_ae.pt")
#     thresh_path = os.path.join(MODEL_DIR, f"{client_id}_thresh.npy")
#     if os.path.exists(model_path) and os.path.exists(thresh_path):
#         model_state = torch.load(model_path, map_location="cpu")
#         threshold = np.load(thresh_path)
#         log(f"📂 Loaded existing model for '{client_id}'")
#         return model_state, threshold
#     else:
#         log(f"⚠ No existing model for '{client_id}', sending None")
#         return None, None

# # ------------------ CLIENT HANDLER ------------------
# def handle_client(conn, addr):
#     try:
#         log(f"🔌 New connection from {addr}")

#         # Receive full message (up to 1024 bytes)
#         data = conn.recv(1024)
#         if not data:
#             log(f"⚠ No data received from {addr}")
#             return

#         message = data.decode('utf-8').strip()
#         log(f"📛 Client message: {message}")

#         # Parse client_id and accuracy from CSV string
#         if ',' not in message:
#             log(f"⚠ Invalid message format: {message}")
#             return
#         client_id, accuracy_str = message.split(',', 1)

#         # Load existing model and threshold for that client_id
#         model_state, threshold = load_client_model(client_id)

#         # Reply with a simple message or with model info if needed
#         if model_state is None:
#             log(f"⚠ No existing model for '{client_id}', sending None")
#             response = "None"
#         else:
#             # For simplicity, just send a confirmation string
#             response = f"Model found for {client_id} with threshold {threshold[0]:.6f}"

#         conn.sendall(response.encode('utf-8'))

#         # You can add code here to save received accuracy, update models, etc.
#         log(f"Received accuracy {accuracy_str} from client '{client_id}'")

#     except Exception as e:
#         log(f"❌ Error handling client {addr}: {e}")

#     finally:
#         conn.close()
#         log(f"🔌 Connection closed for {addr}")



# # ------------------ MAIN SERVER ------------------
# def start_server():
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.bind((HOST, PORT))
#         s.listen()
#         log(f"🚀 Server running on {HOST}:{PORT}")
#         while True:
#             conn, addr = s.accept()
#             thread = threading.Thread(target=handle_client, args=(conn, addr))
#             thread.start()

# if __name__ == "__main__":
#     start_server()





# import socket
# import threading
# import torch
# import os
# import pickle
# import sqlite3
# import csv
# import numpy as np
# from datetime import datetime

# # ------------------ CONFIG ------------------
# HOST = "127.0.0.1"   # Server address
# PORT = 8080          # Server port
# MODEL_DIR = "models" # Folder to store per-client models
# DB_NAME = "server_updates.db"
# CSV_LOG = "server_updates.csv"

# os.makedirs(MODEL_DIR, exist_ok=True)

# # ------------------ LOGGING ------------------
# def log(msg: str):
#     """Timestamped print for easier debugging"""
#     print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# # ------------------ DATABASE UTILS ------------------
# def init_db():
#     """Create SQLite DB and table if they don't exist"""
#     conn = sqlite3.connect(DB_NAME)
#     cur = conn.cursor()
#     cur.execute("""
#         CREATE TABLE IF NOT EXISTS updates (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             client_id TEXT NOT NULL,
#             accuracy REAL,
#             ts TEXT NOT NULL,
#             weights BLOB
#         )
#     """)
#     conn.commit()
#     conn.close()
#     # also ensure a CSV log with headers exists
#     if not os.path.exists(CSV_LOG):
#         with open(CSV_LOG, "w", newline="", encoding="utf-8") as f:
#             w = csv.writer(f)
#             w.writerow(["id","client_id","accuracy","ts"])

# def log_update(client_id: str, accuracy: float, weights_bytes: bytes | None = None):
#     """Insert a row into SQLite and append to CSV (id is from SQLite)"""
#     ts = datetime.now().isoformat()
#     conn = sqlite3.connect(DB_NAME)
#     cur = conn.cursor()
#     cur.execute(
#         "INSERT INTO updates (client_id, accuracy, ts, weights) VALUES (?, ?, ?, ?)",
#         (client_id, accuracy, ts, weights_bytes)
#     )
#     conn.commit()
#     row_id = cur.lastrowid
#     conn.close()

#     # CSV append (no weights in CSV)
#     with open(CSV_LOG, "a", newline="", encoding="utf-8") as f:
#         w = csv.writer(f)
#         w.writerow([row_id, client_id, accuracy, ts])

#     log(f"🗄️  Logged update -> id={row_id}, client='{client_id}', acc={accuracy:.4f}")

# # ------------------ MODEL REGISTRY ------------------
# def save_client_model(client_id, model_state, threshold):
#     """Save model state_dict and threshold for a client"""
#     model_path = os.path.join(MODEL_DIR, f"{client_id}_ae.pt")
#     thresh_path = os.path.join(MODEL_DIR, f"{client_id}_thresh.npy")
#     torch.save(model_state, model_path)
#     np.save(thresh_path, threshold)
#     log(f"✅ Saved model for '{client_id}' -> {model_path}, {thresh_path}")

# def load_client_model(client_id):
#     """Load model state_dict and threshold for a client if exists"""
#     model_path = os.path.join(MODEL_DIR, f"{client_id}_ae.pt")
#     thresh_path = os.path.join(MODEL_DIR, f"{client_id}_thresh.npy")
#     if os.path.exists(model_path) and os.path.exists(thresh_path):
#         model_state = torch.load(model_path, map_location="cpu")
#         threshold = np.load(thresh_path)
#         log(f"📂 Loaded existing model for '{client_id}'")
#         return model_state, threshold
#     else:
#         log(f"⚠ No existing model for '{client_id}', sending None")
#         return None, None

# # ------------------ CLIENT HANDLER ------------------
# def handle_client(conn, addr):
#     try:
#         log(f"🔌 New connection from {addr}")

#         # ---- Receive one line (CSV: client_id,accuracy) ----
#         # Using a simple loop to be robust to partial TCP reads
#         chunks = []
#         conn.settimeout(10.0)  # avoid hanging sockets
#         while True:
#             buf = conn.recv(4096)
#             if not buf:
#                 break
#             chunks.append(buf)
#             # if client sent a tiny CSV and closed, we'll exit by break above
#             # if client someday sends '\n', we could stop earlier:
#             if b"\n" in buf:
#                 break

#         if not chunks:
#             log(f"⚠ No data received from {addr}")
#             return

#         raw = b"".join(chunks).decode("utf-8", errors="replace").strip()
#         # If the client ever includes trailing newline, keep first line
#         message = raw.splitlines()[0].strip()
#         log(f"📛 Client message: {message}")

#         if "," not in message:
#             log(f"⚠ Invalid message format (expected 'client,accuracy'): {message}")
#             conn.sendall(b"ERR: invalid format")
#             return

#         client_id, accuracy_str = message.split(",", 1)
#         client_id = client_id.strip()
#         try:
#             accuracy = float(accuracy_str.strip())
#         except ValueError:
#             log(f"⚠ Could not parse accuracy as float: {accuracy_str}")
#             conn.sendall(b"ERR: invalid accuracy")
#             return

#         # ---- Load existing model/threshold (optional info for the reply) ----
#         model_state, threshold = load_client_model(client_id)

#         # ---- Log this update in SQLite + CSV ----
#         log_update(client_id, accuracy)

#         # ---- Send a friendly ACK back to client ----
#         if threshold is None:
#             response = f"ACK: logged {client_id} acc={accuracy:.4f}; no model on server".encode("utf-8")
#         else:
#             response = f"ACK: logged {client_id} acc={accuracy:.4f}; server-thresh={float(threshold[0]):.6f}".encode("utf-8")

#         conn.sendall(response)
#         log(f"✅ Received accuracy {accuracy:.4f} from client '{client_id}'")

#         # NOTE (future-proofing):
#         # In Phase 2 we will also accept a second binary payload for model weights,
#         # then call: log_update(client_id, accuracy, weights_bytes)

#     except Exception as e:
#         log(f"❌ Error handling client {addr}: {e}")

#     finally:
#         try:
#             conn.shutdown(socket.SHUT_RDWR)
#         except Exception:
#             pass
#         conn.close()
#         log(f"🔌 Connection closed for {addr}")

# # ------------------ MAIN SERVER ------------------
# def start_server():
#     init_db()  # ensure DB is ready before listening
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         # Allow quick restarts on Windows/macOS/Linux
#         try:
#             s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#         except Exception:
#             pass

#         s.bind((HOST, PORT))
#         s.listen()
#         log(f"🚀 Server running on {HOST}:{PORT}")
#         while True:
#             conn, addr = s.accept()
#             thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
#             thread.start()

# if __name__ == "__main__":
#     start_server()








# server.py
import socket
import threading
import torch
import os
import pickle
import sqlite3
import csv
import numpy as np
import json
from datetime import datetime

# ------------------ CONFIG ------------------
HOST = "127.0.0.1"
PORT = 8080
MODEL_DIR = "models"
DB_NAME = "server_updates.db"
CSV_LOG = "server_updates.csv"
GLOBAL_MODEL_PATH = os.path.join(MODEL_DIR, "global_ae.pt")
GLOBAL_THRESH_PATH = os.path.join(MODEL_DIR, "global_thresh.npy")
GLOBAL_META_PATH = os.path.join(MODEL_DIR, "global_meta.json")
MAX_PAYLOAD_SIZE = 100 * 1024 * 1024  # 100 MB

os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------ LOGGING ------------------
def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# ------------------ DB & CSV ------------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS updates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id TEXT NOT NULL,
            accuracy REAL,
            ts TEXT NOT NULL,
            weights BLOB
        )
    """)
    conn.commit()
    conn.close()
    if not os.path.exists(CSV_LOG):
        with open(CSV_LOG, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id","client_id","accuracy","ts"])

def log_update(client_id: str, accuracy: float, weights_bytes: bytes | None = None):
    ts = datetime.now().isoformat()
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO updates (client_id, accuracy, ts, weights) VALUES (?, ?, ?, ?)",
        (client_id, accuracy, ts, weights_bytes)
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    with open(CSV_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([row_id, client_id, accuracy, ts])
    log(f"🗄️  Logged update -> id={row_id}, client='{client_id}', acc={accuracy:.4f}")

# ------------------ MODEL FILE HELPERS ------------------
def save_client_model(client_id, model_state, threshold):
    model_path = os.path.join(MODEL_DIR, f"{client_id}_ae.pt")
    thresh_path = os.path.join(MODEL_DIR, f"{client_id}_thresh.npy")
    torch.save(model_state, model_path)
    np.save(thresh_path, np.array([float(threshold) if threshold is not None else 0.0]))
    log(f"✅ Saved model for '{client_id}' -> {model_path}, {thresh_path}")

def load_client_model(client_id):
    model_path = os.path.join(MODEL_DIR, f"{client_id}_ae.pt")
    thresh_path = os.path.join(MODEL_DIR, f"{client_id}_thresh.npy")
    if os.path.exists(model_path) and os.path.exists(thresh_path):
        model_state = torch.load(model_path, map_location="cpu")
        threshold = np.load(thresh_path)
        return model_state, threshold
    return None, None

# Global lock for aggregation
agg_lock = threading.Lock()

# ------------------ UTILS ------------------
def recvall(conn, n):
    data = b""
    while len(data) < n:
        packet = conn.recv(min(65536, n - len(data)))
        if not packet:
            return None
        data += packet
    return data

def load_global_meta():
    if os.path.exists(GLOBAL_META_PATH):
        with open(GLOBAL_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"n_total": 0}

def save_global_meta(meta):
    with open(GLOBAL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f)

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    try:
        return torch.tensor(x)
    except Exception:
        return None

# ------------------ AGGREGATION ------------------
def aggregate_with_client(client_state: dict, n_client: int, client_threshold: float | None):
    """
    Weighted FedAvg aggregation (element-wise):
    new_global = (global*n_total + client*n_client) / (n_total + n_client)
    """
    with agg_lock:
        # Load existing global state and n_total
        if not os.path.exists(GLOBAL_MODEL_PATH):
            # No global yet -> set global = client
            torch.save(client_state, GLOBAL_MODEL_PATH)
            if client_threshold is None:
                client_threshold = 0.0
            np.save(GLOBAL_THRESH_PATH, np.array([float(client_threshold)]))
            meta = {"n_total": int(n_client)}
            save_global_meta(meta)
            log(f"🔁 Initialized global model from client (n={n_client})")
            return client_state, meta["n_total"], float(client_threshold)

        # Load existing global
        global_state = torch.load(GLOBAL_MODEL_PATH, map_location="cpu")
        meta = load_global_meta()
        n_global = int(meta.get("n_total", 0)) or 0
        if n_client <= 0:
            n_client = 1

        # Weighted average
        new_state = {}
        try:
            for k in global_state.keys():
                g = to_tensor(global_state[k]).float()
                c_raw = client_state.get(k)
                if c_raw is None:
                    # if client lacks a key, keep global
                    new_state[k] = g
                    continue
                c = to_tensor(c_raw).float()
                # compute weighted average
                if n_global <= 0:
                    merged = c
                else:
                    merged = (g * n_global + c * n_client) / (n_global + n_client)
                new_state[k] = merged.clone()
        except Exception as e:
            log(f"❌ Aggregation error: {e}. Falling back to client state as global.")
            new_state = {k: to_tensor(v).float() for k, v in client_state.items()}

        # new threshold aggregation (weighted)
        if os.path.exists(GLOBAL_THRESH_PATH):
            old_thresh = float(np.load(GLOBAL_THRESH_PATH)[0])
        else:
            old_thresh = 0.0
        client_thresh = float(client_threshold) if client_threshold is not None else 0.0
        new_n = n_global + n_client
        if new_n > 0:
            new_thresh = (old_thresh * n_global + client_thresh * n_client) / new_n
        else:
            new_thresh = client_thresh

        # Save new global
        torch.save(new_state, GLOBAL_MODEL_PATH)
        np.save(GLOBAL_THRESH_PATH, np.array([float(new_thresh)]))
        meta["n_total"] = new_n
        save_global_meta(meta)

        log(f"🔁 Aggregated global model: prev_n={n_global}, added={n_client}, new_n={new_n}")
        return new_state, new_n, float(new_thresh)

# ------------------ CLIENT HANDLER ------------------
def handle_client(conn, addr):
    try:
        log(f"🔌 Connection from {addr}")
        conn.settimeout(20.0)

        # Read 8-byte length header (blocking until exactly 8 bytes)
        initial = recvall(conn, 8)
        if initial is None:
            # fallback: try small text read once
            try:
                raw = conn.recv(4096).decode('utf-8', errors='replace').strip()
                message = raw.splitlines()[0].strip() if raw else ""
            except Exception:
                message = ""
            if not message:
                log(f"⚠ No data from {addr}")
                conn.close()
                return
            # text path
            log(f"📛 Text message: {message}")
            if "," not in message:
                conn.sendall(b"ERR: invalid format")
                return
            client_id, accuracy_str = message.split(",", 1)
            try:
                accuracy = float(accuracy_str.strip())
            except:
                conn.sendall(b"ERR: invalid accuracy")
                return
            log_update(client_id, accuracy, None)
            # reply legacy ack
            _, threshold = load_client_model(client_id)
            if threshold is None:
                conn.sendall(f"ACK: logged {client_id} acc={accuracy:.4f}; no model".encode('utf-8'))
            else:
                conn.sendall(f"ACK: logged {client_id} acc={accuracy:.4f}; server-thresh={float(threshold[0]):.6f}".encode('utf-8'))
            return

        # parse header size
        msg_size = int.from_bytes(initial, 'big')
        if msg_size <= 0 or msg_size > MAX_PAYLOAD_SIZE:
            log(f"⚠ Invalid payload size {msg_size} from {addr}, rejecting")
            conn.sendall(b"ERR: invalid size")
            return

        # read payload
        payload = recvall(conn, msg_size)
        if payload is None or len(payload) != msg_size:
            log(f"⚠ Incomplete payload from {addr}")
            conn.sendall(b"ERR: incomplete payload")
            return

        # try to unpickle
        try:
            obj = pickle.loads(payload)
        except Exception as e:
            log(f"⚠ Unpickle failed: {e}")
            conn.sendall(b"ERR: invalid payload")
            return

        # Expect dict with keys
        if not isinstance(obj, dict) or 'client_id' not in obj:
            log("⚠ Payload missing 'client_id' → invalid")
            conn.sendall(b"ERR: invalid structure")
            return

        client_id = str(obj.get('client_id'))
        accuracy = float(obj.get('accuracy', 0.0))
        client_state = obj.get('state_dict', None)
        threshold = obj.get('threshold', None)
        n_samples = int(obj.get('n_samples', 1) or 1)

        # Log raw update in DB
        log_update(client_id, accuracy, weights_bytes=payload)

        # Save client model file
        if client_state is not None:
            try:
                save_client_model(client_id, client_state, threshold)
            except Exception as e:
                log(f"⚠ Could not save client model: {e}")

        # Aggregate into global model (asynchronous FedAvg)
        global_state, new_n_total, new_thresh = aggregate_with_client(client_state or {}, n_samples, threshold)

        # Prepare response payload (global model)
        resp_obj = {
            "global_state_dict": global_state,
            "n_total": new_n_total,
            "global_threshold": new_thresh
        }
        resp_bytes = pickle.dumps(resp_obj)
        resp_size = len(resp_bytes)
        # send length-prefixed response
        conn.sendall(resp_size.to_bytes(8, 'big') + resp_bytes)
        log(f"⬆ Sent aggregated global model back to {client_id} (size {resp_size} bytes)")
        

    except Exception as e:
        log(f"❌ Error handling client {addr}: {e}")

    finally:
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        conn.close()
        log(f"🔌 Closed {addr}")

# ------------------ MAIN SERVER ------------------
def start_server():
    init_db()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(8)
        log(f"🚀 Server running on {HOST}:{PORT}")
        while True:
            conn, addr = s.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            t.start()

if __name__ == "__main__":
    start_server()
