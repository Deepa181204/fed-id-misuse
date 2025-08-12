import socket
import threading
import torch
import os
import pickle
import numpy as np
from datetime import datetime

# ------------------ CONFIG ------------------
HOST = "127.0.0.1"  # Server address
PORT = 8080         # Server port
MODEL_DIR = "models" # Folder to store per-client models
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------ HELPERS ------------------
def log(msg):
    """Timestamped print for easier debugging"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def save_client_model(client_id, model_state, threshold):
    """Save model state_dict and threshold for a client"""
    model_path = os.path.join(MODEL_DIR, f"{client_id}_ae.pt")
    thresh_path = os.path.join(MODEL_DIR, f"{client_id}_thresh.npy")
    torch.save(model_state, model_path)
    np.save(thresh_path, threshold)
    log(f"✅ Saved model for '{client_id}' -> {model_path}, {thresh_path}")

def load_client_model(client_id):
    """Load model state_dict and threshold for a client if exists"""
    model_path = os.path.join(MODEL_DIR, f"{client_id}_ae.pt")
    thresh_path = os.path.join(MODEL_DIR, f"{client_id}_thresh.npy")
    if os.path.exists(model_path) and os.path.exists(thresh_path):
        model_state = torch.load(model_path, map_location="cpu")
        threshold = np.load(thresh_path)
        log(f"📂 Loaded existing model for '{client_id}'")
        return model_state, threshold
    else:
        log(f"⚠ No existing model for '{client_id}', sending None")
        return None, None

# ------------------ CLIENT HANDLER ------------------
def handle_client(conn, addr):
    try:
        log(f"🔌 New connection from {addr}")

        # Receive full message (up to 1024 bytes)
        data = conn.recv(1024)
        if not data:
            log(f"⚠ No data received from {addr}")
            return

        message = data.decode('utf-8').strip()
        log(f"📛 Client message: {message}")

        # Parse client_id and accuracy from CSV string
        if ',' not in message:
            log(f"⚠ Invalid message format: {message}")
            return
        client_id, accuracy_str = message.split(',', 1)

        # Load existing model and threshold for that client_id
        model_state, threshold = load_client_model(client_id)

        # Reply with a simple message or with model info if needed
        if model_state is None:
            log(f"⚠ No existing model for '{client_id}', sending None")
            response = "None"
        else:
            # For simplicity, just send a confirmation string
            response = f"Model found for {client_id} with threshold {threshold[0]:.6f}"

        conn.sendall(response.encode('utf-8'))

        # You can add code here to save received accuracy, update models, etc.
        log(f"Received accuracy {accuracy_str} from client '{client_id}'")

    except Exception as e:
        log(f"❌ Error handling client {addr}: {e}")

    finally:
        conn.close()
        log(f"🔌 Connection closed for {addr}")



# ------------------ MAIN SERVER ------------------
def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        log(f"🚀 Server running on {HOST}:{PORT}")
        while True:
            conn, addr = s.accept()
            thread = threading.Thread(target=handle_client, args=(conn, addr))
            thread.start()

if __name__ == "__main__":
    start_server()
