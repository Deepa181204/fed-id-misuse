# # # api_server.py
# # from flask import Flask, jsonify, request, send_file
# # import sqlite3, os, json, numpy as np, torch
# # from datetime import datetime
# # from pathlib import Path

# # BASE = Path(__file__).parent.resolve()
# # DB_PATH = BASE / "server_updates.db"
# # MODEL_DIR = BASE / "models"
# # FEATURE_DIR = BASE / "features"

# # app = Flask(__name__)

# # def db_query(q, params=()):
# #     if not DB_PATH.exists():
# #         return []
# #     conn = sqlite3.connect(str(DB_PATH))
# #     conn.row_factory = sqlite3.Row
# #     cur = conn.cursor()
# #     cur.execute(q, params)
# #     rows = [dict(r) for r in cur.fetchall()]
# #     conn.close()
# #     return rows

# # @app.route("/clients", methods=["GET"])
# # def clients():
# #     # clients known by models folder or features folder
# #     clients = set()
# #     if MODEL_DIR.exists():
# #         for f in MODEL_DIR.glob("*_ae.pt"):
# #             clients.add(f.name.rsplit("_ae.pt", 1)[0])
# #     if FEATURE_DIR.exists():
# #         for f in FEATURE_DIR.glob("*_X_test.npy"):
# #             clients.add(f.name.rsplit("_X_test.npy", 1)[0])
# #     return jsonify(sorted(list(clients)))

# # @app.route("/updates", methods=["GET"])
# # def updates():
# #     # optional query arg ?limit=50
# #     limit = int(request.args.get("limit", 50))
# #     rows = db_query("SELECT id, client_id, accuracy, ts FROM updates ORDER BY id DESC LIMIT ?", (limit,))
# #     return jsonify(rows)

# # @app.route("/anomalies", methods=["GET"])
# # def anomalies():
# #     """
# #     Query params:
# #       client_id (required)
# #       top (optional, default 50) - number of most anomalous rows to return
# #       return_features (0/1 default 0) - whether to include feature vectors in response
# #     Response:
# #       { client_id, threshold, results: [ { idx, mse, is_anomaly (bool), features (optional) } ] }
# #     """
# #     client_id = request.args.get("client_id")
# #     if not client_id:
# #         return jsonify({"error":"client_id required"}), 400
# #     top_k = int(request.args.get("top", 50))
# #     return_features = request.args.get("return_features", "0") in ("1","true","True")

# #     x_test_path = FEATURE_DIR / f"{client_id}_X_test.npy"
# #     model_path = MODEL_DIR / f"{client_id}_ae.pt"
# #     thresh_path = MODEL_DIR / f"{client_id}_thresh.npy"

# #     if not x_test_path.exists():
# #         return jsonify({"error": f"Missing features for {client_id}"}), 404
# #     if not model_path.exists() or not thresh_path.exists():
# #         return jsonify({"error": f"Missing model/threshold for {client_id}"}), 404

# #     X_test = np.load(str(x_test_path))
# #     state = torch.load(str(model_path), map_location="cpu")
# #     # build AE skeleton dynamically using dimension from X_test
# #     input_dim = X_test.shape[1]
# #     # define small AE same as training
# #     import torch.nn as nn
# #     class AE(nn.Module):
# #         def __init__(self, dim):
# #             super().__init__()
# #             self.encoder = nn.Sequential(nn.Linear(dim,32), nn.ReLU(), nn.Linear(32,8), nn.ReLU(), nn.Linear(8,4))
# #             self.decoder = nn.Sequential(nn.Linear(4,8), nn.ReLU(), nn.Linear(8,32), nn.ReLU(), nn.Linear(32,dim))
# #         def forward(self,x): return self.decoder(self.encoder(x))

# #     model = AE(input_dim)
# #     model.load_state_dict(state)
# #     model.eval()

# #     with torch.no_grad():
# #         X_t = torch.tensor(X_test, dtype=torch.float32)
# #         recon = model(X_t).numpy()
# #     mses = np.mean((recon - X_test)**2, axis=1)
# #     threshold = float(np.load(str(thresh_path))[0])

# #     # build sorted list by mse, largest first
# #     indices = np.argsort(-mses)[:top_k]
# #     results = []
# #     for idx in indices:
# #         m = float(mses[idx])
# #         results.append({
# #             "idx": int(idx),
# #             "mse": m,
# #             "is_anomaly": bool(m > threshold),
# #             "features": (X_test[idx].tolist() if return_features else None)
# #         })

# #     return jsonify({
# #         "client_id": client_id,
# #         "threshold": threshold,
# #         "n_test": int(X_test.shape[0]),
# #         "results": results
# #     })

# # @app.route("/global", methods=["GET"])
# # def global_meta():
# #     global_model = MODEL_DIR / "global_ae.pt"
# #     global_thresh = MODEL_DIR / "global_thresh.npy"
# #     meta = {}
# #     if global_model.exists():
# #         try:
# #             state = torch.load(str(global_model), map_location="cpu")
# #             meta["has_global"] = True
# #             meta["n_params"] = sum(p.numel() for p in state.values()) if isinstance(state, dict) else 0
# #         except Exception:
# #             meta["has_global"] = True
# #     else:
# #         meta["has_global"] = False
# #     if global_thresh.exists():
# #         meta["global_thresh"] = float(np.load(str(global_thresh))[0])
# #     return jsonify(meta)

# # if __name__ == "__main__":
# #     app.run(host="0.0.0.0", port=5000, debug=False)




# # api_server.py
# from flask import Flask, jsonify, request
# from flask_cors import CORS
# import sqlite3, os, numpy as np, torch
# from pathlib import Path

# BASE = Path(__file__).parent.resolve()
# DB_PATH = BASE / "server_updates.db"
# MODEL_DIR = BASE / "models"
# FEATURE_DIR = BASE / "features"

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})

# def db_query(q, params=()):
#     if not DB_PATH.exists():
#         return []
#     conn = sqlite3.connect(str(DB_PATH))
#     conn.row_factory = sqlite3.Row
#     cur = conn.cursor()
#     cur.execute(q, params)
#     rows = [dict(r) for r in cur.fetchall()]
#     conn.close()
#     return rows

# @app.route("/clients", methods=["GET"])
# def clients():
#     clients = set()
#     if MODEL_DIR.exists():
#         for f in MODEL_DIR.glob("*_ae.pt"):
#             clients.add(f.name.rsplit("_ae.pt", 1)[0])
#     if FEATURE_DIR.exists():
#         for f in FEATURE_DIR.glob("*_X_test.npy"):
#             clients.add(f.name.rsplit("_X_test.npy", 1)[0])
#     # if DB exists, also add clients from DB for safety
#     rows = db_query("SELECT DISTINCT client_id FROM updates")
#     for r in rows:
#         clients.add(r.get("client_id"))
#     return jsonify(sorted([c for c in clients if c]))

# @app.route("/updates", methods=["GET"])
# def updates():
#     limit = int(request.args.get("limit", 50))
#     rows = db_query("SELECT id, client_id, accuracy, ts FROM updates ORDER BY id DESC LIMIT ?", (limit,))
#     return jsonify(rows)

# @app.route("/anomalies", methods=["GET"])
# def anomalies():
#     client_id = request.args.get("client_id")
#     if not client_id:
#         return jsonify({"error":"client_id required"}), 400
#     top_k = int(request.args.get("top", 50))
#     return_features = request.args.get("return_features", "0") in ("1","true","True")

#     x_test_path = FEATURE_DIR / f"{client_id}_X_test.npy"
#     model_path = MODEL_DIR / f"{client_id}_ae.pt"
#     thresh_path = MODEL_DIR / f"{client_id}_thresh.npy"

#     if not x_test_path.exists():
#         return jsonify({"error": f"Missing features for {client_id}"}), 404
#     if not model_path.exists() or not thresh_path.exists():
#         return jsonify({"error": f"Missing model/threshold for {client_id}"}), 404

#     X_test = np.load(str(x_test_path))
#     state = torch.load(str(model_path), map_location="cpu")
#     input_dim = X_test.shape[1]

#     import torch.nn as nn
#     class AE(nn.Module):
#         def __init__(self, dim):
#             super().__init__()
#             self.encoder = nn.Sequential(nn.Linear(dim,32), nn.ReLU(), nn.Linear(32,8), nn.ReLU(), nn.Linear(8,4))
#             self.decoder = nn.Sequential(nn.Linear(4,8), nn.ReLU(), nn.Linear(8,32), nn.ReLU(), nn.Linear(32,dim))
#         def forward(self,x): return self.decoder(self.encoder(x))

#     model = AE(input_dim)
#     model.load_state_dict(state)
#     model.eval()

#     with torch.no_grad():
#         X_t = torch.tensor(X_test, dtype=torch.float32)
#         recon = model(X_t).numpy()
#     mses = np.mean((recon - X_test)**2, axis=1)
#     threshold = float(np.load(str(thresh_path))[0])

#     indices = np.argsort(-mses)[:top_k]
#     results = []
#     for idx in indices:
#         m = float(mses[idx])
#         results.append({
#             "idx": int(idx),
#             "mse": m,
#             "is_anomaly": bool(m > threshold),
#             "features": (X_test[idx].tolist() if return_features else None)
#         })

#     return jsonify({
#         "client_id": client_id,
#         "threshold": threshold,
#         "n_test": int(X_test.shape[0]),
#         "results": results
#     })

# @app.route("/global", methods=["GET"])
# def global_meta():
#     global_model = MODEL_DIR / "global_ae.pt"
#     global_thresh = MODEL_DIR / "global_thresh.npy"
#     meta = {}
#     meta["has_global"] = global_model.exists()
#     if global_model.exists():
#         try:
#             state = torch.load(str(global_model), map_location="cpu")
#             meta["n_params"] = sum(v.numel() for v in state.values()) if isinstance(state, dict) else 0
#         except Exception:
#             meta["n_params"] = 0
#     if global_thresh.exists():
#         meta["global_thresh"] = float(np.load(str(global_thresh))[0])
#     return jsonify(meta)

# if __name__ == "__main__":
#     # runs on port 5000 by default
#     app.run(host="0.0.0.0", port=5000, debug=False)







# api_server.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import sqlite3, os, numpy as np, torch
from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent.resolve()
DB_PATH = BASE / "server_updates.db"
MODEL_DIR = BASE / "models"
FEATURE_DIR = BASE / "features"

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def db_query(q, params=()):
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(q, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

@app.route("/clients", methods=["GET"])
def clients():
    clients = set()
    if MODEL_DIR.exists():
        for f in MODEL_DIR.glob("*_ae.pt"):
            clients.add(f.name.rsplit("_ae.pt", 1)[0])
    if FEATURE_DIR.exists():
        for f in FEATURE_DIR.glob("*_X_test.npy"):
            clients.add(f.name.rsplit("_X_test.npy", 1)[0])
    rows = db_query("SELECT DISTINCT client_id FROM updates")
    for r in rows:
        clients.add(r.get("client_id"))
    return jsonify(sorted([c for c in clients if c]))

@app.route("/updates", methods=["GET"])
def updates():
    limit = int(request.args.get("limit", 50))
    rows = db_query("SELECT id, client_id, accuracy, ts FROM updates ORDER BY id DESC LIMIT ?", (limit,))
    return jsonify(rows)

@app.route("/anomalies", methods=["GET"])
def anomalies():
    """
    Returns top-k anomalous rows with the original CSV record attached.
    Query params:
      client_id (required)
      top (optional, default 50)
      return_features (optional, 0/1)  -> kept for backwards compatibility
    """
    client_id = request.args.get("client_id")
    if not client_id:
        return jsonify({"error":"client_id required"}), 400
    top_k = int(request.args.get("top", 50))
    return_features = request.args.get("return_features", "0") in ("1","true","True")

    x_test_path = FEATURE_DIR / f"{client_id}_X_test.npy"
    model_path = MODEL_DIR / f"{client_id}_ae.pt"
    thresh_path = MODEL_DIR / f"{client_id}_thresh.npy"
    raw_csv_path = FEATURE_DIR / f"{client_id}_X_test.csv"   # raw readable csv

    if not x_test_path.exists():
        return jsonify({"error": f"Missing features for {client_id}"}), 404
    if not model_path.exists() or not thresh_path.exists():
        return jsonify({"error": f"Missing model/threshold for {client_id}"}), 404
    if not raw_csv_path.exists():
        return jsonify({"error": f"Missing raw CSV for {client_id}. Please save CSV during preprocessing."}), 404

    # load
    X_test = np.load(str(x_test_path))
    raw_df = pd.read_csv(str(raw_csv_path))

    state = torch.load(str(model_path), map_location="cpu")
    input_dim = int(X_test.shape[1])

    # AE architecture (same as training)
    import torch.nn as nn
    class AE(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(dim,32), nn.ReLU(), nn.Linear(32,8), nn.ReLU(), nn.Linear(8,4))
            self.decoder = nn.Sequential(nn.Linear(4,8), nn.ReLU(), nn.Linear(8,32), nn.ReLU(), nn.Linear(32,dim))
        def forward(self,x): return self.decoder(self.encoder(x))

    model = AE(input_dim)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32)
        recon = model(X_t).numpy()
    mses = np.mean((recon - X_test)**2, axis=1)
    threshold = float(np.load(str(thresh_path))[0])

    indices = np.argsort(-mses)[:top_k]
    results = []
    for idx in indices:
        idx = int(idx)
        m = float(mses[idx])
        rec = raw_df.iloc[idx].to_dict() if idx < len(raw_df) else {}
        results.append({
            "idx": idx,
            "mse": m,
            "is_anomaly": bool(m > threshold),
            "record": rec if not return_features else (X_test[idx].tolist() if idx < len(X_test) else None)
        })

    return jsonify({
        "client_id": client_id,
        "threshold": threshold,
        "n_test": int(X_test.shape[0]),
        "results": results
    })

@app.route("/global", methods=["GET"])
def global_meta():
    global_model = MODEL_DIR / "global_ae.pt"
    global_thresh = MODEL_DIR / "global_thresh.npy"
    meta = {"has_global": global_model.exists()}
    if global_model.exists():
        try:
            state = torch.load(str(global_model), map_location="cpu")
            meta["n_params"] = sum(v.numel() for v in state.values()) if isinstance(state, dict) else 0
        except Exception:
            meta["n_params"] = 0
    if global_thresh.exists():
        meta["global_thresh"] = float(np.load(str(global_thresh))[0])
    return jsonify(meta)

if __name__ == "__main__":
    # Listen on all interfaces (when deployed) or 127.0.0.1 locally
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
