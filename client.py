# # import os
# # import sys
# # import torch
# # import torch.nn as nn
# # import flwr as fl
# # from torch.utils.data import DataLoader, TensorDataset

# # # Detect the client ID from command line args
# # if len(sys.argv) < 2:
# #     print("Usage: python client.py <client_id>")
# #     sys.exit(1)

# # client_id = sys.argv[1]

# # # Paths
# # features_dir = "features"
# # model_dir = "models"

# # # Load features
# # x_train_path = os.path.join(features_dir, f"{client_id}_X_train.npy")
# # y_train_path = os.path.join(features_dir, f"{client_id}_labels.npy")
# # x_test_path = os.path.join(features_dir, f"{client_id}_X_val.npy")
# # y_test_path = os.path.join(features_dir, f"{client_id}_scaler.joblib")

# # if not os.path.exists(x_train_path):
# #     raise FileNotFoundError(f"Missing feature file: {x_train_path}")

# # X_train = torch.load(x_train_path)
# # y_train = torch.load(y_train_path)
# # X_test = torch.load(x_test_path)
# # y_test = torch.load(y_test_path)

# # input_dim = X_train.shape[1]

# # # Autoencoder definition (matching your original training arch)
# # class Autoencoder(nn.Module):
# #     def __init__(self, input_dim):
# #         super(Autoencoder, self).__init__()
# #         # Adjusted to match your saved model: 12→32→8→4→8→32→12 (for 12-dim case)
# #         self.encoder = nn.Sequential(
# #             nn.Linear(input_dim, 32),
# #             nn.ReLU(),
# #             nn.Linear(32, 8),
# #             nn.ReLU(),
# #             nn.Linear(8, 4),
# #             nn.ReLU()
# #         )
# #         self.decoder = nn.Sequential(
# #             nn.Linear(4, 8),
# #             nn.ReLU(),
# #             nn.Linear(8, 32),
# #             nn.ReLU(),
# #             nn.Linear(32, input_dim),
# #             nn.Sigmoid()
# #         )

# #     def forward(self, x):
# #         encoded = self.encoder(x)
# #         decoded = self.decoder(encoded)
# #         return decoded

# # # Instantiate model
# # model = Autoencoder(input_dim)

# # # Load pretrained weights
# # model_path = os.path.join(model_dir, f"{client_id}.pt")
# # if not os.path.exists(model_path):
# #     raise FileNotFoundError(f"Model file not found: {model_path}")

# # state_dict = torch.load(model_path, map_location=torch.device("cpu"))
# # model.load_state_dict(state_dict)

# # # Loss and optimizer
# # criterion = nn.MSELoss()
# # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # # Create DataLoaders
# # train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
# # test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

# # # Flower client
# # class FlowerClient(fl.client.NumPyClient):
# #     def get_parameters(self, config):
# #         return [val.cpu().numpy() for val in model.state_dict().values()]

# #     def set_parameters(self, parameters):
# #         params_dict = zip(model.state_dict().keys(), parameters)
# #         state_dict = {k: torch.tensor(v) for k, v in params_dict}
# #         model.load_state_dict(state_dict, strict=True)

# #     def fit(self, parameters, config):
# #         self.set_parameters(parameters)
# #         model.train()
# #         for epoch in range(1):  # One local epoch
# #             for xb, yb in train_loader:
# #                 optimizer.zero_grad()
# #                 outputs = model(xb)
# #                 loss = criterion(outputs, yb)
# #                 loss.backward()
# #                 optimizer.step()
# #         return self.get_parameters(config={}), len(train_loader.dataset), {}

# #     def evaluate(self, parameters, config):
# #         self.set_parameters(parameters)
# #         model.eval()
# #         loss = 0
# #         with torch.no_grad():
# #             for xb, yb in test_loader:
# #                 outputs = model(xb)
# #                 loss += criterion(outputs, yb).item()
# #         loss /= len(test_loader)
# #         return float(loss), len(test_loader.dataset), {}

# # # Start client
# # fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient())

# import os
# import argparse
# import numpy as np
# import joblib
# import torch
# import torch.nn as nn
# import flwr as fl
# from torch.utils.data import DataLoader, TensorDataset

# # --------------------------
# # Parse arguments
# # --------------------------
# parser = argparse.ArgumentParser()
# parser.add_argument("--client_id", required=True, help="Client identifier (e.g., bank)")
# parser.add_argument("--features_dir", required=True, help="Directory containing features")
# parser.add_argument("--server_addr", required=True, help="Server address (e.g., localhost:8080)")
# args = parser.parse_args()

# client_id = args.client_id
# features_dir = args.features_dir
# model_dir = "models"

# # --------------------------
# # Load features (.npy or .joblib)
# # --------------------------
# def load_file(base_path):
#     if os.path.exists(base_path + ".npy"):
#         return torch.tensor(np.load(base_path + ".npy"), dtype=torch.float32)
#     elif os.path.exists(base_path + ".joblib"):
#         return torch.tensor(joblib.load(base_path + ".joblib"), dtype=torch.float32)
#     else:
#         raise FileNotFoundError(f"Missing file for base path: {base_path}")

# X_train = load_file(os.path.join(features_dir, f"{client_id}_X_train"))
# y_train = load_file(os.path.join(features_dir, f"{client_id}_labels"))
# X_test = load_file(os.path.join(features_dir, f"{client_id}_X_val"))
# y_test = load_file(os.path.join(features_dir, f"{client_id}_labels"))  # adjust if you have y_val separately

# input_dim = X_train.shape[1]

# # --------------------------
# # Autoencoder
# # --------------------------
# class Autoencoder(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 32),
#             nn.ReLU(),
#             nn.Linear(32, 8),
#             nn.ReLU(),
#             nn.Linear(8, 4),
#             nn.ReLU()
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(4, 8),
#             nn.ReLU(),
#             nn.Linear(8, 32),
#             nn.ReLU(),
#             nn.Linear(32, input_dim),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.decoder(self.encoder(x))

# model = Autoencoder(input_dim)

# # --------------------------
# # Load pretrained weights
# # --------------------------
# model_path = os.path.join(model_dir, f"{client_id}.pt")
# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"Model file not found: {model_path}")
# model.load_state_dict(torch.load(model_path, map_location="cpu"))

# # --------------------------
# # Training setup
# # --------------------------
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
# test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

# # --------------------------
# # Flower client
# # --------------------------
# class FlowerClient(fl.client.NumPyClient):
#     def get_parameters(self, config):
#         return [val.cpu().numpy() for val in model.state_dict().values()]

#     def set_parameters(self, parameters):
#         keys = list(model.state_dict().keys())
#         state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
#         model.load_state_dict(state_dict, strict=True)

#     def fit(self, parameters, config):
#         self.set_parameters(parameters)
#         model.train()
#         for xb, yb in train_loader:
#             optimizer.zero_grad()
#             outputs = model(xb)
#             loss = criterion(outputs, yb)
#             loss.backward()
#             optimizer.step()
#         return self.get_parameters({}), len(train_loader.dataset), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         model.eval()
#         loss = 0
#         with torch.no_grad():
#             for xb, yb in test_loader:
#                 outputs = model(xb)
#                 loss += criterion(outputs, yb).item()
#         return float(loss / len(test_loader)), len(test_loader.dataset), {}

# # --------------------------
# # Start client
# # --------------------------
# fl.client.start_numpy_client(server_address=args.server_addr, client=FlowerClient())


# import argparse
# import socket
# import torch
# import torch.nn as nn
# import numpy as np
# import joblib
# import os

# # ---- Autoencoder Model Definition ----
# class Autoencoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim=64):
#         super(Autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU()
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(hidden_dim, input_dim),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded

# # ---- Client Main Script ----
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--client_id", required=True, help="Client identifier (e.g., bank, aadhar)")
#     parser.add_argument("--features_dir", required=True, help="Path to features folder")
#     parser.add_argument("--local_epochs", type=int, default=1, help="Local training epochs")
#     parser.add_argument("--batch_size", type=int, default=128, help="Batch size for local training")
#     parser.add_argument("--server_addr", required=True, help="Server address in host:port format")
#     args = parser.parse_args()

#     # ---- Resolve feature file paths ----
#     x_train_path = os.path.join(args.features_dir, f"{args.client_id}_x_train.npy")
#     x_test_path = os.path.join(args.features_dir, f"{args.client_id}_x_test.npy")

#     if not os.path.exists(x_train_path):
#         raise FileNotFoundError(f"Missing feature file: {x_train_path}")
#     if not os.path.exists(x_test_path):
#         raise FileNotFoundError(f"Missing feature file: {x_test_path}")

#     X_train = np.load(x_train_path)
#     X_test = np.load(x_test_path)

#     input_dim = X_train.shape[1]

#     # ---- Load model and threshold ----
#     model_path = os.path.join("models", f"{args.client_id}_ae.pt")
#     thresh_path = os.path.join("models", f"{args.client_id}_thresh.npy")

#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file not found: {model_path}")
#     if not os.path.exists(thresh_path):
#         raise FileNotFoundError(f"Threshold file not found: {thresh_path}")

#     model = Autoencoder(input_dim)
#     model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
#     model.eval()

#     threshold = np.load(thresh_path)

#     # ---- Evaluate on test data ----
#     X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
#     with torch.no_grad():
#         recon = model(X_test_tensor)
#         mse = torch.mean((recon - X_test_tensor) ** 2, dim=1).numpy()

#     # Calculate accuracy (as a simple example)
#     y_pred = (mse > threshold).astype(int)  # 1 = anomaly
#     accuracy = (y_pred == 0).mean()  # Assuming all test samples are normal

#     print(f"[{args.client_id}] Local evaluation complete. Accuracy: {accuracy:.4f}")

#     # ---- Send results to server ----
#     host, port = args.server_addr.split(":")
#     port = int(port)

#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.connect((host, port))
#         message = f"{args.client_id},{accuracy}"
#         s.sendall(message.encode("utf-8"))
#         print(f"[{args.client_id}] Sent accuracy to server at {args.server_addr}")




# import argparse
# import socket
# import torch
# import torch.nn as nn
# import numpy as np
# import os

# # ---- Autoencoder definition (must match train_local_ae.py exactly) ----
# class AE(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(dim, 32), nn.ReLU(),
#             nn.Linear(32, 8), nn.ReLU(),
#             nn.Linear(8, 4)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(4, 8), nn.ReLU(),
#             nn.Linear(8, 32), nn.ReLU(),
#             nn.Linear(32, dim)
#         )
#     def forward(self, x):
#         z = self.encoder(x)
#         return self.decoder(z)

# # ---- Client Script ----
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--client_id", required=True, help="Client identifier (e.g., bank, telecom)")
#     parser.add_argument("--features_dir", required=True, help="Path to features folder")
#     parser.add_argument("--server_addr", required=True, help="Server address in host:port format")
    
#     args = parser.parse_args()

#     # Feature file paths
#     x_train_path = os.path.join(args.features_dir, f"{args.client_id}_X_train.npy")
#     x_test_path = os.path.join(args.features_dir, f"{args.client_id}_X_test.npy")

#     if not os.path.exists(x_train_path):
#         raise FileNotFoundError(f"Missing feature file: {x_train_path}")
#     if not os.path.exists(x_test_path):
#         raise FileNotFoundError(f"Missing feature file: {x_test_path}")

#     # Load data
#     X_train = np.load(x_train_path)
#     X_test = np.load(x_test_path)
#     input_dim = X_train.shape[1]

#     # Model + threshold paths
#     model_path = os.path.join("models", f"{args.client_id}_ae.pt")
#     thresh_path = os.path.join("models", f"{args.client_id}_thresh.npy")

#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file not found: {model_path}")
#     if not os.path.exists(thresh_path):
#         raise FileNotFoundError(f"Threshold file not found: {thresh_path}")

#     # Load trained model
#     model = AE(input_dim)
#     model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
#     model.eval()

#     # Load threshold
#     threshold = np.load(thresh_path)[0]

#     # Evaluate on test set
#     X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
#     with torch.no_grad():
#         recon = model(X_test_tensor)
#         mse = torch.mean((recon - X_test_tensor) ** 2, dim=1).numpy()

#     # Predictions
#     y_pred = (mse > threshold).astype(int)  # 1 = anomaly, 0 = normal
#     accuracy = (y_pred == 0).mean()  # assuming test set is normal

#     print(f"[{args.client_id}] Accuracy: {accuracy:.4f} (Threshold: {threshold:.6f})")

#     # Send results to server
#     host, port = args.server_addr.split(":")
#     port = int(port)

#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.connect((host, port))
#         message = f"{args.client_id},{accuracy}"
#         s.sendall(message.encode("utf-8"))
#         print(f"[{args.client_id}] Sent accuracy to server at {args.server_addr}")


#sends send_binary_update








# # client.py
# import argparse
# import socket
# import torch
# import torch.nn as nn
# import numpy as np
# import os
# import pickle

# # ---- Autoencoder (must match train_local_ae.py) ----
# class AE(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(dim, 32), nn.ReLU(),
#             nn.Linear(32, 8), nn.ReLU(),
#             nn.Linear(8, 4)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(4, 8), nn.ReLU(),
#             nn.Linear(8, 32), nn.ReLU(),
#             nn.Linear(32, dim)
#         )
#     def forward(self, x):
#         z = self.encoder(x)
#         return self.decoder(z)

# def recvall(conn, n):
#     data = b""
#     while len(data) < n:
#         packet = conn.recv(min(65536, n - len(data)))
#         if not packet:
#             return None
#         data += packet
#     return data

# def send_length_prefixed(sock, data_bytes: bytes):
#     sock.sendall(len(data_bytes).to_bytes(8, 'big'))
#     sock.sendall(data_bytes)

# def receive_length_prefixed(sock):
#     header = recvall(sock, 8)
#     if not header:
#         return None
#     size = int.from_bytes(header, 'big')
#     if size <= 0:
#         return None
#     payload = recvall(sock, size)
#     return payload

# # ---- Client script ----
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--client_id", required=True)
#     parser.add_argument("--features_dir", required=True)
#     parser.add_argument("--server_addr", required=True, help="host:port")
#     parser.add_argument("--send_weights", action="store_true", help="send model weights and wait for global model back")
#     args = parser.parse_args()

#     host, port_s = args.server_addr.split(":")
#     port = int(port_s)

#     x_train_path = os.path.join(args.features_dir, f"{args.client_id}_X_train.npy")
#     x_test_path = os.path.join(args.features_dir, f"{args.client_id}_X_test.npy")
#     if not os.path.exists(x_train_path) or not os.path.exists(x_test_path):
#         raise FileNotFoundError("Missing feature files; run preprocess first")

#     X_train = np.load(x_train_path)
#     X_test = np.load(x_test_path)
#     n_train = X_train.shape[0]
#     input_dim = X_train.shape[1]

#     model_path = os.path.join("models", f"{args.client_id}_ae.pt")
#     thresh_path = os.path.join("models", f"{args.client_id}_thresh.npy")
#     if not os.path.exists(model_path) or not os.path.exists(thresh_path):
#         raise FileNotFoundError("Missing trained model/threshold for client")

#     model = AE(input_dim)
#     model.load_state_dict(torch.load(model_path, map_location="cpu"))
#     model.eval()

#     threshold = float(np.load(thresh_path)[0])

#     # Evaluate on test
#     X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
#     with torch.no_grad():
#         recon = model(X_test_tensor)
#         mse = torch.mean((recon - X_test_tensor) ** 2, dim=1).numpy()
#     y_pred = (mse > threshold).astype(int)
#     accuracy = float((y_pred == 0).mean())

#     print(f"[{args.client_id}] Accuracy: {accuracy:.4f} (thresh={threshold:.6f})")

#     # Connect and send
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.connect((host, port))

#         if args.send_weights:
#             # build payload: include n_samples so server can do weighted avg
#             # ensure state_dict values are CPU tensors (they usually are)
#             state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model.state_dict().items()}
#             payload = {
#                 "client_id": args.client_id,
#                 "accuracy": accuracy,
#                 "state_dict": state_dict,
#                 "threshold": float(threshold),
#                 "n_samples": int(n_train)
#             }
#             data_bytes = pickle.dumps(payload)
#             print(f"[{args.client_id}] Sending binary update ({len(data_bytes)/1024:.1f} KB)...")
#             send_length_prefixed(s, data_bytes)

#             # Wait for length-prefixed response (global model)
#             resp = receive_length_prefixed(s)
#             if resp is None:
#                 print("[client] No response or invalid response from server")
#             else:
#                 try:
#                     resp_obj = pickle.loads(resp)
#                 except Exception as e:
#                     print(f"[client] Could not unpickle server response: {e}")
#                 else:
#                     gstate = resp_obj.get("global_state_dict")
#                     g_n = resp_obj.get("n_total")
#                     g_thresh = resp_obj.get("global_threshold", None)
#                     if gstate:
#                         # Convert values to tensors if needed
#                         safe_state = {}
#                         for k, v in gstate.items():
#                             if isinstance(v, torch.Tensor):
#                                 safe_state[k] = v
#                             else:
#                                 try:
#                                     safe_state[k] = torch.tensor(v)
#                                 except:
#                                     safe_state[k] = torch.tensor(np.array(v))
#                         # overwrite local model
#                         model.load_state_dict(safe_state)
#                         torch.save(model.state_dict(), model_path)
#                         if g_thresh is not None:
#                             np.save(thresh_path, np.array([float(g_thresh)]))
#                         print(f"[{args.client_id}] Received global model (n_total={g_n}). Local model updated and saved.")
#                     else:
#                         print("[client] Server response did not contain global model.")
#         else:
#             # fallback: send simple csv text (legacy)
#             message = f"{args.client_id},{accuracy}"
#             s.sendall(message.encode('utf-8'))
#             resp = s.recv(4096)
#             print("[client] Server response:", resp.decode('utf-8', errors='replace'))

#     print(f"[{args.client_id}] Done.")




# client.py
import argparse, socket, pickle, os, numpy as np, torch, torch.nn as nn

class AE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 32), nn.ReLU(),
            nn.Linear(32, 8), nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(),
            nn.Linear(8, 32), nn.ReLU(),
            nn.Linear(32, dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def recvall(conn, n):
    data = b""
    while len(data) < n:
        packet = conn.recv(min(65536, n - len(data)))
        if not packet:
            return None
        data += packet
    return data

def send_length_prefixed(sock, data_bytes: bytes):
    sock.sendall(len(data_bytes).to_bytes(8, 'big') + data_bytes)

def receive_length_prefixed(sock):
    header = recvall(sock, 8)
    if not header:
        return None
    size = int.from_bytes(header, 'big')
    payload = recvall(sock, size)
    return payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", required=True)
    parser.add_argument("--features_dir", required=True)
    parser.add_argument("--server_addr", required=True, help="host:port")
    parser.add_argument("--send_weights", action="store_true")
    args = parser.parse_args()

    host, port_s = args.server_addr.split(":")
    port = int(port_s)

    x_train_path = os.path.join(args.features_dir, f"{args.client_id}_X_train.npy")
    x_test_path = os.path.join(args.features_dir, f"{args.client_id}_X_test.npy")
    if not os.path.exists(x_train_path) or not os.path.exists(x_test_path):
        raise FileNotFoundError("Missing feature files; run preprocess first")

    X_train = np.load(x_train_path)
    X_test = np.load(x_test_path)
    n_train = int(X_train.shape[0])
    input_dim = int(X_train.shape[1])

    model_path = os.path.join("models", f"{args.client_id}_ae.pt")
    thresh_path = os.path.join("models", f"{args.client_id}_thresh.npy")
    if not os.path.exists(model_path) or not os.path.exists(thresh_path):
        raise FileNotFoundError("Missing trained model/threshold for client")

    model = AE(input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    threshold = float(np.load(thresh_path)[0])

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        recon = model(X_test_tensor)
        mse = torch.mean((recon - X_test_tensor) ** 2, dim=1).numpy()
    y_pred = (mse > threshold).astype(int)
    accuracy = float((y_pred == 0).mean())

    print(f"[{args.client_id}] Accuracy: {accuracy:.4f} (thresh={threshold:.6f})")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))

        if args.send_weights:
            # pack state_dict (cpu tensors)
            state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model.state_dict().items()}
            payload = {
                "client_id": args.client_id,
                "accuracy": accuracy,
                "state_dict": state_dict,
                "threshold": float(threshold),
                "n_samples": int(n_train)
            }
            data_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[{args.client_id}] Sending binary update ({len(data_bytes)/1024:.1f} KB)...")
            send_length_prefixed(s, data_bytes)

            # receive reply
            resp = receive_length_prefixed(s)
            if resp is None:
                print("[client] No response or invalid response from server")
            else:
                try:
                    resp_obj = pickle.loads(resp)
                except Exception as e:
                    print(f"[client] Could not unpickle server response: {e}")
                else:
                    gstate = resp_obj.get("global_state_dict")
                    g_n = resp_obj.get("n_total")
                    g_thresh = resp_obj.get("global_threshold", None)
                    if gstate:
                        # ensure convert to tensors if necessary
                        safe_state = {}
                        for k, v in gstate.items():
                            if isinstance(v, torch.Tensor):
                                safe_state[k] = v
                            else:
                                try:
                                    safe_state[k] = torch.tensor(v)
                                except:
                                    safe_state[k] = torch.tensor(np.array(v))
                        model.load_state_dict(safe_state)
                        torch.save(model.state_dict(), model_path)
                        if g_thresh is not None:
                            np.save(thresh_path, np.array([float(g_thresh)]))
                        print(f"[{args.client_id}] Received global model (n_total={g_n}). Local model updated and saved.")
                    else:
                        print("[client] Server response did not contain global model.")
        else:
            # legacy: send "client,accuracy" text
            message = f"{args.client_id},{accuracy}"
            s.sendall(message.encode('utf-8'))
            resp = s.recv(4096)
            print("[client] Server response:", resp.decode('utf-8', errors='replace'))

    print(f"[{args.client_id}] Done.")
