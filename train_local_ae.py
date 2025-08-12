# train_local_ae.py
import argparse, os, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.metrics import precision_recall_fscore_support

ROOT = r"C:\Users\MANGIPUDI DEEPA\Desktop\fed-id-misuse"
FEATURE_DIR = os.path.join(ROOT, "features")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

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

def train(client, epochs=20, batch=64, lr=1e-3):
    X_train = np.load(os.path.join(FEATURE_DIR, f"{client}_X_train.npy"))
    X_val = np.load(os.path.join(FEATURE_DIR, f"{client}_X_test.npy"))
    labels_path = os.path.join(FEATURE_DIR, f"{client}_labels.npy")
    labels = np.load(labels_path, allow_pickle=True) if os.path.exists(labels_path) else None

    device = torch.device("cpu")
    input_dim = X_train.shape[1]
    model = AE(input_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train,dtype=torch.float32)),
                              batch_size=batch, shuffle=True)

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for (xb,) in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        print(f"[{client}] Epoch {ep+1}/{epochs} loss={total_loss/len(X_train):.6f}")

    # compute recon errors on val
    model.eval()
    with torch.no_grad():
        Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
        recon = model(Xv).cpu().numpy()
    errors = np.mean((recon - X_val)**2, axis=1)
    thresh = float(np.percentile(errors, 95))
    # save
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{client}_ae.pt"))
    np.save(os.path.join(MODEL_DIR, f"{client}_thresh.npy"), np.array([thresh]))
    print("Saved model and threshold:", client, "threshold=", thresh)

    # optional evaluation if labels exist
    if labels is not None:
        y_true = (labels != 'none').astype(int)
        y_pred = (errors > thresh).astype(int)
        p,r,f,_ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        print("Eval P/R/F:", p, r, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    train(args.client, epochs=args.epochs, batch=args.batch)
