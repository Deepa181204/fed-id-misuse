import os, numpy as np, torch
ROOT = r"C:\Users\MANGIPUDI DEEPA\Desktop\fed-id-misuse"
FEATURE_DIR = os.path.join(ROOT, "features")
MODEL_DIR = os.path.join(ROOT, "models")

def score(client):
    Xv = np.load(os.path.join(FEATURE_DIR, f"{client}_X_val.npy"))
    model_path = os.path.join(MODEL_DIR, f"{client}_ae.pt")
    thresh = float(np.load(os.path.join(MODEL_DIR, f"{client}_thresh.npy"))[0])
    import torch, torch.nn as nn
    # define AE same as training
    class AE(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(dim,32), nn.ReLU(), nn.Linear(32,8), nn.ReLU(), nn.Linear(8,4))
            self.decoder = nn.Sequential(nn.Linear(4,8), nn.ReLU(), nn.Linear(8,32), nn.ReLU(), nn.Linear(32,dim))
        def forward(self,x):
            return self.decoder(self.encoder(x))
    model = AE(Xv.shape[1])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    import torch
    with torch.no_grad():
        recon = model(torch.tensor(Xv,dtype=torch.float32)).numpy()
    errors = ((recon - Xv)**2).mean(axis=1)
    flagged = np.where(errors > thresh)[0]
    print(f"{client}: flagged {len(flagged)}/{len(Xv)} rows (threshold={thresh})")
    return flagged, errors

if __name__ == "__main__":
    import sys
    client = sys.argv[1] if len(sys.argv)>1 else "bank"
    score(client)
