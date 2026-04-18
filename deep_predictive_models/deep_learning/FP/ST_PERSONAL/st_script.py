import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm

import rdkit as rd
from rdkit import DataStructs
from rdkit.Chem import AllChem

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

# =========================
# DEVICE + SEED
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 124
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# =========================
# SETTINGS
# =========================
morgan_bits = 4096
morgan_radius = 2
epochs = 20
batch_size = 512

tox21_tasks = [
    'NR-AR','NR-Aromatase','NR-PPAR-gamma','SR-HSE',
    'NR-AR-LBD','NR-ER','SR-ARE','SR-MMP',
    'NR-AhR','NR-ER-LBD','SR-ATAD5','SR-p53'
]

os.makedirs("models_baseline", exist_ok=True)
os.makedirs("models_perm", exist_ok=True)

# =========================
# DATA
# =========================
data = pd.read_csv('data/datasets/tox21/raw_data/tox21.csv')

train, temp = train_test_split(data, test_size=0.2, random_state=seed)
valid, test = train_test_split(temp, test_size=0.5, random_state=seed)

# =========================
# FINGERPRINTS
# =========================
def compute_fp(df):
    mols = [rd.Chem.MolFromSmiles(x) for x in df['smiles']]
    fps = []

    for mol in mols:
        if mol is None:
            fps.append(np.zeros(morgan_bits))
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, morgan_radius, nBits=morgan_bits
        )
        arr = np.zeros((morgan_bits,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)

    return np.array(fps, dtype=np.float32)

X_train = compute_fp(train) - 0.5
Y_train = train[tox21_tasks].fillna(-1).values.astype(np.float32)

# =========================
# MODEL
# =========================
class SingleTaskDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.out = nn.Linear(512, 1)
        self.act = nn.LeakyReLU(0.05)

    def forward(self, x):
        x = self.act(self.bn1(self.fc1(x)))
        x = self.act(self.bn2(self.fc2(x)))
        return self.out(x).squeeze(1)


# =========================
# TRAIN FUNCTION
# =========================
def train_model(X, y, task_idx, save_path):

    if os.path.exists(save_path):
        print(f"✔ Skipping (exists): {save_path}")
        return

    model = SingleTaskDNN(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for ep in tqdm(range(epochs), desc=f"Training {os.path.basename(save_path)}"):

        model.train()
        idx = np.random.permutation(len(X))

        for i in range(0, len(idx), batch_size):
            batch = idx[i:i+batch_size]

            x = torch.tensor(X[batch], dtype=torch.float32).to(device)
            yb = torch.tensor(y[batch, task_idx], dtype=torch.float32).to(device)

            mask = yb >= 0
            if mask.sum() == 0:
                continue

            logits = model(x)
            loss = loss_fn(logits[mask], yb[mask])

            opt.zero_grad()
            loss.backward()
            opt.step()

    torch.save(model.state_dict(), save_path)

# =========================
# PERMUTATION FUNCTION
# =========================
def permute_features(X):
    Xp = X.copy()
    for i in range(Xp.shape[0]):
        np.random.shuffle(Xp[i])
    return Xp

# =========================
# MAIN LOOP
# =========================
for t_idx, task in enumerate(tqdm(tox21_tasks, desc="Tasks")):

    print(f"\n=== TASK: {task} ===")

    # -------------------------
    # BASELINE
    # -------------------------
    base_path = f"models_baseline/{task}.pt"

    train_model(
        X_train,
        Y_train,
        t_idx,
        base_path
    )

    # -------------------------
    # PERMUTED
    # -------------------------
    perm_path = f"models_perm/{task}.pt"

    X_perm = permute_features(X_train)

    train_model(
        X_perm,
        Y_train,
        t_idx,
        perm_path
    )

print("\nDONE")