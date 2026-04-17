# =========================
# IMPORTS
# =========================
import numpy as np
import pandas as pd
import os
import random

import rdkit as rd
from rdkit import DataStructs
from rdkit.Chem import AllChem

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# DEVICE + SEED
# =========================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed_value = 124
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

# =========================
# SETTINGS
# =========================
morgan_bits = 4096
morgan_radius = 2
train_epoch = 50
batch_size = 512

tox21_tasks = [
    'NR-AR', 'NR-Aromatase', 'NR-PPAR-gamma', 'SR-HSE',
    'NR-AR-LBD', 'NR-ER', 'SR-ARE', 'SR-MMP',
    'NR-AhR', 'NR-ER-LBD', 'SR-ATAD5', 'SR-p53'
]

# 🔥 CUSTOMIZE THIS
pos_weights = [20, 20, 20, 30, 15, 50, 40, 15, 40, 50, 15, 30]

# =========================
# LOAD DATA
# =========================
tox21_file = 'data/datasets/tox21/raw_data/tox21.csv'
data = pd.read_csv(tox21_file)

train_data, temp_data = train_test_split(data, test_size=0.2, random_state=124)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=124)

train_data = train_data.reset_index(drop=True)
valid_data = valid_data.reset_index(drop=True)
test_data  = test_data.reset_index(drop=True)

datasets = [train_data, test_data, valid_data]

# =========================
# MORGAN FINGERPRINTS
# =========================
for i in range(len(datasets)):

    datasets[i]['mol'] = [rd.Chem.MolFromSmiles(x) for x in datasets[i]['smiles']]

    fps = []
    for mol in datasets[i]['mol']:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, morgan_radius, nBits=morgan_bits)
        arr = np.zeros((morgan_bits,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)

    datasets[i]['morgan'] = fps

# =========================
# PREP DATA
# =========================
for i in range(3):
    datasets[i] = datasets[i].fillna(-1)

train_data, test_data, valid_data = datasets

def prepare_xy(df):
    x = np.array(df['morgan'].tolist()) - 0.5
    y = df[tox21_tasks].values
    return x.astype(np.float32), y.astype(np.float32)

x_train, y_train = prepare_xy(train_data)
x_test,  y_test  = prepare_xy(test_data)
x_valid, y_valid = prepare_xy(valid_data)

# =========================
# DATASET
# =========================
class DNNData(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_loader = DataLoader(DNNData(x_train, y_train), batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(DNNData(x_valid, y_valid), batch_size=len(x_valid), shuffle=False)

input_dim = x_train.shape[1]

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
        return self.out(x).squeeze(1)  # logits

# =========================
# CREATE MODELS
# =========================
models = []
optimizers = []
criterions = []

for i in range(len(tox21_tasks)):

    model = SingleTaskDNN(input_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weights[i]], device=device)
    )

    models.append(model)
    optimizers.append(optimizer)
    criterions.append(criterion)

# =========================
# TRAINING
# =========================
for epoch in range(train_epoch):

    print(f"\nEpoch {epoch+1}/{train_epoch}")

    for task_idx, task_name in enumerate(tox21_tasks):

        model = models[task_idx]
        optimizer = optimizers[task_idx]
        criterion = criterions[task_idx]

        model.train()
        total_loss = 0

        for x_batch, y_batch in train_loader:

            x_batch = x_batch.to(device)
            y_task = y_batch[:, task_idx].to(device)

            mask = y_task >= 0
            if mask.sum() == 0:
                continue

            logits = model(x_batch)
            loss = criterion(logits[mask], y_task[mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"{task_name}: Loss = {total_loss:.4f}")

# =========================
# VALIDATION
# =========================
print("\n=== VALIDATION ===")

for task_idx, task_name in enumerate(tox21_tasks):

    model = models[task_idx]
    model.eval()

    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():

        for x_batch, y_batch in valid_loader:

            x_batch = x_batch.to(device)
            y_task = y_batch[:, task_idx].to(device)

            mask = y_task >= 0
            if mask.sum() == 0:
                continue

            logits = model(x_batch)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            y_true.extend(y_task[mask].cpu().numpy())
            y_pred.extend(preds[mask].cpu().numpy())
            y_prob.extend(probs[mask].cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    print(f"{task_name}: ACC={acc:.3f} F1={f1:.3f} AUC={auc:.3f}")

# =========================
# SAVE MODELS
# =========================
os.makedirs("independent_models", exist_ok=True)

for i, model in enumerate(models):
    torch.save(model.state_dict(), f"independent_models/{tox21_tasks[i]}.pt")

print("\n✓ Models saved")