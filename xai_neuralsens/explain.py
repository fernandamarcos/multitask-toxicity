#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

import torch
import torch.nn as nn

# ========================
# DEVICE
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========================
# SETTINGS
# ========================
morgan_bits = 4096
morgan_radius = 2
batch_size = 128  # increase if GPU allows

tox21_file = 'data/datasets/tox21/raw_data/tox21.csv'

tox21_tasks = [
    'NR-AR', 'NR-Aromatase', 'NR-PPAR-gamma', 'SR-HSE',
    'NR-AR-LBD', 'NR-ER', 'SR-ARE', 'SR-MMP',
    'NR-AhR', 'NR-ER-LBD', 'SR-ATAD5', 'SR-p53'
]

# ========================
# LOAD DATA
# ========================
raw_data = pd.read_csv(tox21_file)
raw_data = raw_data.dropna(subset=['smiles'])
raw_data['mol'] = raw_data['smiles'].apply(Chem.MolFromSmiles)

# ========================
# FINGERPRINTS
# ========================
bit_infos = [{} for _ in range(len(raw_data))]
fps = []

for i, mol in enumerate(raw_data['mol']):
    fp = GetMorganFingerprintAsBitVect(
        mol, morgan_radius, nBits=morgan_bits, bitInfo=bit_infos[i]
    )
    arr = np.zeros((morgan_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    fps.append(arr)

raw_data['bitInfo'] = bit_infos

# 🔥 IMPORTANT: match training preprocessing
X = np.array(fps, dtype=np.float32) - 0.5
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

# ========================
# MODEL
# ========================
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

# ========================
# LOOP OVER TASKS
# ========================
for task_name in tox21_tasks:

    print("\n======================")
    print(f"Processing task: {task_name}")
    print("======================")

    # ========================
    # LOAD MODEL
    # ========================
    model_path = f"independent_models/{task_name}.pt"

    if not os.path.exists(model_path):
        print(f"⚠️ Model not found: {model_path}")
        continue

    model = SingleTaskDNN(X.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ========================
    # DEBUG
    # ========================
    with torch.no_grad():
        logits = model(X_tensor[:100])
        probs = torch.sigmoid(logits)

        print("Logits mean/std:", logits.mean().item(), logits.std().item())
        print("Probs mean/std:", probs.mean().item(), probs.std().item())

    # ========================
    # ATTRIBUTIONS (FAST)
    # ========================
    print("Computing attributions...")

    attr = []

    for i in tqdm(range(0, X_tensor.shape[0], batch_size)):

        x = X_tensor[i:i+batch_size].clone().detach().requires_grad_(True)

        model.zero_grad(set_to_none=True)
        outputs = model(x)

        outputs.sum().backward()

        grads = x.grad.detach().cpu().numpy()
        attr.append(grads)

    attr = np.vstack(attr)

    # ========================
    # AGGREGATE
    # ========================
    mean_attr = np.mean(np.abs(attr), axis=0)

    top_n = 10
    important_bits = np.argsort(mean_attr)[::-1][:top_n]

    # ========================
    # MAP BITS → SUBSTRUCTURES
    # ========================
    results = []

    for bit in important_bits:

        idx = np.where(X[:, bit] > 0)[0]
        if len(idx) == 0:
            continue

        i = idx[0]

        mol = raw_data.iloc[i]['mol']
        bitInfo = raw_data.iloc[i]['bitInfo']

        if bit not in bitInfo:
            continue

        atom_center, bit_radius = bitInfo[bit][0]

        env = Chem.FindAtomEnvironmentOfRadiusN(mol, bit_radius, atom_center)
        amap = {}

        submol = Chem.PathToSubmol(mol, env, atomMap=amap)

        try:
            if bit_radius != 0:
                bit_smiles = Chem.MolToSmiles(
                    submol,
                    rootedAtAtom=amap.get(atom_center, -1),
                    canonical=True
                )
            else:
                bit_smiles = mol.GetAtomWithIdx(atom_center).GetSymbol()
        except:
            bit_smiles = None

        svg = Draw.DrawMorganBit(mol, bit, bitInfo, useSVG=True)

        results.append({
            'bit': bit,
            'mean_abs_attr': mean_attr[bit],
            'example_smiles': raw_data.iloc[i]['smiles'],
            'bit_smiles': bit_smiles,
            'svg': svg
        })

    # ========================
    # SAVE RESULTS
    # ========================
    results_df = pd.DataFrame(results)

    output_file = f'neuralsens_{task_name}_attributions.csv'
    results_df.to_csv(output_file, index=False)

    print(f"✅ Saved: {output_file}")

print("\n🎯 All tasks completed.")