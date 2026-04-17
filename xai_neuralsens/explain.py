#!/usr/bin/env python
# coding: utf-8

# ========================
# IMPORTS
# ========================
import os
import numpy as np
import pandas as pd
import random

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

import torch
import torch.nn as nn

# ========================
# REPRODUCIBILITY
# ========================
seed_value = 122
np.random.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)

# ========================
# SETTINGS
# ========================
morgan_bits = 4096
morgan_radius = 2

tox21_file = 'data/datasets/tox21/raw_data/tox21.csv'

tox21_tasks = [
    'NR-AR', 'NR-Aromatase', 'NR-PPAR-gamma', 'SR-HSE',
    'NR-AR-LBD', 'NR-ER', 'SR-ARE', 'SR-MMP',
    'NR-AhR', 'NR-ER-LBD', 'SR-ATAD5', 'SR-p53'
]

# 🔥 choose task index here
explain_task = 1
task_name = tox21_tasks[explain_task]

# ========================
# LOAD DATA
# ========================
raw_data = pd.read_csv(tox21_file)
raw_data = raw_data.dropna(subset=['smiles'])

raw_data['mol'] = raw_data['smiles'].apply(Chem.MolFromSmiles)

# ========================
# MORGAN FINGERPRINTS + bitInfo
# ========================
bit_infos = [{} for _ in range(len(raw_data))]

fps = []
for i, mol in enumerate(raw_data['mol']):
    fp = GetMorganFingerprintAsBitVect(
        mol,
        morgan_radius,
        nBits=morgan_bits,
        bitInfo=bit_infos[i]
    )
    arr = np.zeros((morgan_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    fps.append(arr)

raw_data['bitInfo'] = bit_infos

X = np.array(fps, dtype=np.float32)

# ========================
# MODEL (MATCH TRAINING)
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
        return self.out(x).squeeze(1)  # logits

# ========================
# LOAD TRAINED MODEL
# ========================
model_path = f"independent_models/{task_name}.pt"

model = SingleTaskDNN(X.shape[1])
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

print(f"Loaded model for task: {task_name}")

# ========================
# DEBUG: CHECK OUTPUTS
# ========================
X_tensor = torch.tensor(X, dtype=torch.float32)

with torch.no_grad():
    logits = model(X_tensor[:100])
    probs = torch.sigmoid(logits)

    print("\nDEBUG:")
    print("Logits mean/std:", logits.mean().item(), logits.std().item())
    print("Probs mean/std:", probs.mean().item(), probs.std().item())

# ========================
# GRADIENT ATTRIBUTIONS
# ========================
print("\nComputing gradient-based attributions...")

attr = []

for i in range(X_tensor.shape[0]):

    x = X_tensor[i:i+1].clone().detach().requires_grad_(True)

    model.zero_grad()

    output = model(x)  # logits
    output.backward()

    grad = x.grad.detach().cpu().numpy().flatten()
    attr.append(grad)

attr = np.array(attr)

print("Attributions shape:", attr.shape)

# ========================
# AGGREGATE IMPORTANCE
# ========================
mean_attr = np.mean(np.abs(attr), axis=0)

top_n = 10
important_bits = np.argsort(mean_attr)[::-1][:top_n]

# ========================
# MAP BITS → SUBSTRUCTURES
# ========================
results = []

for bit in important_bits:

    idx = np.where(X[:, bit] == 1)[0]
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
results_df.to_csv(f'neuralsens_{task_name}_attributions.csv', index=False)

print("\nTop features:")
print(results_df[['bit', 'mean_abs_attr', 'example_smiles', 'bit_smiles']])