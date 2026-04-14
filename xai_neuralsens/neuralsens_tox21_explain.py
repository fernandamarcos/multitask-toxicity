#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

import torch
import torch.nn as nn
import random

# ========================
# Reproducibility
# ========================
seed_value = 122
np.random.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)

# ========================
# Settings
# ========================
morgan_bits = 4096   # ✅ MUST match trained model
morgan_radius = 2
explain_task = 0

tox21_file = 'data/datasets/tox21/raw_data/tox21.csv'

tox21_tasks = [
    'NR-AR', 'NR-Aromatase', 'NR-PPAR-gamma', 'SR-HSE',
    'NR-AR-LBD', 'NR-ER', 'SR-ARE', 'SR-MMP',
    'NR-AhR', 'NR-ER-LBD', 'SR-ATAD5', 'SR-p53'
]

task = tox21_tasks[explain_task]

# ========================
# Load Data
# ========================
raw_data = pd.read_csv(tox21_file)
raw_data = raw_data.dropna(subset=['smiles'])
raw_data['mol'] = raw_data['smiles'].apply(Chem.MolFromSmiles)

# Morgan fingerprints + bitInfo
bi = [{} for _ in range(len(raw_data))]

raw_data['morgan'] = [
    GetMorganFingerprintAsBitVect(
        m, morgan_radius, nBits=morgan_bits, bitInfo=bi[i]
    )
    for i, m in enumerate(raw_data['mol'])
]

raw_data['bitInfo'] = bi

# Convert to numpy
X = []
for fp in raw_data['morgan']:
    arr = np.zeros((morgan_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    X.append(arr)

X = np.array(X)

# ========================
# Model Definition (MATCH TRAINING)
# ========================
class DNN(torch.nn.Module):
    def __init__(self, input_shape, all_tasks):
        super(DNN, self).__init__()

        self.hidden_1 = torch.nn.ModuleList([torch.nn.Linear(input_shape, 1024) for task in all_tasks])
        self.batchnorm_1 = torch.nn.ModuleList([torch.nn.BatchNorm1d(1024) for task in all_tasks])
        
        self.hidden_2 = torch.nn.ModuleList([torch.nn.Linear(1024, 512) for task in all_tasks])
        self.batchnorm_2 = torch.nn.ModuleList([torch.nn.BatchNorm1d(512) for task in all_tasks])
        
        self.output   = torch.nn.ModuleList([torch.nn.Linear(512, 1) for task in all_tasks])
        
        # function for leaky ReLU
        self.leakyReLU = torch.nn.LeakyReLU(0.05)

    def forward(self, x):        
        x_task = [None for i in range(len(self.output))]  # initialize
        for task in range(len(self.output)):
            x_task[task] = self.hidden_1[task](x)
            x_task[task] = self.batchnorm_1[task](x_task[task])
            x_task[task] = self.leakyReLU(x_task[task])
            
            x_task[task] = self.hidden_2[task](x_task[task])
            x_task[task] = self.batchnorm_2[task](x_task[task])
            x_task[task] = self.leakyReLU(x_task[task])
            
            x_task[task] = self.output[task](x_task[task])
            x_task[task] = torch.sigmoid(x_task[task])
        
        y_pred = x_task
        
        return y_pred
    
# ========================
# Load model weights
# ========================
model = DNN(X.shape[1], tox21_tasks)

model_path = f'deep_predictive_models/deep_learning/FP/STDNN/trained_models/STDNN_FP_tox21_seed124/best_model.pt'
checkpoint = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'], strict=True)

model.eval()

# ========================
# Single-task wrapper
# ========================
class SingleTaskModel(nn.Module):
    def __init__(self, base_model, task_idx):
        super().__init__()
        self.base_model = base_model
        self.task_idx = task_idx

    def forward(self, x):
        return self.base_model(x)[self.task_idx]

wrapped_model = SingleTaskModel(model, explain_task)

# ========================
# Gradient-based attribution
# ========================
print("Computing gradient-based attributions...")

X_tensor = torch.tensor(X, dtype=torch.float32)

attr = []

for i in range(X_tensor.shape[0]):
    x = X_tensor[i:i+1].clone().detach().requires_grad_(True)

    wrapped_model.zero_grad()

    output = wrapped_model(x).squeeze()
    output.backward()

    grad = x.grad.detach().cpu().numpy().flatten()
    attr.append(grad)

attr = np.array(attr)

print("Attributions shape:", attr.shape)

# ========================
# Aggregate importance
# ========================
mean_attr = np.mean(np.abs(attr), axis=0)

top_n = 10
important_bits = np.argsort(mean_attr)[::-1][:top_n]

# ========================
# Map bits → substructures
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
# Save results
# ========================
results_df = pd.DataFrame(results)
results_df.to_csv('neuralsens_tox21_feature_attributions.csv', index=False)

print("\nTop features:")
print(results_df[['bit', 'mean_abs_attr', 'example_smiles', 'bit_smiles']])