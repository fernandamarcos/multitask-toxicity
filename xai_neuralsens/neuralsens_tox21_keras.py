import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import DataStructs
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set seeds for reproducibility
seed_value = 122
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# Settings
morgan_bits = 4096
morgan_radius = 2
explain_task = 0

tox21_file = 'data/datasets/tox21/raw_data/tox21.csv'
tox21_tasks = ['NR-AR', 'NR-Aromatase', 'NR-PPAR-gamma', 'SR-HSE',
               'NR-AR-LBD', 'NR-ER', 'SR-ARE', 'SR-MMP',
               'NR-AhR', 'NR-ER-LBD', 'SR-ATAD5', 'SR-p53']
task = tox21_tasks[explain_task]

# Load data
raw_data = pd.read_csv(tox21_file)
raw_data = raw_data.dropna(subset=['smiles'])
raw_data['mol'] = raw_data['smiles'].apply(Chem.MolFromSmiles)

# Compute Morgan fingerprints
bi = [{} for _ in range(len(raw_data))]
raw_data['morgan'] = [
    AllChem.GetMorganFingerprintAsBitVect(
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
Y = raw_data[tox21_tasks].astype(int).values
y_task = Y[:, explain_task]

# Build Keras model (single-task for NeuralSens)
def build_keras_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(1024, activation=keras.layers.LeakyReLU(0.05)),
        layers.BatchNormalization(),
        layers.Dense(512, activation=keras.layers.LeakyReLU(0.05)),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

input_shape = X.shape[1]
model = build_keras_model(input_shape)

# Train model (uncomment to train)
model.fit(X, y_task, epochs=10, batch_size=128, validation_split=0.1)

