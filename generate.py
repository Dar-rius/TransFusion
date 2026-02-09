import torch from ddpm import * import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_make import *
from sklearn.preprocessing import MinMaxScaler

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OBJECTIVE = "pred_v"
TIMESTEPS = 1000
SEQ_LEN = 100
SEQ_DAY = 250
BETA_SCHEDULE = "cosine"
data = pd.read_csv("./data/metric_market.csv")
model_pth = "./custom-transformers-stock-l1-cosine-100-pred_v-final.pth"

train_data, _ = LoadData("stock", SEQ_LEN)
train_data = np.asarray(train_data)
N, L, C = train_data.shape
train_reshaped = train_data.reshape(-1, C)

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(train_reshaped)

model = TransEncoder(
        features = C,
        latent_dim = 128,
        num_heads = 4,
        num_layers = 4
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = SEQ_LEN,
    timesteps = TIMESTEPS,
    objective = OBJECTIVE,
    loss_type = 'l2',
    beta_schedule = BETA_SCHEDULE
).to(DEVICE)

checkpoint = torch.load(model_pth, map_location=DEVICE)
diffusion.load_state_dict(checkpoint['diffusion_state_dict'])
with torch.no_grad():
    samples = diffusion.sample(batch_size=SEQ_DAY)
    samples = samples.cpu().numpy()

samples_reshaped = samples.transpose(0, 2, 1).reshape(-1, C)
samples_inverse = scaler.inverse_transform(samples_reshaped)
data_generated = samples_inverse.reshape(SEQ_DAY, SEQ_LEN, C)
np.save(f'./data/sample.npy', data_generated)


plt.figure(figsize=(10, 6))
sample_idx = 0
# On suppose que la colonne 0 est le Prix (adapté selon vos données)
plt.plot(data_generated[sample_idx, :, 0], label='Synthetic BTC Price', color='red')
plt.title("Exemple de Donnée Générée (Dénormalisée)")
plt.legend()
plt.grid(True)
plt.savefig("./data/sample_generated.png")
