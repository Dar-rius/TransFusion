import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import json
import pathlib
import seaborn as sb
import argparse
import warnings
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter
from ddpm import *
from data_make import *

warnings.filterwarnings('ignore')

def visualize(ori_data, fake_data, seq_len, save_path, epoch, writer):
    ori_data = np.asarray(ori_data)
    fake_data = np.asarray(fake_data)
    ori_data = ori_data[:fake_data.shape[0]]
    sample_size = 250
    idx = np.random.permutation(len(ori_data))[:sample_size]
    randn_num = np.random.permutation(sample_size)[:1]
    real_sample = ori_data[idx]
    fake_sample = fake_data[idx]
    real_sample_2d = real_sample.reshape(-1, seq_len)
    fake_sample_2d = fake_sample.reshape(-1, seq_len)
    ### PCA
    pca = PCA(n_components=2)
    pca.fit(real_sample_2d)
    pca_real = (pd.DataFrame(pca.transform(real_sample_2d)).assign(Data='Real'))
    pca_synthetic = (pd.DataFrame(pca.transform(fake_sample_2d)).assign(Data='Synthetic'))
    pca_result = pd.concat([pca_real, pca_synthetic]).rename(columns={0: '1st Component', 1: '2nd Component'})
    ### TSNE
    tsne_data = np.concatenate((real_sample_2d, fake_sample_2d), axis=0)
    tsne = TSNE(n_components=2,verbose=0,perplexity=40)
    tsne_result = tsne.fit_transform(tsne_data)
    tsne_result = pd.DataFrame(tsne_result, columns=['X', 'Y']).assign(Data='Real')
    tsne_result.loc[len(real_sample_2d):, 'Data'] = 'Synthetic'
    fig, axs = plt.subplots(ncols = 2, nrows=2, figsize=(20, 20))
    sb.scatterplot(x='1st Component', y='2nd Component', data=pca_result,hue='Data', style='Data', ax=axs[0,0])
    sb.despine()
    axs[0,0].set_title('PCA Result')
    # plot
    sb.scatterplot(x='X', y='Y', data=tsne_result, hue='Data', style='Data', ax=axs[0,1])
    sb.despine()
    axs[0,1].set_title('t-SNE Result')
    axs[1,0].plot(real_sample[randn_num[0], :, :])
    axs[1,0].set_title('Original Data')
    axs[1,1].plot(fake_sample[randn_num[0], :, :])
    axs[1,1].set_title('Synthetic Data')
    fig.suptitle('Assessing Diversity: Qualitative Comparison of Real and Synthetic Data Distributions', fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=.88)
    plt.savefig(os.path.join(f'{save_path}', f'{time.time()}-tsne-result-{epoch}.png'))
    writer.add_figure('visualization', fig, epoch)

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")
    seq_len = args.seq_len
    epochs = args.training_epoch
    timesteps = args.timesteps
    batch_size = args.batch_size
    latent_dim = args.latent_dim
    num_layers = args.num_layers
    n_heads = args.n_head
    objective = args.objective
    model_name = args.model_name

    #Other Variable
    lr = 1e-4
    betas = (0.9, 0.99)
    dataset_name = "metric_market.csv"
    beta_schedule = "cosine"
    loss_type = "l2"

    #Transform data
    train_data, test_data = LoadData(dataset_name, seq_len)
    train_data, test_data = np.asarray(train_data), np.asarray(test_data)
    N_train, L, C = train_data.shape
    N_test, _, _ = test_data.shape
    train_reshaped = train_data.reshape(-1, C)
    test_reshaped = test_data.reshape(-1, C)
    # Init and fit scaler in train
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_reshaped_norm = scaler.fit_transform(train_reshaped)
    # Transform test data
    test_reshaped_norm = scaler.transform(test_reshaped)
    # reshape to (N, L, C)
    train_data = train_reshaped_norm.reshape(N_train, L, C)
    test_data = test_reshaped_norm.reshape(N_test, L, C)
    # Transposed for (Batch, Channels, Length) -> (N, C, L)
    # Channels refers to features of dataset
    train_data, test_data = train_data.transpose(0,2,1), test_data.transpose(0,2,1)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, len(test_data))

    # Note: scaler norm for real dataset
    real_data = next(iter(test_loader))
    file_name = f'{model_name}-{seq_len}'
    folder_name = f'saved_files/{time.time():.4f}-{file_name}'
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)
    output = f'{folder_name}/output'
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)
    with open(f'{folder_name}/params.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    writer = SummaryWriter(log_dir = folder_name, comment = f'{file_name}-output', flush_secs = 45)

    #Init encoder model
    model = TransEncoder(
        features = C,
        latent_dim = latent_dim,
        num_heads = n_heads,
        num_layers = num_layers
    ).to(device=device)

    # Init the diffusion model
    diffusion = GaussianDiffusion1D(
        model,
        seq_length = seq_len,
        timesteps = timesteps,
        objective = objective, # pred_x0, pred_v
        loss_type = loss_type,
        beta_schedule = beta_schedule
    ).to(device=device)

    optim = torch.optim.Adam(diffusion.parameters(), lr = lr, betas = betas)
    for running_epoch in tqdm(range(epochs)):
        for i, data in enumerate(train_loader):
            data = data.to(device=device)
            optim.zero_grad()
            loss = diffusion(data)
            loss.backward()
            optim.step()
            if i % len(train_loader) == 0:
                writer.add_scalar('Loss', loss.item(), running_epoch)
                
            if i % len(train_loader) == 0 and running_epoch % 100 == 0:
                print(f'Epoch: {running_epoch+1}, Loss: {loss.item()}')
                
            # Save and Visualize
            if i % len(train_loader) == 0 and running_epoch % 500 == 0:
                with torch.no_grad():
                    samples = diffusion.sample(len(test_data))
                    samples = samples.cpu().numpy()
                    # Transpose data to (N, L, C)
                    samples_reshaped = samples.transpose(0, 2, 1).reshape(-1, C)
                    real_data_reshaped = real_data.cpu().numpy().transpose(0, 2, 1).reshape(-1, C)
                    # Reverse the data normalised
                    samples_inverse = scaler.inverse_transform(samples_reshaped)
                    print(samples_reshaped[:,1])
                    real_inverse = scaler.inverse_transform(real_data_reshaped)
                    #  Reshape data to (N, C, L)
                    samples_to_save = samples_inverse.reshape(len(test_data), seq_len, C).transpose(0, 2, 1)
                    real_to_plot = real_inverse.reshape(len(test_data), seq_len, C).transpose(0, 2, 1)
                    # Save data
                    np.save(f'./{folder_name}/synth-{running_epoch}.npy', samples_to_save)
                    visualize(real_to_plot, samples_to_save, seq_len, output, running_epoch, writer)

    torch.save({'diffusion_state_dict': diffusion.state_dict(),
        'diffusion_optim_state_dict': optim.state_dict()},
        os.path.join(f'model_saved', f'{model_name}-final.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--objective',
        choices=['pred_x0','pred_v', 'pred_noise'],
        default='pred_v',
        type=str)
    
    parser.add_argument(
        '--seq_len',
        help='sequence length',
        default=100,
        type=int)
    
    parser.add_argument(
        '--batch_size',
        help='batch size for the network',
        default=256,
        type=int)
    
    parser.add_argument(
        '--n_head',
        help='number of heads for the attention',
        default=4,
        type=int)
    
    parser.add_argument(
        '--latent_dim',
        help='number of hidden state',
        default=128,
        type=int)
    
    parser.add_argument(
        '--num_layers',
        help='Number of Layers',
        default=4,
        type=int)
    
    parser.add_argument(
        '--training_epoch',
        help='Diffusion Training Epoch',
        default=3500,
        type=int)

    parser.add_argument(
        '--timesteps',
        help='Timesteps for Diffusion',
        default=1000,
        type=int)

    parser.add_argument(
        '--model_name',
        help='Model Name',
        default="Diffusion",
        type=str)
    
    args = parser.parse_args() 
    main(args)
