from long_discriminative_score import long_discriminative_score_metrics
from long_predictive_score import long_predictive_score_metrics
import pandas as pd
import numpy as np
import torch
from data_make import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def generate_plot(synth_data, ori_data):
    ori_data = np.asarray(ori_data)
    synth_data = synth_data.transpose(0, 2, 1)
    plt.plot(synth_data[0, :, 0], label='Synthétique', color='red')
    plt.plot(ori_data[0, :, 0], label='Original', color='blue')
    plt.legend()
    plt.savefig("./saved_files/fig.png")

_, original_data= LoadData("stock", 100)
synthetic_data = np.load("./data/synth-stock-other.npy")
generate_plot(synthetic_data, original_data)

print("min original data: ", np.min(original_data))
print("max original data: ", np.max(original_data))
print("mean original data: ", np.mean(original_data))
print("std original data: ", np.std(original_data))
print(long_discriminative_score_metrics(original_data, synthetic_data))
print(long_predictive_score_metrics(original_data, synthetic_data))
