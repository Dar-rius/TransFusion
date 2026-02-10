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
    plt.plot(synth_data[0, :, 0], label='Synthétique', color='red')
    plt.plot(ori_data[0, :, 0], label='Original', color='blue')
    plt.legend()
    plt.savefig("./saved_files/fig.png")

_, original_data = LoadData("metric_market.csv", 100)
ori_data = np.asarray(original_data)
synth_data = np.load("./data/synth-4500.npy")
generate_plot(synth_data, ori_data)

print(long_discriminative_score_metrics(ori_data, synth_data))
print(long_predictive_score_metrics(ori_data, synth_data))
