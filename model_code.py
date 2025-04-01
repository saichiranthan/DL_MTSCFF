import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
import numpy as np
from scipy.stats import kendalltau
import pywt
from sklearn.preprocessing import StandardScaler
import pandas as pd

class ImprovedGIN(nn.Module):
    def __init__(self, in_dim=6, hidden_dim=64, num_classes=4):
        super().__init__()
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ))
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, num_classes)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        batch = torch.arange(x.size(0), device=x.device)
        x = global_mean_pool(x, batch)
        return self.fc(x)

def parse_multivariate_ts_fixed(file_content):
    time_series_list = []
    labels = []
    for line in file_content:
        if line.startswith('#') or line.startswith('@'):
            continue
        parts = line.strip().split(":")
        series = [list(map(float, ts.split(","))) for ts in parts[:-1]]
        label = parts[-1].strip()
        time_series_list.append(np.array(series))
        labels.append(label)
    return np.array(time_series_list, dtype=object), np.array(labels)

def compute_cwt(data, scales=np.arange(1, 31), wavelet='morl'):
    cwt_output = np.zeros((data.shape[0], data.shape[1], len(scales), data.shape[2]))
    for sample_idx in range(data.shape[0]):
        for feature_idx in range(data.shape[1]):
            coef, _ = pywt.cwt(data[sample_idx, feature_idx, :], scales, wavelet)
            cwt_output[sample_idx, feature_idx, :, :] = coef
    return cwt_output

def compute_kendall_correlation(data):
    num_features = data.shape[1]
    correlation_matrix = np.zeros((num_features, num_features))
    for i in range(num_features):
        for j in range(num_features):
            if i != j:
                tau, _ = kendalltau(data[:, i, :].flatten(), data[:, j, :].flatten())
                correlation_matrix[i, j] = tau
            else:
                correlation_matrix[i, j] = 1
    return correlation_matrix

def print_model_parameters(model):
    param_info = []
    for name, param in model.named_parameters():
        param_info.append({
            "Layer": name,
            "Shape": str(tuple(param.shape)),
            "Trainable": param.requires_grad,
            "Mean": f"{param.data.mean().item():.6f}",
            "Std": f"{param.data.std().item():.6f}",
            "Min": f"{param.data.min().item():.6f}",
            "Max": f"{param.data.max().item():.6f}"
        })
    
    bn_info = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            bn_info.append({
                "BN Layer": name,
                "Running Mean": f"{module.running_mean.mean().item():.6f}",
                "Running Var": f"{module.running_var.mean().item():.6f}",
                "Features": module.num_features
            })
    
    return pd.DataFrame(param_info), pd.DataFrame(bn_info)