# %%
import torch
from util.gpu_info import gpuInfo
from util.data_augmentation import database_construction
from util.loss import BCEIoULoss
from util.training import train
from util.testing import test
from util.data_load import data_load
import numpy as np
import os

from networks.LinkNetB7 import *
from networks.DLinkNet34 import *
from networks.DLinkNet50 import *
from networks.DLinkNet101 import *
from networks.LinkNet34 import *
from networks.UNet import *
from networks.DLinkNet152 import *
from networks.LinkNet152 import *

training_data_processing = True
model_training = True
model_validation = True
model_loading = False
data_loading = False

batch_size = 10

path_training = "./training/"
path_testing = "./test/"
path_data = "./data/"
path_model = "./models/LinkNet34.model"

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

gpu_info = gpuInfo()

model = LinkNet34()
if cuda_available:
    model.cuda()

print(model)

num_parametros = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Número total de parâmetros treináveis: {num_parametros}")
# %%
if training_data_processing:
    training_resize = 512
    # Load and generate the resized training dataset and validation dataset
    database_construction(training_resize, path_training, path_data)

# %%

k_acc_history = []
K_f1_history = []
k_iou_history = []
k_recall_history = []
k = 6

for fold in range(1, k):
    print(f"\nFold {fold}/{k-1}")

    output_dir = f"{path_data}{fold}/"

    training_generator, validation_generator = data_load(
        output_dir, model_validation, batch_size
    )

    k_acc, K_f1, k_iou, k_recall = train(
        model,
        training_generator,
        validation_generator,
        loss_func=BCEIoULoss(),
        learning_rate=2e-4,
        epochs=100,
        model_validation=model_validation,
        cuda_available=cuda_available,
        path_model=path_model,
        patience=5,
    )

    k_acc_history.append(k_acc)
    K_f1_history.append(K_f1)
    k_iou_history.append(k_iou)
    k_recall_history.append(k_recall)

    del training_generator, validation_generator

    for aux in range(747, 753):
        test(path_testing, aux, model, cuda_available)

k_acc_history = np.array(k_acc_history)
K_f1_history = np.array(K_f1_history)
k_iou_history = np.array(k_iou_history)
k_recall_history = np.array(k_recall_history)

print(f"Média da Acurácia: {k_acc_history.mean():.4f}")
print(f"Desvio padrão da Acurácia: {k_acc_history.std():.4f}")
print(f"Média do f1-score: {K_f1_history.mean():.4f}")
print(f"Desvio padrão do f1-score: {K_f1_history.std():.4f}")
print(f"Média do IoU: {k_iou_history.mean():.4f}")
print(f"Desvio padrão do IoU: {k_iou_history.std():.4f}")
print(f"Média do Recall: {k_recall_history.mean():.4f}")
print(f"Desvio padrão do Recall: {k_recall_history.std():.4f}")

# %%
if model_loading:
    checkpoint = torch.load(path_model)
    model.load_state_dict(checkpoint["model_state_dict"])
    for aux in range(747, 753):
        test(path_testing, aux, model, cuda_available)
