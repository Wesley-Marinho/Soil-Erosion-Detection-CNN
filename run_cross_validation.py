# %%
import torch
from util.gpu_info import gpuInfo
from util.data_augmentation import (
    data_augmentation,
    training_data_loading,
    training_data_augmentation,
)
from util.loss import BCEIoULoss
from util.training import train, train_k_fold
from util.testing import test
from util.data_load import data_load, unify_augmented_data
import numpy as np

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

batch_size = 4

path_training = "./training/"
path_testing = "./test/"
path_data = "./data/"
path_model = "./models/"

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

gpu_info = gpuInfo()

model = LinkNet34()
if cuda_available:
    model.cuda()

print(model)
# %%
if training_data_processing:
    training_resize = 512
    data_augmentation(training_resize, path_training, path_data)
    # %%

k_acc_history = []
K_f1_history = []
k_iou_history = []
k_recall_history = []

for fold in range(1, 6):
    print(f"\nFold {fold}/{k}")

    output_dir = f"{path_data}{fold}/"

    images_augmented, labels_augmented = unify_augmented_data(output_dir)

    val_images = np.load(f"{path_data}{fold}/images_validation.npy")
    val_labels = np.load(f"{path_data}{fold}/labels_validation.npy")

    training_generator, validation_generator = data_load(
        train_images_augmented,
        train_labels_augmented,
        val_images,
        val_labels,
        model_validation,
        batch_size,
    )

    del train_images_augmented, train_labels_augmented, val_images, val_labels

    path_models = f"{path_model}/{fold}/LinkNet34.model"

    k_acc, K_f1, k_iou, k_recall = train_k_fold(
        model,
        training_generator,
        validation_generator,
        loss_func=BCEIoULoss(),
        learning_rate=2e-4,
        epochs=2,
        model_validation=model_validation,
        cuda_available=cuda_available,
        path_model=path_models,
        patience=5,
    )

    k_acc_history.append(k_acc)
    K_f1_history.append(K_f1)
    k_iou_history.append(k_iou)
    k_recall_history.append(k_recall)

    for aux in range(747, 753):
        test(path_testing, aux, model, cuda_available)

print("Média da Acurácia: ", {k_acc_history.mean(): 0.4})
print("Desvio padrão da Acurácia: ", {k_acc_history.std(): 0.4})
print("Média do f1-score: ", {k_f1_history.mean(): 0.4})
print("Desvio padrão do f1-score: ", {k_f1_history.std(): 0.4})
print("Média do Iou: ", {k_iou_history.mean(): 0.4})
print("Desvio padrão do Iou: ", {k_iou_history.std(): 0.4})
print("Média do Recall: ", {k_recall_history.mean(): 0.4})
print("Desvio padrão do Recall: ", {k_recall_history.std(): 0.4})

# %%
if model_loading:
    checkpoint = torch.load(path_model)
    model.load_state_dict(checkpoint["model_state_dict"])
    for aux in range(747, 753):
        test(path_testing, aux, model, cuda_available)
