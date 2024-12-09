# %%
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from util.cuda import cuda
from util.gpu_info import gpuInfo
from util.data_augmentation import training_data_loading, training_data_augmentation
from util.loss import BCEIoULoss
from util.training import train
from util.testing import submission_creating, test
from networks.LinkNetB7 import *
from networks.DLinkNet34 import *
from networks.DLinkNet50 import *
from networks.DLinkNet101 import *
from networks.LinkNet34 import *
from networks.UNet import *

use_google_colab = False
training_data_processing = False
model_training = True
model_validation = True
model_loading = False

batch_size = 3

path_training = "./training/"
path_testing = "./test/"
path_data = "./data/"
path_model = "./models/UNet.model"

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

gpu_info = gpuInfo()

model = UNet()
if cuda_available:
    model.cuda()

print(model)
# %%

training_resize = 512
training_number = 367
testing_resize = int(608 * training_resize / 400)
if testing_resize % 2 == 1:
    testing_resize += 1

if training_data_processing:
    images_training, labels_training, images_validation, labels_validation = (
        training_data_loading(path_training, training_resize, training_number)
    )

    rotations = [0, 45, 90, 135]
    flips = ["original", np.flipud, np.fliplr]
    shifts = [(-16, 16)]

    images_augmented, labels_augmented = training_data_augmentation(
        images_training, labels_training, rotations, flips, shifts, training_resize
    )

    np.save(f"{path_data}images_training", images_augmented)
    np.save(f"{path_data}labels_training", labels_augmented)
    np.save(f"{path_data}images_validation", images_validation)
    np.save(f"{path_data}labels_validation", labels_validation)
    print("\n Fim do Processamento")

# %%

if not model_loading:
    images_augmented = np.load(f"{path_data}images_training.npy")
    labels_augmented = np.load(f"{path_data}labels_training.npy")

    images_augmented = torch.Tensor(images_augmented)
    labels_augmented = torch.Tensor(labels_augmented)

    training_set = TensorDataset(images_augmented, labels_augmented)
    del images_augmented, labels_augmented

    training_generator = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    del training_set

    if model_validation:
        images_validation = np.load(f"{path_data}images_validation.npy")
        labels_validation = np.load(f"{path_data}labels_validation.npy")
        images_validation = torch.Tensor(images_validation)
        labels_validation = torch.Tensor(labels_validation)

        validation_set = TensorDataset(images_validation, labels_validation)
        del images_validation, labels_validation

        validation_generator = DataLoader(
            validation_set, batch_size=batch_size, shuffle=True
        )
        del validation_set

    print("\n Fim do Carregamento")
# %%
if model_training:
    train(
        model,
        training_generator,
        validation_generator,
        loss_func=BCEIoULoss(),
        learning_rate=2e-4,
        epochs=100,
        model_validation=model_validation,
        cuda_available=cuda_available,
        path_model=path_model,
        patience=3,
    )

if model_loading:
    checkpoint = torch.load(path_model)
    model.load_state_dict(checkpoint["model_state_dict"])

for aux in range(747, 753):
    test(path_testing, aux, model, cuda_available)

# %%
