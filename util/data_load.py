import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def data_load(path_data, model_validation, batch_size):
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
    return training_generator, validation_generator
