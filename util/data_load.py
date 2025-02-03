import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def data_load(
    train_images_augmented,
    train_labels_augmented,
    val_images,
    val_labels,
    model_validation,
    batch_size,
):

    images_augmented = torch.Tensor(train_images_augmented)
    labels_augmented = torch.Tensor(train_labels_augmented)
    print("\n Fim do Carregamento Training Data")
    training_set = TensorDataset(images_augmented, labels_augmented)
    del images_augmented, labels_augmented

    training_generator = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    del training_set

    if model_validation:
        images_validation = torch.Tensor(val_images)
        labels_validation = torch.Tensor(val_labels)
        print("\n Fim do Carregamento Validation Data")

        validation_set = TensorDataset(images_validation, labels_validation)
        del images_validation, labels_validation

        validation_generator = DataLoader(
            validation_set, batch_size=batch_size, shuffle=True
        )
        del validation_set
    print("\n Fim do Carregamento")
    return training_generator, validation_generator


def unify_augmented_data(output_dir):
    """
    Unify augmented data saved in batches into a single NumPy array.
    Args:
        output_dir (str): Directory where the augmented data is saved.
    Returns:
        images_augmented (numpy): Unified augmented images.
        labels_augmented (numpy): Unified augmented labels.
    """
    # List all image and label files
    image_files = sorted(
        [f for f in os.listdir(output_dir) if f.startswith("images_augmented_")]
    )
    label_files = sorted(
        [f for f in os.listdir(output_dir) if f.startswith("labels_augmented_")]
    )

    # Initialize lists to store loaded data
    images_augmented = []
    labels_augmented = []

    # Load and concatenate data
    for img_file, lbl_file in zip(image_files, label_files):
        images_augmented.append(np.load(os.path.join(output_dir, img_file)))
        labels_augmented.append(np.load(os.path.join(output_dir, lbl_file)))

    # Concatenate all batches into a single array
    images_augmented = np.concatenate(images_augmented, axis=0)
    labels_augmented = np.concatenate(labels_augmented, axis=0)

    print(f"Unified images_augmented.shape = {images_augmented.shape}")
    print(f"Unified labels_augmented.shape = {labels_augmented.shape}")

    return images_augmented, labels_augmented
