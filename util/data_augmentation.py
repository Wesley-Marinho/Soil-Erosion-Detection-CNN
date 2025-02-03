import numpy as np
from PIL import Image
from skimage.transform import resize
import scipy
from tqdm import tqdm
import os
from sklearn.model_selection import KFold


def training_data_loading(path_training, training_resize):
    images_loading = np.empty(shape=(751, 3, training_resize, training_resize))
    labels_loading = np.empty(shape=(751, 1, training_resize, training_resize))

    image_files = sorted(os.listdir(f"{path_training}images/"))
    label_files = sorted(os.listdir(f"{path_training}groundtruth/"))

    for i, (img_file, lbl_file) in tqdm(
        enumerate(zip(image_files, label_files)), total=len(image_files)
    ):
        image = (
            np.array(Image.open(f"{path_training}images/{img_file}")).astype(float)
            / 255
        )
        label = (
            np.array(Image.open(f"{path_training}groundtruth/{lbl_file}")).astype(float)
            / 255
        )

        label = np.expand_dims(label, 2)
        image = resize(image, (training_resize, training_resize), anti_aliasing=True)
        label = resize(label, (training_resize, training_resize), anti_aliasing=True)

        image = np.transpose(image, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))

        images_loading[i] = image
        labels_loading[i] = label

    return images_loading, labels_loading


def training_data_augmentation(
    images_training,
    labels_training,
    rotations,
    flips,
    shifts,
    training_resize,
    output_dir,
    batch_size=100,
):
    """
    training_data_augmentation - Generate the augmented training dataset and save it to disk in batches.
    Args:
        images_training, labels_training (numpy): the resized training dataset
        rotations (list): the parameters for rotating resized training images and their corresponding masks (training pairs)
        flips (list): the parameters for flipping rotated training pairs
        shifts (list): the parameters for shifting flipped training pairs
        training_resize (int): the resolution of resized training pairs (default: 384)
        batch_size (int): number of augmented samples to process and save at a time
        output_dir (str): directory to save augmented data
    Returns:
        None (saves augmented data to disk)
    """
    num_rota = len(rotations)
    num_flip = len(flips)
    num_shft = len(shifts)

    # Calculate total number of augmented samples
    num_training = images_training.shape[0]
    num_augmented = num_training * num_rota * num_flip * num_shft

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize batch storage
    batch_images = []
    batch_labels = []
    batch_counter = 0
    file_counter = 0
    print(num_training)
    for index in tqdm(range(num_training)):

        image = np.transpose(images_training[index], (1, 2, 0))
        label = np.transpose(labels_training[index], (1, 2, 0))

        for rota in rotations:
            # Rotate a resized training pair
            image_rota = scipy.ndimage.rotate(
                image, rota, reshape=False, mode="reflect"
            )
            label_rota = scipy.ndimage.rotate(
                label, rota, reshape=False, mode="reflect"
            )

            for flip in flips:
                # Flip the rotated training pair
                if flip == "original":
                    image_flip = image_rota
                    label_flip = label_rota
                else:
                    image_flip = flip(image_rota)
                    label_flip = flip(label_rota)

                for shft in shifts:
                    # Shift the flipped training pair
                    shft_H = np.random.uniform(low=shft[0], high=shft[1], size=1)[0]
                    shft_W = np.random.uniform(low=shft[0], high=shft[1], size=1)[0]
                    image_shft = scipy.ndimage.shift(
                        image_flip, (shft_H, shft_W, 0), mode="reflect"
                    )
                    label_shft = scipy.ndimage.shift(
                        label_flip, (shft_H, shft_W, 0), mode="reflect"
                    )

                    # Append the augmented image and label to the batch
                    batch_images.append(
                        np.clip(np.transpose(image_shft, (2, 0, 1)), 0, 1)
                    )
                    batch_labels.append((np.transpose(label_shft, (2, 0, 1)) > 0.3))
                batch_counter += 1

                # Save batch to disk if it reaches the batch size
        if batch_counter >= batch_size:
            save_batch(batch_images, batch_labels, output_dir, file_counter)
            file_counter += 1
            batch_images = []
            batch_labels = []
            batch_counter = 0

        # Clear memory for the current image and label
        del (
            image,
            label,
            image_rota,
            label_rota,
            image_flip,
            label_flip,
            image_shft,
            label_shft,
        )

    # Save any remaining data in the last batch
    if batch_counter > 0:
        save_batch(batch_images, batch_labels, output_dir, file_counter)

    print(f"Augmented data saved to {output_dir}")


def save_batch(batch_images, batch_labels, output_dir, file_counter):
    """
    Save a batch of augmented images and labels to disk.
    """
    batch_images = np.array(batch_images)
    batch_labels = np.array(batch_labels)

    np.save(
        os.path.join(output_dir, f"images_augmented_{file_counter}.npy"), batch_images
    )
    np.save(
        os.path.join(output_dir, f"labels_augmented_{file_counter}.npy"), batch_labels
    )

    # Clear memory
    del batch_images, batch_labels


def data_augmentation(training_resize, path_training, path_data):
    images_loading, labels_loading = training_data_loading(
        path_training, training_resize
    )

    k = 5
    rotations = [0, 45, 90, 135]
    flips = ["original", np.flipud, np.fliplr]
    shifts = [(-16, 16)]
    kf = KFold(n_splits=k, shuffle=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(images_loading)):
        folds = fold + 1
        print(f"\nFold {folds}/{k}")

        train_images, train_labels = (
            images_loading[train_idx],
            labels_loading[train_idx],
        )

        val_images, val_labels = images_loading[val_idx], labels_loading[val_idx]

        np.save(f"{path_data}{folds}/images_validation", val_images)
        np.save(f"{path_data}{folds}/labels_validation", val_images)
        del val_images, val_labels

        output_dir = f"{path_data}{folds}/"

        training_data_augmentation(
            train_images,
            train_labels,
            rotations,
            flips,
            shifts,
            training_resize,
            output_dir,
        )
        print(f"\n Fim do Processamento {folds}")
