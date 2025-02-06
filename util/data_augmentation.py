import numpy as np
from PIL import Image
from skimage.transform import resize
import scipy
from tqdm import tqdm
from sklearn.model_selection import KFold
import os
import numpy as np
import scipy.ndimage


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
    path_data,
):
    """
    training_data_augmentation - Generate the augmented training dataset.
    Args:
        images_training, labels_training (numpy): the resized training dataset
        rotations (list): the parameters for rotating resized training images and their corresponding masks (training pairs)
        flips (list): the parameters for flipping rotated training pairs
        shifts (list): the parameters for shifting flipped training pairs
        training_resize (int): the resolution of resized training pairs (default: 384)
    Returns:
        images_augmented, labels_augmented (numpy): the augmented training dataset
    """
    num_rota = len(rotations)
    num_flip = len(flips)
    num_shft = len(shifts)

    # Calculate the total number of augmented images
    num_training = images_training.shape[0]
    num_augmented = num_training * num_rota * num_flip * num_shft

    # Pre-allocate arrays for augmented data
    images_augmented = np.empty(
        (num_augmented, 3, training_resize, training_resize), dtype=np.float32
    )
    labels_augmented = np.empty(
        (num_augmented, 1, training_resize, training_resize), dtype=np.uint8
    )

    print(f"images_augmented.shape = {images_augmented.shape}")
    print(f"labels_augmented.shape = {labels_augmented.shape}")

    counter = 0
    for index in tqdm(range(num_training)):
        image = np.transpose(images_training[index], (1, 2, 0)).astype(np.float32)
        label = np.transpose(labels_training[index], (1, 2, 0)).astype(np.float32)

        for rota in rotations:
            # Rotate the image and label
            image_rota = scipy.ndimage.rotate(
                image, rota, reshape=False, mode="reflect"
            )
            label_rota = scipy.ndimage.rotate(
                label, rota, reshape=False, mode="reflect"
            )

            for flip in flips:
                # Flip the rotated image and label
                if flip == "original":
                    image_flip = image_rota
                    label_flip = label_rota
                else:
                    image_flip = flip(image_rota)
                    label_flip = flip(label_rota)

                for shft in shifts:
                    # Shift the flipped image and label
                    shft_H = np.random.uniform(low=shft[0], high=shft[1], size=1)[0]
                    shft_W = np.random.uniform(low=shft[0], high=shft[1], size=1)[0]
                    image_shft = scipy.ndimage.shift(
                        image_flip, (shft_H, shft_W, 0), mode="reflect"
                    )
                    label_shft = scipy.ndimage.shift(
                        label_flip, (shft_H, shft_W, 0), mode="reflect"
                    )

                    # Store the augmented data directly in the pre-allocated arrays
                    images_augmented[counter] = np.clip(
                        np.transpose(image_shft, (2, 0, 1)), 0, 1
                    )
                    labels_augmented[counter] = (
                        np.transpose(label_shft, (2, 0, 1)) > 0.3
                    ).astype(np.uint8)
                    counter += 1

    np.save(f"{path_data}images_training", images_augmented)
    del images_augmented
    np.save(f"{path_data}labels_training", labels_augmented)
    del labels_augmented


def database_construction(training_resize, path_training, path_data):
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
        np.save(f"{path_data}{folds}/labels_validation", val_labels)
        np.save(f"{path_data}{folds}/images_validation", val_images)

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
