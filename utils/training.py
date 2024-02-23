import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, jaccard_score


def train(
    model,
    images_training,
    labels_training,
    images_validation,
    labels_validation,
    loss_func,
    batch_size,
    learning_rate,
    epochs,
    model_validation,
    cuda_available,
    path_model,
):
    """
    train - Train the instance of the neural network.
    Args:
        model (torch): the instance of the neural network
        images_training, labels_training (numpy): the augmented training dataset
        images_validation, labels_validation (numpy): the resized validation dataset
        loss_func (class): the loss function
        batch_size (int): the number of samples per batch to load (default: 8)
        learning_rate (float): the learning rate (default: 1e-3)
        epochs (int): the learning epochs (default: 80)
        if_validation (bool): the flag indicating whether or not to implement validation (default: False)
        cuda_available (bool): the flag indicating whether CUDA is available (default: True)
    """
    # Use torch.utils.data to create a training_generator
    images_training = torch.Tensor(images_training)
    labels_training = torch.Tensor(labels_training)
    training_set = TensorDataset(images_training, labels_training)
    training_generator = DataLoader(training_set, batch_size=batch_size, shuffle=True)

    # Use torch.utils.data to create a validation_generator
    if model_validation and len(images_validation) > 0:
        images_validation = torch.Tensor(images_validation)
        labels_validation = torch.Tensor(labels_validation)
        validation_set = TensorDataset(images_validation, labels_validation)
        validation_generator = DataLoader(
            validation_set, batch_size=batch_size, shuffle=True
        )

    # Implement Adam algorithm
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Decay the learning rate by gamma every step_size epochs.
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5, verbose=True
    )

    # Lists to store training and validation metrics
    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []
    f1_score_history = []
    f1_score_history_training = []
    f1_epoch_history_training = []
    f1_epoch_history_validation = []
    iou_score_history = []
    iou_epoch_history_training = []
    iou_score_history_training = []
    iou_epoch_history_validation = []
    acc_score_history = []
    # Loop over epochs
    for epoch in tqdm(range(epochs)):
        # Training
        print(f"\n---------Training for Epoch {epoch + 1} starting:---------")
        model.train()
        loss_training = 0
        accuracy_training = 0
        cont = 0
        accuracy_validation = 0
        # Loop over batches in an epoch using training_generator
        for index, (inputs, labels) in enumerate(training_generator):
            if cuda_available:
                # Transfer to GPU
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_training += loss

            predicted = (outputs > 0.5).float()
            correct = (predicted == labels).float()
            accuracy = correct.sum() / correct.numel()

            accuracy_training += accuracy
            cont += 1

            ACC = accuracy_score(
                labels.cpu().flatten().int(), predicted.cpu().flatten()
            )
            f1_train = f1_score(
                labels.cpu().flatten().int(),
                predicted.cpu().flatten(),
                average="binary",
            )
            iou_train = jaccard_score(
                labels.cpu().flatten().int(),
                predicted.cpu().flatten(),
                average="weighted",
            )
            # Use 'weighted' para calcular o IOU
            f1_score_history_training.append(f1_train)
            iou_score_history_training.append(iou_train)

            if index % 20 == 0:
                loss_item = loss.item()
                print(f"→ Running_loss for Batch {index + 1}: {loss_item}")
                print(f"→ ACC for Batch {index + 1}: {accuracy}")

        accuracy_training_final = accuracy_training / cont
        loss_training_final = loss_training / cont
        print(f"\033[1mTraining loss for Epoch {epoch + 1}: {loss_training_final}")
        print(f"\033[1mTraining ACC for Epoch {epoch + 1}: {accuracy_training_final}")
        f1_epoch_history_training.append(
            sum(f1_score_history_training) / len(f1_score_history_training)
        )
        iou_epoch_history_training.append(
            sum(iou_score_history_training) / len(iou_score_history_training)
        )
        # Append epoch training loss and accuracy to the history lists
        train_loss_history.append(loss_training.item() / cont)
        train_accuracy_history.append(accuracy_training.item() / cont)

        if model_validation and len(images_validation) > 0:
            # Validation
            print(f"--------Validation for Epoch {epoch + 1} starting:--------")
            model.eval()
            with torch.no_grad():
                loss_validation = 0

                cont1 = 0
                # Loop over batches in an epoch using validation_generator
                for index, (inputs, labels) in enumerate(validation_generator):
                    if cuda_available:
                        # Transfer to GPU
                        inputs, labels = inputs.cuda(), labels.cuda()

                    outputs = model(inputs)
                    loss_validation += loss_func(outputs, labels)

                    predicted = (outputs > 0.5).float()
                    correct = (predicted == labels).float()
                    accuracy = correct.sum() / correct.numel()

                    cont1 += 1
                    accuracy_validation += accuracy

                    #  Calculate F1-score and IOU
                    ACC = accuracy_score(
                        labels.cpu().flatten().int(), predicted.cpu().flatten()
                    )
                    f1 = f1_score(
                        labels.cpu().flatten().int(),
                        predicted.cpu().flatten(),
                        average="binary",
                    )
                    iou = jaccard_score(
                        labels.cpu().flatten().int(),
                        predicted.cpu().flatten(),
                        average="weighted",
                    )
                    # Use 'weighted' para calcular o IOU
                    f1_score_history.append(f1)
                    iou_score_history.append(iou)
                    acc_score_history.append(ACC)
            # accuracy_validation_final=accuracy_validation/cont1
            loss_validation_final = loss_validation / cont1
            print(
                f"\033[1mValidation loss for Epoch {epoch + 1}: {loss_validation_final}\033[0m\n"
            )

        val_loss_history.append(loss_validation.item() / cont1)
        val_accuracy_history.append(accuracy_validation.item() / cont1)
        f1_epoch_history_validation.append(
            sum(f1_score_history) / len(f1_score_history)
        )
        iou_epoch_history_validation.append(
            sum(iou_score_history) / len(iou_score_history)
        )
        print("Acurácia de validação:", sum(acc_score_history) / len(acc_score_history))
        print("F1-score de validação:", sum(f1_score_history) / len(f1_score_history))
        print("IoU de validação:", sum(iou_score_history) / len(iou_score_history))
        scheduler.step()

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss_func,
            },
            path_model + "model.model",
        )

    # Plot learning curves
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 2)
    plt.plot(train_loss_history, label="Train Loss", linestyle="--", marker="o")
    plt.plot(val_loss_history, label="Validation Loss", linestyle="--", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 1)
    plt.plot(train_accuracy_history, label="Train Accuracy", linestyle="--", marker="o")
    plt.plot(
        val_accuracy_history, label="Validation Accuracy", linestyle="--", marker="o"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.50, 1)
    plt.legend()
    plt.grid()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(
        f1_epoch_history_validation,
        label="Validaton F1 Score",
        linestyle="--",
        marker="o",
    )
    plt.plot(
        f1_epoch_history_training, label="Training F1 Score", linestyle="--", marker="o"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(
        iou_epoch_history_validation,
        label="Validation IOU Score",
        linestyle="--",
        marker="o",
    )
    plt.plot(
        iou_epoch_history_training,
        label="Training IOU Score",
        linestyle="--",
        marker="o",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid()
