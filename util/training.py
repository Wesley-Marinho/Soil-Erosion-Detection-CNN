import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    recall_score,
    confusion_matrix,
)
import seaborn as sns
from sklearn.model_selection import validation_curve


def train(
    model,
    training_generator,
    validation_generator,
    loss_func,
    learning_rate,
    epochs,
    model_validation,
    cuda_available,
    path_model,
    patience,
):
    # Implement Adam algorithm
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Decay the learning rate by gamma every step_size epochs.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

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

    recall_history_training = []
    recall_epoch_history_training = []
    recall_epoch_history_validation = []
    recall_history_validation = []

    confusion_matrix_history_training = []
    confusion_matrix_epoch_history_training = []
    confusion_matrix_epoch_history_validation = []
    confusion_matrix_history_validation = []

    best_val_loss = float("inf")
    patience_counter = 0

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
                average="weighted",
            )
            iou_train = jaccard_score(
                labels.cpu().flatten().int(),
                predicted.cpu().flatten(),
                average="weighted",
            )
            recall_train = recall_score(
                labels.cpu().flatten().int(),
                predicted.cpu().flatten(),
                average="weighted",
            )
            confusion_matrix_train = confusion_matrix(
                labels.cpu().flatten().int(),
                predicted.cpu().flatten(),
            )
            # Use 'weighted' para calcular o IOU
            f1_score_history_training.append(f1_train)
            iou_score_history_training.append(iou_train)
            recall_history_training.append(recall_train)
            confusion_matrix_history_training.append(confusion_matrix_train)

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

        recall_epoch_history_training.append(
            sum(recall_history_training) / len(recall_history_training)
        )

        confusion_matrix_epoch_history_training.append(
            sum(confusion_matrix_history_training)
            / len(confusion_matrix_history_training)
        )

        if model_validation:
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
                        average="weighted",
                    )
                    iou = jaccard_score(
                        labels.cpu().flatten().int(),
                        predicted.cpu().flatten(),
                        average="weighted",
                    )
                    recall = recall_score(
                        labels.cpu().flatten().int(),
                        predicted.cpu().flatten(),
                        average="weighted",
                    )
                    confusion_matrix_validation = confusion_matrix(
                        labels.cpu().flatten().int(),
                        predicted.cpu().flatten(),
                    )

                    # Use 'weighted' para calcular o IOU
                    f1_score_history.append(f1)
                    iou_score_history.append(iou)
                    acc_score_history.append(ACC)
                    recall_history_validation.append(recall)
                    confusion_matrix_history_validation.append(
                        confusion_matrix_validation
                    )
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
        recall_epoch_history_validation.append(
            sum(recall_history_validation) / len(recall_history_validation)
        )
        confusion_matrix_epoch_history_validation.append(
            sum(confusion_matrix_history_validation)
            / len(confusion_matrix_history_validation)
        )

        print(
            "Acurácia de validação: ", sum(acc_score_history) / len(acc_score_history)
        )
        print("F1-score de validação: ", sum(f1_score_history) / len(f1_score_history))
        print("IoU de validação: ", sum(iou_score_history) / len(iou_score_history))
        print(
            "Recall de validação: ",
            sum(recall_history_validation) / len(recall_history_validation),
        )
        # print(
        #    "Matriz de Confusão validação: ",
        #    sum(confusion_matrix_history_validation)
        #    / len(confusion_matrix_history_validation),
        # )
        scheduler.step()

        # Early stopping
        if val_loss_history[-1] < best_val_loss:
            best_val_loss = val_loss_history[-1]
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_func,
                },
                path_model,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        scheduler.step()

    tn_values = []
    fp_values = []
    fn_values = []
    tp_values = []

    # Iterar sobre a lista de matrizes de confusão e extrair os valores
    for cm in confusion_matrix_epoch_history_training:
        tn, fp, fn, tp = cm.ravel()  # Desembrulha os valores da matriz de confusão
        # print(tn)
        tn_values.append(tn)
        fp_values.append(fp)
        fn_values.append(fn)
        tp_values.append(tp)

    # Criando o gráfico de linha
    epochs = range(1, len(confusion_matrix_epoch_history_training) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, tn_values, label="True Negatives (TN)", marker="o")
    plt.plot(epochs, fp_values, label="False Positives (FP)", marker="o")
    plt.plot(epochs, fn_values, label="False Negatives (FN)", marker="o")
    plt.plot(epochs, tp_values, label="True Positives (TP)", marker="o")

    plt.xlabel("Epochs")
    plt.ylabel("Count")
    plt.title("Progressão de TN, FP, FN e TP ao Longo das Épocas")
    plt.legend()
    plt.grid(True)
    plt.show()

    tn_values = []
    fp_values = []
    fn_values = []
    tp_values = []

    # Iterar sobre a lista de matrizes de confusão e extrair os valores
    for cm in confusion_matrix_epoch_history_validation:
        tn, fp, fn, tp = cm.ravel()  # Desembrulha os valores da matriz de confusão
        # print(tn)
        tn_values.append(tn)
        fp_values.append(fp)
        fn_values.append(fn)
        tp_values.append(tp)

    # Criando o gráfico de linha
    epochs = range(1, len(confusion_matrix_epoch_history_validation) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, tn_values, label="True Negatives (TN)", marker="o")
    plt.plot(epochs, fp_values, label="False Positives (FP)", marker="o")
    plt.plot(epochs, fn_values, label="False Negatives (FN)", marker="o")
    plt.plot(epochs, tp_values, label="True Positives (TP)", marker="o")

    plt.xlabel("Epochs")
    plt.ylabel("Count")
    plt.title("Progressão de TN, FP, FN e TP ao Longo das Épocas")
    plt.legend()
    plt.grid(True)
    plt.show()

    confusion_mastrix_val = sum(confusion_matrix_history_validation) / len(
        confusion_matrix_history_validation
    )
    confusion_mastrix_train = sum(confusion_matrix_history_training) / len(
        confusion_matrix_history_training
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_mastrix_val,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predito Negativo", "Predito Positivo"],
        yticklabels=["Real Negativo", "Real Positivo"],
    )
    plt.title("Matriz de Confusão")
    plt.xlabel("Classe Predita")
    plt.ylabel("Classe Real")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_mastrix_train,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predito Negativo", "Predito Positivo"],
        yticklabels=["Real Negativo", "Real Positivo"],
    )
    plt.title("Matriz de Confusão")
    plt.xlabel("Classe Predita")
    plt.ylabel("Classe Real")
    plt.show()

    # Plot learning curves
    fig, axes = plt.subplots(
        3, 2, figsize=(12, 9)
    )  # Ajuste o tamanho conforme necessário

    # Loss (Perda)
    axes[0, 1].plot(train_loss_history, label="Train Loss", linestyle="--", marker="o")
    axes[0, 1].plot(
        val_loss_history, label="Validation Loss", linestyle="--", marker="o"
    )
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()
    axes[0, 1].grid()

    # Accuracy (Acurácia)
    axes[0, 0].plot(
        train_accuracy_history, label="Train Accuracy", linestyle="--", marker="o"
    )
    axes[0, 0].plot(
        val_accuracy_history, label="Validation Accuracy", linestyle="--", marker="o"
    )
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_ylim(0.50, 1)
    axes[0, 0].legend()
    axes[0, 0].grid()

    # F1 Score
    axes[1, 0].plot(
        f1_epoch_history_validation,
        label="Validation F1 Score",
        linestyle="--",
        marker="o",
    )
    axes[1, 0].plot(
        f1_epoch_history_training, label="Training F1 Score", linestyle="--", marker="o"
    )
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].legend()
    axes[1, 0].grid()

    # IOU Score
    axes[1, 1].plot(
        iou_epoch_history_validation,
        label="Validation IOU Score",
        linestyle="--",
        marker="o",
    )
    axes[1, 1].plot(
        iou_epoch_history_training,
        label="Training IOU Score",
        linestyle="--",
        marker="o",
    )
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].legend()
    axes[1, 1].grid()

    # Recall
    axes[2, 0].plot(
        recall_epoch_history_validation,
        label="Validation Recall",
        linestyle="--",
        marker="o",
    )
    axes[2, 0].plot(
        recall_epoch_history_training,
        label="Training Recall",
        linestyle="--",
        marker="o",
    )
    axes[2, 0].set_xlabel("Epoch")
    axes[2, 0].set_ylabel("Score")
    axes[2, 0].legend()
    axes[2, 0].grid()

    fig.delaxes(axes[2, 1])

    plt.tight_layout()
    plt.show()

    k_acc = sum(val_accuracy_history) / len(val_accuracy_history)
    K_f1 = sum(f1_epoch_history_validation) / len(f1_epoch_history_validation)
    k_iou = sum(iou_epoch_history_validation) / len(iou_epoch_history_validation)
    k_recall = sum(recall_history_validation) / len(recall_history_validation)

    return k_acc, K_f1, k_iou, k_recall
