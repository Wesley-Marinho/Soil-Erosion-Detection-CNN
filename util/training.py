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
    balanced_accuracy_score,
)
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau


def configure_optimizers(model, learning_rate=1e-3, weight_decay=1e-4):
    """
    Configura o otimizador e o scheduler de learning rate.
    O ReduceLROnPlateau reduz a taxa de aprendizado quando a métrica de validação para de melhorar.
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=5)

    return optimizer, scheduler


def train(
    model,
    training_generator,
    validation_generator,
    loss_func,
    learning_rate,
    epochs,
    model_validation,
    cuda_available,
    model_save_path,
    patience,
):
    optimizer, scheduler = configure_optimizers(model, learning_rate)

    f1_score_history = []
    f1_score_training_history = []
    f1_epoch_training_history = []
    f1_epoch_validation_history = []

    iou_score_history = []
    iou_epoch_training_history = []
    iou_score_training_history = []
    iou_epoch_validation_history = []

    accuracy_score_training_history = []
    accuracy_score_validation_history = []
    accuracy_epoch_training_history = []
    accuracy_epoch_validation_history = []

    loss_score_training_history = []
    loss_score_validation_history = []
    loss_epoch_training_history = []
    loss_epoch_validation_history = []

    recall_training_history = []
    recall_epoch_training_history = []
    recall_epoch_validation_history = []
    recall_validation_history = []

    confusion_matrix_training_history = []
    confusion_matrix_epoch_training_history = []
    confusion_matrix_epoch_validation_history = []
    confusion_matrix_validation_history = []

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in tqdm(range(epochs)):
        print(f"\n--------- Training for Epoch {epoch + 1} starting: ---------")
        model.train()
        loss_training = 0
        accuracy_training = 0
        batch_count = 0
        accuracy_validation = 0

        for index, (inputs, labels) in enumerate(training_generator):
            if cuda_available:
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            predicted = (outputs > 0.5).float()
            correct = (predicted == labels).float()
            batch_count += 1

            accuracy = balanced_accuracy_score(
                labels.cpu().flatten().int(), predicted.cpu().flatten()
            )
            f1_train = f1_score(
                labels.cpu().flatten().int(),
                predicted.cpu().flatten(),
                average="macro",
            )
            iou_train = jaccard_score(
                labels.cpu().flatten().int(),
                predicted.cpu().flatten(),
                average="macro",
            )
            recall_train = recall_score(
                labels.cpu().flatten().int(),
                predicted.cpu().flatten(),
                average="macro",
            )
            confusion_matrix_train = confusion_matrix(
                labels.cpu().flatten().int(),
                predicted.cpu().flatten(),
            )

            accuracy_score_training_history.append(accuracy)
            f1_score_training_history.append(f1_train)
            iou_score_training_history.append(iou_train)
            recall_training_history.append(recall_train)
            confusion_matrix_training_history.append(confusion_matrix_train)
            loss_score_training_history.append(loss.item())

            if index % 20 == 0:
                print(f"→ Running loss for Batch {index + 1}: {loss.item()}")

        print(
            f"\033[1mTraining loss for Epoch {epoch + 1}: {sum(loss_score_training_history) / len(loss_score_training_history)}"
        )
        print(
            f"\033[1mTraining accuracy for Epoch {epoch + 1}: {sum(accuracy_score_training_history) / len(accuracy_score_training_history)}"
        )

        f1_epoch_training_history.append(
            sum(f1_score_training_history) / len(f1_score_training_history)
        )
        iou_epoch_training_history.append(
            sum(iou_score_training_history) / len(iou_score_training_history)
        )
        loss_epoch_training_history.append(
            sum(loss_score_training_history) / len(loss_score_training_history)
        )
        accuracy_epoch_training_history.append(
            sum(accuracy_score_training_history) / len(accuracy_score_training_history)
        )
        recall_epoch_training_history.append(
            sum(recall_training_history) / len(recall_training_history)
        )
        confusion_matrix_epoch_training_history.append(
            sum(confusion_matrix_training_history)
            / len(confusion_matrix_training_history)
        )

        if model_validation:
            print(f"-------- Validation for Epoch {epoch + 1} starting: --------")
            model.eval()
            with torch.no_grad():
                loss_validation = 0
                validation_batch_count = 0

                for index, (inputs, labels) in enumerate(validation_generator):
                    if cuda_available:
                        inputs, labels = inputs.cuda(), labels.cuda()

                    outputs = model(inputs)
                    loss_validation = loss_func(outputs, labels)

                    predicted = (outputs > 0.5).float()
                    correct = (predicted == labels).float()
                    accuracy = correct.sum() / correct.numel()

                    validation_batch_count += 1

                    accuracy_val = balanced_accuracy_score(
                        labels.cpu().flatten().int(), predicted.cpu().flatten()
                    )
                    f1_val = f1_score(
                        labels.cpu().flatten().int(),
                        predicted.cpu().flatten(),
                        average="macro",
                    )
                    iou_val = jaccard_score(
                        labels.cpu().flatten().int(),
                        predicted.cpu().flatten(),
                        average="macro",
                    )
                    recall_val = recall_score(
                        labels.cpu().flatten().int(),
                        predicted.cpu().flatten(),
                        average="macro",
                    )
                    confusion_matrix_val = confusion_matrix(
                        labels.cpu().flatten().int(),
                        predicted.cpu().flatten(),
                    )

                    f1_score_history.append(f1_val)
                    iou_score_history.append(iou_val)
                    accuracy_score_validation_history.append(accuracy_val)
                    recall_validation_history.append(recall_val)
                    confusion_matrix_validation_history.append(confusion_matrix_val)
                    loss_score_validation_history.append(loss_validation.item())

            print(
                f"\033[1mValidation loss for Epoch {epoch + 1}: {sum(loss_score_validation_history) / len(loss_score_validation_history)}\033[0m\n"
            )

        loss_epoch_validation_history.append(
            sum(loss_score_validation_history) / len(loss_score_validation_history)
        )
        accuracy_epoch_validation_history.append(
            sum(accuracy_score_validation_history)
            / len(accuracy_score_validation_history)
        )
        f1_epoch_validation_history.append(
            sum(f1_score_history) / len(f1_score_history)
        )
        iou_epoch_validation_history.append(
            sum(iou_score_history) / len(iou_score_history)
        )
        recall_epoch_validation_history.append(
            sum(recall_validation_history) / len(recall_validation_history)
        )
        confusion_matrix_epoch_validation_history.append(
            sum(confusion_matrix_validation_history)
            / len(confusion_matrix_validation_history)
        )

        print(
            "Validation accuracy: ",
            sum(accuracy_epoch_validation_history)
            / len(accuracy_epoch_validation_history),
        )
        print("Validation F1-score: ", sum(f1_score_history) / len(f1_score_history))
        print("Validation IoU: ", sum(iou_score_history) / len(iou_score_history))
        print(
            "Validation recall: ",
            sum(recall_validation_history) / len(recall_validation_history),
        )

        validation_loss_final = sum(loss_epoch_validation_history) / len(
            loss_epoch_validation_history
        )

        scheduler.step(validation_loss_final)

        if loss_epoch_validation_history[-1] < best_val_loss:
            best_val_loss = loss_epoch_validation_history[-1]
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_func,
                },
                model_save_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        scheduler.step(best_val_loss)

    tn_values = []
    fp_values = []
    fn_values = []
    tp_values = []

    for cm in confusion_matrix_epoch_training_history:
        tn, fp, fn, tp = cm.ravel()
        tn_values.append(tn)
        fp_values.append(fp)
        fn_values.append(fn)
        tp_values.append(tp)

    epochs = range(1, len(confusion_matrix_epoch_training_history) + 1)

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

    for cm in confusion_matrix_epoch_validation_history:
        tn, fp, fn, tp = cm.ravel()
        tn_values.append(tn)
        fp_values.append(fp)
        fn_values.append(fn)
        tp_values.append(tp)

    epochs = range(1, len(confusion_matrix_epoch_validation_history) + 1)

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

    confusion_matrix_val = sum(confusion_matrix_epoch_validation_history) / len(
        confusion_matrix_epoch_validation_history
    )
    confusion_matrix_train = sum(confusion_matrix_epoch_training_history) / len(
        confusion_matrix_epoch_training_history
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix_val,
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
        confusion_matrix_train,
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

    fig, axes = plt.subplots(3, 2, figsize=(12, 9))

    axes[0, 1].plot(
        loss_epoch_training_history, label="Train Loss", linestyle="--", marker="o"
    )
    axes[0, 1].plot(
        loss_epoch_validation_history,
        label="Validation Loss",
        linestyle="--",
        marker="o",
    )
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()
    axes[0, 1].grid()

    axes[0, 0].plot(
        accuracy_epoch_training_history,
        label="Train Accuracy",
        linestyle="--",
        marker="o",
    )
    axes[0, 0].plot(
        accuracy_epoch_validation_history,
        label="Validation Accuracy",
        linestyle="--",
        marker="o",
    )
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_ylim(0.50, 1)
    axes[0, 0].legend()
    axes[0, 0].grid()

    axes[1, 0].plot(
        f1_epoch_training_history, label="Training F1 Score", linestyle="--", marker="o"
    )
    axes[1, 0].plot(
        f1_epoch_validation_history,
        label="Validation F1 Score",
        linestyle="--",
        marker="o",
    )

    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].legend()
    axes[1, 0].grid()

    axes[1, 1].plot(
        iou_epoch_training_history,
        label="Training IOU Score",
        linestyle="--",
        marker="o",
    )

    axes[1, 1].plot(
        iou_epoch_validation_history,
        label="Validation IOU Score",
        linestyle="--",
        marker="o",
    )
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].legend()
    axes[1, 1].grid()

    axes[2, 0].plot(
        recall_epoch_training_history,
        label="Training Recall",
        linestyle="--",
        marker="o",
    )

    axes[2, 0].plot(
        recall_epoch_validation_history,
        label="Validation Recall",
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

    final_accuracy = sum(accuracy_epoch_validation_history) / len(
        accuracy_epoch_validation_history
    )
    final_f1_score = sum(f1_epoch_validation_history) / len(f1_epoch_validation_history)
    final_iou = sum(iou_epoch_validation_history) / len(iou_epoch_validation_history)
    final_recall = sum(recall_validation_history) / len(recall_validation_history)

    return final_accuracy, final_f1_score, final_iou, final_recall
