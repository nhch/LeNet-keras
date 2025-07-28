import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools

# Imposta uno stile gradevole e coerente per tutti i grafici
plt.style.use('ggplot')


def show_samples(x, y, rows: int = 2, cols: int = 5) -> None:
    """Mostra un collage di immagini del dataset con la rispettiva etichetta.

    Args:
        x (np.ndarray): immagini normalizzate in scala 0‑1, shape (n, h, w, 1).
        y (np.ndarray): etichette one‑hot, shape (n, num_classes).
        rows (int, optional): righe del collage. Defaults to 2.
        cols (int, optional): colonne del collage. Defaults to 5.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    idx = np.random.choice(np.arange(x.shape[0]), size=rows * cols, replace=False)

    for i, ax in enumerate(axes.flatten()):
        img = x[idx[i]].squeeze()
        label = int(np.argmax(y[idx[i]]))
        ax.imshow(img, cmap="gray")
        ax.set_title(label)
        ax.axis("off")

    fig.tight_layout()
    plt.show()


def plot_loss(history) -> None:
    """Grafico della funzione di perdita per train e validation."""
    plt.figure()
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="best")
    plt.show()


def plot_accuracy(history) -> None:
    """Grafico dell'accuracy per train e validation."""
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="best")
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list | None = None,
    normalize: bool = True,
) -> None:
    """Disegna la confusion matrix.

    Args:
        y_true (np.ndarray): etichette vere, shape (n,).
        y_pred (np.ndarray): etichette predette, shape (n,).
        class_names (list | None, optional): nomi delle classi da mostrare. Defaults to None.
        normalize (bool, optional): se True normalizza la matrice per riga. Defaults to True.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix" + (" (normalized)" if normalize else ""),
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j,
            i,
            format(cm[i, j], fmt),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    fig.tight_layout()
    plt.show()
