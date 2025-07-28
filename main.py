import numpy as np
from data import DATA
from lenet import LeNet
from keras.src.callbacks import EarlyStopping

from visualizations import (
    show_samples,
    plot_loss,
    plot_accuracy,
    plot_confusion_matrix,
)

def main():
    batch_size = 128
    epochs = 25

    data = DATA()
    model = LeNet(data.input_shape, data.num_classes)

    # 1) Uno sguardo ai dati
    show_samples(data.x_train, data.y_train)


    early_stop = EarlyStopping(
        monitor='val_loss',      # metrica da osservare
        patience=3,              # quante epoche di “stallo” tollerare
        restore_best_weights=True
    )

    # 2) Addestramento
    hist = model.fit(
        data.x_train,
        data.y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=2,
    )

    # 3) Curve loss + accuracy
    plot_loss(hist)
    plot_accuracy(hist)

    # 4) Valutazione e confusion matrix
    y_pred = model.predict(data.x_test, batch_size=batch_size)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(data.y_test, axis=1)
    plot_confusion_matrix(y_true, y_pred_classes)

if __name__ == '__main__':
    main()