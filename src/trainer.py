import matplotlib.pyplot as plt
import tensorflow as tf

def compile_and_train(model, train_gen, valid_gen, lr=1e-4, epochs=5):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    history = model.fit(train_gen, epochs=epochs, validation_data=valid_gen)
    return history

def evaluate(model, test_gen):
    return model.evaluate(test_gen)

def plot_history(history: tf.keras.callbacks.History):
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    plt.figure()
    plt.plot(epochs, hist["loss"], label="train loss")
    plt.plot(epochs, hist["val_loss"], label="val loss")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, hist["accuracy"], label="train acc")
    plt.plot(epochs, hist["val_accuracy"], label="val acc")
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.show()
