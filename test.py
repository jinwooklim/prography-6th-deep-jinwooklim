import tensorflow as tf
from vgg16 import vgg16_skip, mnist_preprocess

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    _, (x_test, y_test) = mnist.load_data()

    model = vgg16_skip()
    model.load_weights('./checkpoints/vgg_mnist')

    # 5. Evaluate the model
    x_test = mnist_preprocess(x_test)
    predictions = model.predict(x_test)
    pred_onehot = [tf.math.argmax(predictions[i]).numpy() for i in range(len(x_test))]
    acc = sum([1 for i in range(len(y_test)) if pred_onehot[i] == y_test[i]]) / len(y_test)
    print(f'Accuracy : {acc*100}%')