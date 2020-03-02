import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model, layers


class vgg16_skip(Model):
    def __init__(self):
        super(vgg16_skip, self).__init__()
        kernel_size = 3
        activation_fn = tf.nn.relu

        self.conv1_1 = layers.Conv2D(64, kernel_size, activation=activation_fn, padding='SAME')
        self.conv1_2 = layers.Conv2D(64, kernel_size, activation=activation_fn, padding='SAME')
        self.pool_1 = layers.MaxPooling2D(strides=(2, 2), padding='SAME')

        self.conv2_1 = layers.Conv2D(128, kernel_size, activation=activation_fn, padding='SAME')
        self.conv2_2 = layers.Conv2D(128, kernel_size, activation=activation_fn, padding='SAME')
        self.pool_2 = layers.MaxPooling2D(strides=(2, 2), padding='SAME')

        self.conv3_1 = layers.Conv2D(256, kernel_size, activation=activation_fn, padding='SAME')
        self.conv3_2 = layers.Conv2D(256, kernel_size, activation=activation_fn, padding='SAME')
        self.conv3_3 = layers.Conv2D(256, kernel_size, activation=activation_fn, padding='SAME')
        self.pool_3 = layers.MaxPooling2D(strides=(2, 2), padding='SAME')

        self.conv4_1 = layers.Conv2D(512, kernel_size, activation=activation_fn, padding='SAME')
        self.conv4_2 = layers.Conv2D(512, kernel_size, activation=activation_fn, padding='SAME')
        self.conv4_3 = layers.Conv2D(512, kernel_size, activation=activation_fn, padding='SAME')
        self.pool_4 = layers.MaxPooling2D(strides=(2, 2), padding='SAME')

        self.conv5_1 = layers.Conv2D(512, kernel_size, activation=activation_fn, padding='SAME')
        self.conv5_2 = layers.Conv2D(512, kernel_size, activation=activation_fn, padding='SAME')
        self.conv5_3 = layers.Conv2D(512, kernel_size, activation=activation_fn, padding='SAME')
        self.pool_5 = layers.MaxPooling2D(strides=(2, 2), padding='SAME')

        self.x_flat = layers.Flatten()
        self.skip_flat = layers.Flatten()
        self.fc_skip = layers.Dense(512)

        self.fc1 = layers.Dense(4096)
        self.fc2 = layers.Dense(1024)
        self.fc3 = layers.Dense(10, activation=tf.nn.softmax)

    def call(self, inputs):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.pool_1(x)
        skip = x

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool_2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool_3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool_4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool_5(x)

        x = self.x_flat(x)
        skip = self.skip_flat(skip)
        skip = self.fc_skip(skip)
        # 3. Skip-connection
        x = x + skip

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


def mnist_preprocess(input):
    # input : np_array
    input = np.expand_dims(input, axis=3)
    input = np.tile(input, (1, 1, 1, 3))
    input = input / 255.0
    return input


if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), _ = mnist.load_data()

    # 1. Build Custom vgg16 model, Convert MNIST's channel to RGB
    # 2. Use Class
    model = vgg16_skip()
    x_train = mnist_preprocess(x_train)

    # 4. Train MNIST
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)
    model.save_weights('./checkpoints/vgg_mnist', save_format='tf')