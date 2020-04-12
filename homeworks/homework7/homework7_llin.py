import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image
from matplotlib import pyplot as plt


tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

# load mnist from keras
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train.astype(np.float32)/255., x_test.astype(np.float32)/255.

# hyper parameters
new_im = Image.new('L', (448, 84))
image_size = 28*28
h_dim = 200
z_dim = 32
num_epochs = 150
batch_size = 100
learning_rate = 1e-3


class VAE(tf.keras.Model):
    def __init__(self):
        super(VAE, self).__init__()
        # input => h
        self.fc1 = keras.layers.Dense(h_dim,)
        self.dropout = keras.layers.Dropout(0.2)
        # h => mu and variance
        self.fc2 = keras.layers.Dense(z_dim)
        self.fc3 = keras.layers.Dense(z_dim)
        # sampled z => h
        self.fc4 = keras.layers.Dense(h_dim)
        # h => image
        self.fc5 = keras.layers.Dense(image_size)

    def encode(self, x):
        h = tf.nn.tanh(self.fc1(x))
        h = self.dropout(h)
        # mu, log_variance
        return self.fc2(h), self.fc3(h)

    #  calculate mean and std var, then use gaussian to produce z
    def reparameterize(self, mu, log_var):
        std = tf.exp(log_var * 0.5)
        eps = tf.random.normal(std.shape)
        return mu + eps * std

    # decode function
    def decode_logits(self, z):
        h = tf.nn.tanh(self.fc4(z))
        return self.fc5(h)

    # activation after decode
    def decode(self, z):
        return tf.nn.sigmoid(self.decode_logits(z))

    def call(self, inputs, training=None, mask=None):
        # encoder
        mu, log_var = self.encode(inputs)
        # sample
        z = self.reparameterize(mu, log_var)
        # decode
        x_reconstructed_logits = self.decode_logits(z)

        return x_reconstructed_logits, mu, log_var


model = VAE()
model.build(input_shape=(4, image_size))
model.summary()
optimizer = keras.optimizers.Adam(learning_rate)

# pre-processing
dataset = tf.data.Dataset.from_tensor_slices(x_train)
dataset = dataset.shuffle(batch_size * 5).batch(batch_size)
num_batches = x_train.shape[0] // batch_size

# train
for epoch in range(num_epochs):
    for step, x in enumerate(dataset):
        x = tf.reshape(x, [-1, image_size])
        with tf.GradientTape() as tape:
            # VAE forward
            x_reconstruction_logits, mu, log_var = model(x)
            # calculate reconstruction loss
            reconstruction_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_reconstruction_logits)
            reconstruction_loss = tf.reduce_sum(reconstruction_loss) / batch_size
            # Calculate the divergence between two gaussian distributions KL
            # the first is an unknown gaussian distribution
            # the second is N (0,1) distribution
            kl_div = - 0.5 * tf.reduce_sum(1. + log_var - tf.square(mu) - tf.exp(log_var), axis=-1)
            kl_div = tf.reduce_mean(kl_div)
            # loss = KL divergence + reconstruction loss
            loss = kl_div + tf.reduce_mean(reconstruction_loss)
            gradients = tape.gradient(loss, model.trainable_variables)
            # gradient clip
            for g in gradients:
                tf.clip_by_norm(g, 15)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            if (step + 1) % 50 == 0:
                print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                      .format(epoch + 1, num_epochs, step + 1, num_batches, float(reconstruction_loss), float(kl_div)))

    out_logits, _, _ = model(x[:16])
    out = tf.nn.sigmoid(out_logits)  # out is just the logits, use sigmoid
    out = tf.reshape(out, [-1, 28, 28]).numpy() * 255
    out = out.astype(np.uint8)
    # print(out.shape)
    x = tf.reshape(x[:16], [-1, 28, 28]).numpy() * 255
    x = x.astype(np.uint8)
    # print(x.shape)
    index = 0
    for i in range(0, 448, 28):
        im = x[index]
        im = Image.fromarray(im, mode='L')
        new_im.paste(im, (i, 0))
        index += 1

    # x_concat = tf.concat([x, out], axis=0).numpy() * 255.
    # x_concat = x_concat.astype(np.uint8)
    # print(x_concat.shape)

    index = 0
    for i in range(0, 448, 28):
        im = out[index]
        im = Image.fromarray(im, mode='L')
        new_im.paste(im, (i, 28))
        index += 1

    # generate new image
    z = tf.random.normal((16, z_dim))
    out = model.decode(z)
    out = tf.reshape(out, [-1, 28, 28]).numpy() * 255
    out = out.astype(np.uint8)

    index = 0
    for i in range(0, 448, 28):
        im = out[index]
        im = Image.fromarray(im, mode='L')
        new_im.paste(im, (i, 56))
        index += 1

    # new_im.save('images/vae_triple_epoch_%d.png' % (epoch + 1))
    new_im.save('homework7_llin_tanh_dropout.png')
    plt.imshow(np.asarray(new_im))
    plt.show()
    print('New images saved !')
