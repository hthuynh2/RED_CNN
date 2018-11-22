from keras import Input, layers, activations, models, losses, optimizers, initializers
import os
import numpy as np
from PIL import Image

PATCH_SIZE = 56
checkpoints_dir = './checkpoints'
outputs_dir = './outputs'
evaluate_dir = './evals'

class RED_CNN(object):
    def __init__(self, num_kernel_per_layer=96, num_kernel_last_layer=1, kernel_size=(5, 5), lr=0.0001):
        self.total_num_epoch_to_train = 10000
        self.save_every_num_epoch = 1
        self.batch_size = 128

        self.num_kernel_per_layer = num_kernel_per_layer
        self.num_kernel_last_layer = num_kernel_last_layer
        self.kernel_size = kernel_size
        self.lr = lr

        gauss_initializer = initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        y_0 = Input(shape=(PATCH_SIZE, PATCH_SIZE, 1))  # adapt this if using `channels_first` image data format
        y_1 = layers.Conv2D(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid', kernel_initializer=gauss_initializer)(y_0)
        y_2 = layers.Conv2D(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid', kernel_initializer=gauss_initializer)(y_1)
        y_3 = layers.Conv2D(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid', kernel_initializer=gauss_initializer)(y_2)
        y_4 = layers.Conv2D(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid', kernel_initializer=gauss_initializer)(y_3)
        y_5 = layers.Conv2D(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid', kernel_initializer=gauss_initializer)(y_4)
        y_5_deconv = layers.Conv2DTranspose(self.num_kernel_per_layer, self.kernel_size, activation=None, padding='valid', kernel_initializer=gauss_initializer)(y_5)
        y4_add_y5deconv = layers.Add()([y_4, y_5_deconv])
        y_6 = activations.relu(y4_add_y5deconv)
        y_7 = layers.Conv2DTranspose(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid', kernel_initializer=gauss_initializer)(y_6)
        y_7_deconv = layers.Conv2DTranspose(self.num_kernel_per_layer, self.kernel_size, activation=None, padding='valid', kernel_initializer=gauss_initializer)(y_7)
        y2_add_y7deconv = layers.Add()([y_2, y_7_deconv])
        y_8 = activations.relu(y2_add_y7deconv)
        y_9 = layers.Conv2DTranspose(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid', kernel_initializer=gauss_initializer)(y_8)
        # Note: last layer only has 1 kernel
        y_9_deconv = layers.Conv2DTranspose(self.num_kernel_last_layer, self.kernel_size, activation=None, padding='valid', kernel_initializer=gauss_initializer)(y_9)
        y0_add_y9_deconv = layers.Add()([y_0, y_9_deconv])
        output = activations.relu(y0_add_y9_deconv)

        self.model = models.Model(y_0, output)
        optimizer = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
        self.model.compile(loss=losses.mean_squared_error, optimizer=optimizer)
        self.current_epoch = 0

    def train(self, train_data, train_labels):
        self.load_model()
        while self.current_epoch < self.total_num_epoch_to_train:
            self.model.fit(x=train_data, y=train_labels,
                           epochs=self.save_every_num_epoch,
                           batch_size=self.batch_size,
                           shuffle=True)
            self.save_model()
            self.current_epoch += self.save_every_num_epoch
            self.eval(noisy_img=train_data[0], save_name='img_4000')

    def eval(self, noisy_img, save_name, clean_img=None):
        prediction = self.model.predict(np.array([noisy_img]))[0]
        prediction = prediction * 255
        prediction = prediction.astype('uint8').reshape((128, 128))
        predicted_img = Image.fromarray(prediction)
        save_name += "_" + str(self.current_epoch) + '.png'
        save_path = os.path.join(evaluate_dir, save_name)
        predicted_img.save(save_path)

    def load_model(self):
        print("[*] Reading checkpoint...")
        if not os.path.exists(checkpoints_dir):
            os.mkdir(checkpoints_dir)
            print("No checkpoint found.")
            return
        files = os.listdir(checkpoints_dir)
        max_checkpoint_num = -1

        for file_name in files:
            checkpoint_num = int(file_name.split('_')[1])
            max_checkpoint_num = max(max_checkpoint_num, checkpoint_num)

        if max_checkpoint_num == -1:
            print("No checkpoint found.")
            return
        file_name = "checkpoint_" + format(max_checkpoint_num, "07") + ".h5"
        latest_checkpoint_path = os.path.join(checkpoints_dir, file_name)
        self.model.load_weights(latest_checkpoint_path)
        self.current_epoch = max_checkpoint_num

    def save_model(self):
        file_name = "checkpoint_" + format(self.current_epoch, "07") + ".h5"
        save_path = os.path.join(checkpoints_dir, file_name)
        self.model.save(save_path)




