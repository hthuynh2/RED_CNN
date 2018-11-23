from keras import Input, layers, models, losses, optimizers, initializers, activations, callbacks
import os
import numpy as np
from PIL import Image
import utils

PATCH_SIZE = 55
checkpoints_dir = os.path.join(os.getcwd(), "checkpoints")
checkpoint_best_path = os.path.join(checkpoints_dir, 'best_weights.hdf5')
checkpoint_last_path = os.path.join(checkpoints_dir, 'last_weights.hdf5')
outputs_dir = './outputs'
evaluate_dir = './evals'

class RED_CNN(object):
    def __init__(self, num_kernel_per_layer=96, num_kernel_last_layer=1, kernel_size=(5, 5), lr=0.0001):
        print("Initializing model...")
        self.total_num_epoch_to_train = 10000
        self.batch_size = 128

        self.num_kernel_per_layer = num_kernel_per_layer
        self.num_kernel_last_layer = num_kernel_last_layer
        self.kernel_size = kernel_size
        self.lr = lr

        gauss_initializer = initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        y_0 = Input(shape=(None, None, 1))  # adapt this if using `channels_first` image data format
        y_1 = layers.Conv2D(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid', kernel_initializer=gauss_initializer)(y_0)
        y_2 = layers.Conv2D(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid', kernel_initializer=gauss_initializer)(y_1)
        y_3 = layers.Conv2D(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid', kernel_initializer=gauss_initializer)(y_2)
        y_4 = layers.Conv2D(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid', kernel_initializer=gauss_initializer)(y_3)
        y_5 = layers.Conv2D(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid', kernel_initializer=gauss_initializer)(y_4)
        y_5_deconv = layers.Conv2DTranspose(self.num_kernel_per_layer, self.kernel_size, activation=None, padding='valid', kernel_initializer=gauss_initializer)(y_5)
        y4_add_y5deconv = layers.Add()([y_4, y_5_deconv])
        y_6 = layers.Activation(activation='relu')(y4_add_y5deconv)
        y_7 = layers.Conv2DTranspose(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid', kernel_initializer=gauss_initializer)(y_6)
        y_7_deconv = layers.Conv2DTranspose(self.num_kernel_per_layer, self.kernel_size, activation=None, padding='valid', kernel_initializer=gauss_initializer)(y_7)
        y2_add_y7deconv = layers.Add()([y_2, y_7_deconv])
        y_8 = layers.Activation(activation='relu')(y2_add_y7deconv)
        y_9 = layers.Conv2DTranspose(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid', kernel_initializer=gauss_initializer)(y_8)
        # Note: last layer only has 1 kernel
        y_9_deconv = layers.Conv2DTranspose(self.num_kernel_last_layer, self.kernel_size, activation=None, padding='valid', kernel_initializer=gauss_initializer)(y_9)
        y0_add_y9_deconv = layers.Add()([y_0, y_9_deconv])
        output = layers.Activation(activation='relu')(y0_add_y9_deconv)

        self.model = models.Model(y_0, output)
        optimizer = optimizers.Adam(lr=lr)
        self.model.compile(loss=losses.mean_squared_error, optimizer=optimizer)
        self.current_epoch = 0
        if not os.path.exists(evaluate_dir):
            os.mkdir(evaluate_dir)

    def train(self, train_data, train_labels):
        print("Start training...")
        self.load_last_model()

        learning_rate_update_callback = callbacks.LearningRateScheduler(step_decay, verbose=1)
        checkpoint_last_callback = callbacks.ModelCheckpoint(checkpoint_last_path, verbose=1)
        checkpoint_best_only_callback = callbacks.ModelCheckpoint(checkpoint_best_path, verbose=1, save_best_only=True)

        my_prediction_callback = My_prediction_callback()
        callbacks_list = [checkpoint_last_callback, checkpoint_best_only_callback, my_prediction_callback, learning_rate_update_callback]
        self.model.fit(x=train_data, y=train_labels,
                       epochs=self.total_num_epoch_to_train,
                       batch_size=self.batch_size,
                       shuffle=True,
                       validation_split=0.1,
                       callbacks=callbacks_list)

    # def eval(self, noisy_img, save_name, clean_img=None):
    #     prediction = self.model.predict(np.array([noisy_img]))[0]
    #     prediction = prediction * 255
    #     prediction = prediction.astype('uint8').reshape((128, 128))
    #     predicted_img = Image.fromarray(prediction)
    #     save_name += "_" + str(self.current_epoch) + '.png'
    #     save_path = os.path.join(evaluate_dir, save_name)
    #     predicted_img.save(save_path)

    def load_last_model(self):
        print("[*] Reading checkpoint...")
        if not os.path.exists(checkpoints_dir):
            os.mkdir(checkpoints_dir)
            print("No checkpoint found.")
            return

        if not os.path.exists(checkpoint_last_path):
            print("No checkpoint found.")
            return
        self.model.load_weights(checkpoint_last_path)
        print("Load checkpoint successfully.")

    # def save_model(self):
    #     print("[*] Saving checkpoint... " + str(self.current_epoch))
    #     file_name = "checkpoint_" + format(self.current_epoch, "07") + ".h5"
    #     save_path = os.path.join(checkpoints_dir, file_name)
    #     self.model.save(save_path)


def step_decay(epoch):
   initial_lrate = 0.0005
   drop = 0.75
   epochs_drop = 10.0
   lrate = initial_lrate * np.power(drop,
           np.floor((1+epoch)/epochs_drop))
   lrate = max(lrate, 0.00001)
   return lrate

class My_prediction_callback(callbacks.Callback):
    def __init__(self):
        test_noisy_image = utils.imread(utils.get_image_path(False, 64, 4003))
        test_noisy_image = utils.scale_image(test_noisy_image, 2.0)  # Image size 128x128
        test_noisy_image /= 255.0
        test_noisy_image = test_noisy_image.reshape(128, 128, 1)
        self.noisy_img1 = test_noisy_image

        test_noisy_image = utils.imread(utils.get_image_path(False, 64, 19983))
        test_noisy_image = utils.scale_image(test_noisy_image, 2.0)  # Image size 128x128
        test_noisy_image /= 255.0
        test_noisy_image = test_noisy_image.reshape(128, 128, 1)
        self.noisy_img2 = test_noisy_image

    def on_epoch_end(self, epoch, logs={}):
        prediction = self.model.predict(np.array([self.noisy_img1]))[0]
        prediction = prediction * 255
        prediction = prediction.astype('uint8').reshape((128, 128))
        predicted_img = Image.fromarray(prediction)
        save_name = "img_4003_" + str(epoch) + '.png'
        save_path = os.path.join(evaluate_dir, save_name)
        predicted_img.save(save_path)

        prediction = self.model.predict(np.array([self.noisy_img2]))[0]
        prediction = prediction * 255
        prediction = prediction.astype('uint8').reshape((128, 128))
        predicted_img = Image.fromarray(prediction)
        save_name = "img_19983_" + str(epoch) + '.png'
        save_path = os.path.join(evaluate_dir, save_name)
        predicted_img.save(save_path)
