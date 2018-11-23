import model
from utils import get_image_path, scale_image, imread, load_data

IS_TRAINING = False
def main():
    if IS_TRAINING:
        data, labels = load_data()
        my_model = model.RED_CNN()
        my_model.train(data, labels)
    else:
        my_model = model.RED_CNN()
        my_model.test()


if __name__ == '__main__':
    main()