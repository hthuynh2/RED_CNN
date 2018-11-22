import model
from utils import get_image_path, scale_image, imread, load_data

def main():
    data, labels = load_data()
    my_model = model.RED_CNN()
    test_noisy_image = imread(get_image_path(False, 64, 4003))
    test_noisy_image = scale_image(test_noisy_image, 2.0)  # Image size 128x128
    test_noisy_image /= 255.0
    test_noisy_image = test_noisy_image.reshape(128, 128, 1)
    my_model.train(data, labels, test_noisy_image=test_noisy_image)

if __name__ == '__main__':
    main()