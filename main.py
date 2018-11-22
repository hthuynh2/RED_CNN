import model
from utils import get_image_path, scale_image, imread, load_data

def main():
    data, labels = load_data()
    my_model = model.RED_CNN()
    my_model.train(data, labels)

if __name__ == '__main__':
    main()