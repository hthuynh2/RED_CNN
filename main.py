import model
import utils

def main():
    data, labels = utils.load_data()
    my_model = model.RED_CNN()
    my_model.train(data, labels)

if __name__ == '__main__':
    main()