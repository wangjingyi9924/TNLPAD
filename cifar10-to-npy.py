import pickle
import glob
import numpy as np


def Dataloader():
    data_list = glob.glob("./data/cifar-10-batches-py/test_batch")

    for data in data_list:
        data = pickle.load(open(data, 'rb'), encoding='bytes')
        labels, data, filenames = data[b'labels'], data[b'data'], data[b'filenames']
        labels, data = map(np.array, [labels, data])
        try:
            Data = np.r_[Data, data]
            Labels = np.r_[Labels, labels]
        except:
            Data = data
            Labels = labels

    np.save("data/cifar10/test_images.npy", Data)
    np.save("data/cifar10/test_labels.npy", Labels)


if __name__ == "__main__":
    Dataloader()
