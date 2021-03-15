from torch.utils.data import Dataset
import torch

path = "/home/adarsh/PycharmProjects/Disentaglement/data/3dshapes.h5"
import h5py

class NewDataset(Dataset):
    def __init__(self, transform=None):

        self.dataset = path
        self.images = None
        with h5py.File(self.dataset,'r', libver='latest', swmr=True) as file:
            self.dataset_len = len(file['images'])

    def __getitem__(self, index):
        if self.images is None:
            self.images = h5py.File(self.dataset,'r',libver='latest', swmr=True)['images']
            self.labels = h5py.File(self.dataset,'r',libver='latest', swmr=True)['labels']
        return self.images[index]

    def __len__(self):
        return self.dataset_len
