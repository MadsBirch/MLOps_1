from unittest import TestLoader
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np


def get_data(phase = 'train'):
    
    BASE_DIR = './corruptmnist/'
    train_imgs, train_labs = [], []
    test_imgs, test_labs = [], []
    
    # load 5 training sets from .npz file  
    for i in range(5):
        train_dict = np.load(BASE_DIR + 'train_' + str(i) +'.npz')
        
        train_imgs.append(train_dict['images'])
        train_labs.append(train_dict['labels'])
    
    # load the single test file from .npz file
    test_dict = np.load(BASE_DIR+'test.npz')
    test_imgs = test_dict['images']
    test_labs = test_dict['labels']
    
    # concat elements in list to single tensor.
    # unsqueezo to get number of color channels for CNN
    train_imgs = torch.unsqueeze(torch.from_numpy(np.concatenate(train_imgs, axis = 0)), dim = 1)
    test_imgs = torch.unsqueeze(torch.from_numpy(np.array(test_imgs)), dim = 1)
    
    train_labs = torch.from_numpy(np.reshape(train_labs, 25000))
    test_labs = torch.from_numpy(np.reshape(test_labs, 5000))
    
    if phase == 'train':
        return train_imgs.type(torch.float32), train_labs
    
    elif phase == 'test':
        return  test_imgs.type(torch.float32), test_labs


class CorruptedMNIST(Dataset):
    def __init__(self, data):
        # data loading
        self.images, self.labels = data
            
    def __len__(self):
        # 25000 train images
        # 5000 test images
        return len(self.labels)
    
    def __getitem__(self, index):        
        return self.images[index,:], self.labels[index]
    
def get_dataloaders(batch_size = 64,
                    num_workers = 0,
                    ):
    get_train_data = get_data(phase='train')
    get_test_data = get_data(phase='test')

    traindata = CorruptedMNIST(get_train_data)
    testdata = CorruptedMNIST(get_test_data)
    
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return trainloader, testloader