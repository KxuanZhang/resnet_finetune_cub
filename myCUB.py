import torch
import numpy as np
import os
from PIL import Image, TarIO
import pickle
import tarfile
import os.path as osp
import random
import math
from  torchvision import transforms

def load_DTD(dataset_dir):
    items_train = []
    items_test = []
    classDir = os.listdir(dataset_dir)

    random.seed(0)
    for id_x,c in enumerate( classDir ):
        tempDir = osp.join( dataset_dir , c )
        imageFiles = os.listdir(tempDir)
        if id_x >= 50:
            break
        # 划分训练测试
        num_files = len(imageFiles)
        idList = [i for i in  range(num_files)]
        train_id = random.sample(idList,math.floor( len(imageFiles)/10 * 9) )
        val_test_id = list( set(idList) - set(train_id) )
        # test_id = random.sample( val_test_id , (len(val_test_id)  ))
        test_id = val_test_id
        for i in idList:
            imname = imageFiles[i]
            impath = osp.join(tempDir , imname)
            label = id_x
            if i in train_id:
                items_train.append(( impath, label ))
            elif i in test_id:
                items_test.append((impath, label))


    # return items_train, items_test
    return items_train, items_train


class myCUB(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        super(myCUB, self).__init__()
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([448,448]),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
        self.root = root
        self.train = train
        self.transform = test_transforms
        self.dataset_dir = osp.join(root , 'images')
        self.train_x,self.train_y,self.test_x , self.test_y = self._read_data()
        print(len(self.train_x))
        
    def __len__(self):
        if self.train:
            return len(self.train_x)
        else:
            return len(self.test_x)
        
    def __getitem__(self, idx):
        if self.train:
            img, label = self.train_x[idx], self.train_y[idx]
        else:
            img, label = self.test_x[idx], self.test_y[idx]
        img = Image.open(img).convert('RGB')
        img = np.array(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label        
    
    
    def _read_data(self):
        items_train = []
        items_test = []
        items_d_train, items_d_test = load_DTD(self.dataset_dir )
        train_x = []
        train_y = []
        test_x = []
        test_y = []

        for impath, label in items_d_train:
            train_x.append( impath)
            train_y.append( label )
            
        for impath, label in items_d_test:
            test_x.append( impath)
            test_y.append( label )
            
        return train_x, train_y, test_x, test_y
        
        
if __name__ == "__main__":
    datasetDir = 'D:/Dataset/CUB/CUB_200_2011'
    ds = myCUB(datasetDir, train=True)
    
    dl = torch.utils.data.DataLoader(
            ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True
        )
    
    for i , (x,y) in enumerate(dl):
        print(x)