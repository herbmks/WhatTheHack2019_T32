
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class WhatTheNet(torch.nn.Module):
     
    def __init__(self, n_inputs, n_outputs):   
        # super().__init__(self,n_inputs, n_outputs)
        super(WhatTheNet, self).__init__()

        # self.bn1 = torch.nn.BatchNorm1d(n_inputs)
        self.fc1 = torch.nn.Linear(n_inputs, 3200)   # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
        self.fc2 = torch.nn.Linear(3200, 1600)       # convert matrix with 120 features to a matrix of 84 features (columns)
        self.fc3 = torch.nn.Linear(1600, 800)       # convert matrix with 120 features to a matrix of 84 features (columns)
        self.fc4 = torch.nn.Linear(800, 400)       # convert matrix with 120 features to a matrix of 84 features (columns)
        self.fc41 = torch.nn.Linear(400, 400)       # convert matrix with 120 features to a matrix of 84 features (columns)
        self.fc5 = torch.nn.Linear(400, n_outputs)        # convert matrix with 84 features to a matrix of 10 features (columns)
        
    def forward(self, x):
        # x = self.bn1(x)
        x = torch.nn.functional.tanh(self.fc1(x))
        x = torch.nn.functional.tanh(self.fc2(x))
        x = torch.nn.functional.tanh(self.fc3(x))
        x = torch.nn.functional.tanh(self.fc4(x))
        x = torch.nn.functional.tanh(self.fc41(x))
        x = self.fc5(x)
        
        
        return x
     

class WTHdataloader(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, allxml, allkeys, predict_key):
        self.allxml = allxml
        self.predict_key = predict_key
        self.allkeys = allkeys

        # get all entries that have the predictkey
        self.filteredxml = []
        for xml in allxml:
            if predict_key in xml:
                self.filteredxml.append(xml)
        print('Found ',len(self),'xmls with the key',predict_key)
        

        # normalize
        total = np.zeros(len(allkeys))
        count = np.zeros(len(allkeys))
        mapping = {}
        for i, k in enumerate(allkeys):
            mapping[k] = i

        for xml in self.filteredxml:
            for key, val in xml.items():
                if key in mapping.keys():
                    if isinstance(val, list):
                        val = val[0]
                    total[mapping[key]] += val
                    count[mapping[key]] += 1
        self.avgval = total / count

        self.avglabel = 0
        self.avglabelcnt = 0
        for xml in self.filteredxml:
            labelval = xml[predict_key]
            if isinstance(labelval, list):
                labelval = labelval[0]
            self.avglabel += labelval
            self.avglabelcnt += 1
        self.avglabel /= self.avglabelcnt
        print('avges', self.avgval, self.avglabel)



    def __len__(self):
        return len(self.filteredxml)

    def __getitem__(self, idx):
        a =self.filteredxml[idx]

        trainvec = np.ones(len(self.allkeys))*-1
        for i, k in enumerate(self.allkeys):
            if k in a:
                val = a[k]
                if isinstance(val, list):
                    val = val[0]

                trainvec[i] = val
        trainvec /= self.avgval # normalize
        label = a[self.predict_key]
        # print(label)
        if isinstance(label, list):
            label = label[0]
        label /= self.avglabel
        trainvec =  torch.from_numpy(trainvec).float()
        label = torch.from_numpy(np.asarray([label])).float()
        # print('label shape',label.shape)
        # print('tv',trainvec, label)
        return trainvec, label
        