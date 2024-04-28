import torch
import transformers
import json

from dataclasses import dataclass
from typing import Dict, Sequence
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np

class PretrainDataset(Dataset):
    def __init__(self,data_path_lst,max_length=256,memmap=False, dtype='uint16'):
        super().__init__()
        if memmap:
            with open(data_path_lst[0],'r') as f:
                nbytes = f.seek(0,2)
                flen = f.tell() // np.dtype(dtype).itemsize
            self.data = np.memmap(data_path_lst[0],dtype=np.dtype(dtype),shape=(flen//max_length,max_length))
        else:
            data_lst=[]
            for data_path in data_path_lst:
                with open(data_path,'rb') as f:
                    data=np.fromfile(f,dtype=np.dtype(dtype))
                    data_lst.append(data)
            data = np.concatenate(data_lst)
            data = data[:max_length*int(len(data)/max_length)]
            #np.random.shuffle(data)
            self.data = data.reshape(-1,max_length)
        #
        print("memmap:{} train data.shape:{}".format(memmap,self.data.shape))
        print("downloading finished.....")
        
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index: int):
        return dict(input_ids=self.data[index], labels=self.data[index])