from torch.utils.data import Dataset
import torch
import numpy as np
class PretrainDataset(Dataset):
    def __init__(self,data_path_lst,max_length=256,memmap=False):
        super().__init__()
        if memmap:
            with open(data_path_lst[0],'r') as f:
                nbytes = f.seek(0,2)
                flen = f.tell() // np.dtype('uint16').itemsize
            self.data = np.memmap(data_path_lst[0],dtype=np.dtype('uint16'),shape=(flen//max_length,max_length))
        else:
            data_lst=[]
            for data_path in data_path_lst:
                with open(data_path,'rb') as f:
                    data=np.fromfile(f,dtype=np.uint16)
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
        #
        sample = self.data[index]
        X=np.array(sample[:-1]).astype(np.int64)
        Y=np.array(sample[1:]).astype(np.int64)
        
        return torch.from_numpy(X),torch.from_numpy(Y)


class SFTDataset(Dataset):
    def __init__(self,df,tokenizer
                 ,max_length=256
                 ,prompt_max_len=128
                 ,answer_max_len=128):
        super().__init__()
        self.df=df
        self.max_length = max_length
        self.prompt_max_len = prompt_max_len
        self.answer_max_len = answer_max_len
        #
        self.tokenizer = tokenizer
        self.bos=self.tokenizer.special_tokens['<bos>']
        self.eos=self.tokenizer.special_tokens['<eos>']
        self.pad=0#self.tokenizer.special_tokens['<pad>']
        
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, index: int):
        #
        sample = self.df.iloc[index]
        prompt = self.tokenizer.encode(sample['prompt'],add_special_tokens=False)
        answer = self.tokenizer.encode(sample['answer'],add_special_tokens=False)
        if len(prompt) > self.prompt_max_len:
            prompt = prompt[:self.prompt_max_len-2]
        if len(answer) > self.answer_max_len:
            answer = answer[:self.answer_max_len-2]
        #
        input_id=prompt+[self.bos]+answer+[self.eos]
        context_length = input_id.index(self.bos)
        mask_position = context_length - 1
        pad_len = self.max_length - len(input_id)
        input_id = input_id + [self.pad] * pad_len
        if pad_len==0:
            loss_mask = [0]*context_length+[1]*(len(input_id[mask_position+1:])) + [0]*pad_len
        else:
            loss_mask = [0]*context_length+[1]*(len(input_id[mask_position+1:-pad_len])) + [0]*pad_len
        #
        input_id=np.array(input_id)
        X=np.array(input_id[:-1]).astype(np.int64)
        Y=np.array(input_id[1:]).astype(np.int64)
        loss_mask=np.array(loss_mask[:-1])
        #
        return torch.from_numpy(X),torch.from_numpy(Y),torch.from_numpy(loss_mask)