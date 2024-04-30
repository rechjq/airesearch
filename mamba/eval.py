import time
import json

import torch


from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import  sentencepiece  as spm
device = "cuda:0"
dtype = torch.float16
tokenizer = spm.SentencePieceProcessor(model_file="tokenizer/tokenizer.model")
model = MambaLMHeadModel.from_pretrained("mytrain", device=device, dtype=dtype)
input_ids=tokenizer.encode("杰米·莫耶（Moyer, Jamie），前美国职业棒球选手，守备位置为投手。曾效力于美国职棒大联盟科罗拉多洛矶等队。")
input_ids = torch.LongTensor([input_ids]).to(device)
ilen = len(input_ids)

# print(input_ids.shape)
out =model.generate(
        input_ids=input_ids,
        max_length=512,
        return_dict_in_generate=True,
        output_scores=True,
        enable_timing=False,
        temperature=0.1,
        top_k=1,
        top_p=1.0,
        min_p=0.1,
        repetition_penalty=1.1)
seq=out.sequences
s=seq[0]
data=[]
for item in s.tolist():
    if item not in [tokenizer.eos_id(),tokenizer.unk_id(), tokenizer.bos_id(), tokenizer.pad_id()]:
        data.append(item)
    else:
        break
        
tokenizer.decode(data)
