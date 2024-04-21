from dataset import PretrainDataset,SFTDataset
import math
import logging
import torch
from llama import Transformer
import time
from contextlib import nullcontext
import os
import  argparse
from config import LLamaConfig,ModelArgs,LLamaTrainConfig
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel  import  DistributedDataParallel  as DDP
import pandas as pd
from sptokenizer  import getTokenizerModel 
import  torch.nn.functional  as F

## torchrun --standalone --nproc_per_node=4 pretrain.py OR python -m torch.distributed.launch --nproc_per_node=4 pretrain.py

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def get_lr(it, train_config:LLamaTrainConfig):
    # 1) linear warmup for warmup_iters steps
    if it < train_config.warmup_iters:
        return train_config.learning_rate * it / train_config.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > train_config.lr_decay_iters:
        return train_config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - train_config.warmup_iters) / (train_config.lr_decay_iters - train_config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return train_config.min_lr + coeff * (train_config.learning_rate - train_config.min_lr)

def scratch_model(model_args:ModelArgs):
    model = Transformer(model_args)
    return model
#  path: ckpt.pt
def load_checkpoint_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    model_args = dict()
    for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
        model_args[k] = checkpoint_model_args[k]
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
   
    #这是由于模型使用Compile才会有
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    return model
def pretrain(max_epoch, train_loader, config:LLamaConfig, model:Transformer,optimizer, ddp,device):
    start_time=time.time()
    train_config = config.trainArgs
    logger = get_logger(os.path.join(train_config.output,'log.log'))
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[train_config.dtype]
    
    iter_per_epoch=len(train_loader)
    ctx = (
        nullcontext()
        if train_config.device == "cpu"
        else torch.amp.autocast(device_type=train_config.device, dtype=ptdtype)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(train_config.dtype == 'float16'))
    raw_model = model.module if ddp else model 
    for epoch in range(train_config.max_epoch):
        for step, (X, Y) in enumerate(train_loader):
            X=X.to(device)
            Y=Y.to(device)
            lr = get_lr(epoch*iter_per_epoch+step, train_config) if train_config.decay_lr else train_config.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            # and using the GradScaler if data type is float16
            #for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = 0 == train_config.gradient_accumulation_steps - 1
            with ctx:
                logits = model(X, Y)
                loss = raw_model.last_loss
                loss = loss / train_config.gradient_accumulation_steps
            scaler.scale(loss).backward()
        #
            if (step + 1) % train_config.gradient_accumulation_steps == 0:
            # clip the gradient
                if train_config.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
                
                scaler.step(optimizer) #内部会调:optimizer.step()
                scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
                optimizer.zero_grad(set_to_none=True)
        #打印日志
            if step % train_config.log_interval == 0:
                spend_time=time.time()-start_time
                logger.info(
                    'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                        epoch,
                        max_epoch, 
                        step, 
                        iter_per_epoch,
                        loss.item(), 
                        optimizer.param_groups[-1]['lr'],
                        spend_time / (step+1) * iter_per_epoch // 60 - spend_time // 60))
        #
            if step % train_config.save_interval == 0:
                if ddp:
                    if torch.distributed.get_rank() == 0:
                        model.eval()
                        torch.save(model.module.state_dict(),'{}/iter_{}.pth'.format(train_config.output,int(step+epoch*iter_per_epoch)))
                        model.train()
                else:
                    model.eval()
                    torch.save(model.state_dict(),'{}/iter_{}.pth'.format(train_config.output,int(step+epoch*iter_per_epoch)))
                    model.train()

def sft(max_epoch, train_loader, config:LLamaConfig, model,optimizer, ddp,device):
    start_time=time.time()
    train_config = config.trainArgs
    logger = get_logger(os.path.join(train_config.output,'log.log'))
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[train_config.dtype]
#optimizer = model.configure_optimizers(train_config.weight_decay, train_config.learning_rate, (train_config.beta1, train_config.beta2), train_config.device_type)
    iter_per_epoch=len(train_loader)
    ctx = (
        nullcontext()
        if train_config.device_type == "cpu"
        else torch.amp.autocast(device_type=train_config.device, dtype=ptdtype)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(train_config.dtype == 'float16')) 
    for epoch in range(train_config.max_epoch):
        for step, (X, Y, loss_mask) in enumerate(train_loader):
            X=X.to(device)
            Y=Y.to(device)
            loss_mask=loss_mask.to(device)
            lr = get_lr(epoch*iter_per_epoch+step) if train_config.decay_lr else train_config.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            # and using the GradScaler if data type is float16
            #for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = 0 == train_config.gradient_accumulation_steps - 1
            with ctx:
                logits = model(X, Y)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=0,reduce=False)
                loss_mask = loss_mask.view(-1)
                loss = torch.sum(loss*loss_mask)/loss_mask.sum()
            scaler.scale(loss).backward()
        #
            if (step + 1) % train_config.gradient_accumulation_steps == 0:
            # clip the gradient
                if train_config.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
                scaler.step(optimizer) #内部会调:optimizer.step()
                scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
                optimizer.zero_grad(set_to_none=True)
        #打印日志
            if step % train_config.log_interval == 0:
                spend_time=time.time()-start_time
                logger.info(
                    'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                        epoch,
                        max_epoch, 
                        step, 
                        iter_per_epoch,
                        loss.item(), 
                        optimizer.param_groups[-1]['lr'],
                        spend_time / (step+1) * iter_per_epoch // 60 - spend_time // 60))
        #
            if step % train_config.save_interval == 0:
                if ddp:
                    if torch.distributed.get_rank() == 0:
                        model.eval()
                        torch.save(model.module.state_dict(),'{}/iter_{}.pth'.format(train_config.output,int(step+epoch*iter_per_epoch)))
                        model.train()
                else:
                    model.eval()
                    torch.save(model.state_dict(),'{}/iter_{}.pth'.format(train_config.output,int(step+epoch*iter_per_epoch)))
                    model.train()

def _getdata(args, max_seq_len, batch_size, module='pretrain'):
    if module == "pretrain":
        data_path_list = []
        # get dataset list
        for parent, _,files in os.walk(args.dataset):
            for file in files:
                data_path_list.append(os.path.join(parent,file))
        assert len(data_path_list) > 0
        train_ds = PretrainDataset(data_path_list, max_length=max_seq_len,memmap=True)
        #DistributedSampler 每个并行进程都会得到一个DistributedSampler，它从DataLoader加载数据
        # 它 负责只提供加载数据集中的一个子集，这些DistributedSampler提供的子集之间不重叠，交叉.
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            pin_memory=False,   #参数直接将数据集加载为 CUDA 张量,True 会在返回张量之前将张量复制到 CUDA 固定内存中。
            drop_last=False,  #数据不能满足就丢弃
            shuffle=False,        
            num_workers=4,
            sampler=train_sampler  #加载数据策列
        )
        return train_loader
    elif module == "sft":
        df=pd.read_csv(args.dataset)
        df=df.sample(frac=1.0)
        tokenizer = getTokenizerModel(args.tokenizer)
        train_ds = SFTDataset(df, tokenizer, max_length=max_seq_len)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            pin_memory=False,   #参数直接将数据集加载为 CUDA 张量,True 会在返回张量之前将张量复制到 CUDA 固定内存中。
            drop_last=False,  #数据不能满足就丢弃
            shuffle=False,        
            num_workers=4,
            sampler=train_sampler  #加载数据策列
        )
        return train_loader





def get_parse():
    args = argparse.ArgumentParser(description="Start Train...")
    args.add_argument("--config",default="config.json", help="配置文件")
    args.add_argument("--checkpoint",default=None, help="load check point")
    args.add_argument("--dataset",default=None, help="dataset dir")
    args.add_argument("--module",choices=["pretrain", "sft"], default="pretrain", help="与训练和sft训练")
    args.add_argument("--tokenizer",default="../tokenizer/mymodel.model", help="Tokenizer 模型")
    return  args
TrainFunction= {
    "pretrain": pretrain,
    "sft":sft
}
def main():
    parse=get_parse()
    args =parse.parse_args()
    llamaconfig = LLamaConfig.from_pretrain(args.config)
    trainArgs = llamaconfig.trainArgs
    if not os.path.exists(trainArgs.output): os.makedirs(trainArgs.output)
    #ddp 分布式
    device = trainArgs.device
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend=llamaconfig.trainArgs.backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        seed_offset = ddp_rank
        if trainArgs.device == 'cuda':
            device = f"cuda:{ddp_local_rank}"
            torch.cuda.set_device(device)
    else:
        seed_offset = 0
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    #ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[trainArgs.dtype]
    
    if args.checkpoint:
        model = load_checkpoint_model(args.checkpoint)
    else:
        model = scratch_model(llamaconfig.modelArgs)
    model.to(device)
    optimizer = model.configure_optimizers(trainArgs.weight_decay, trainArgs.learning_rate, (trainArgs.beta1, trainArgs.beta2), trainArgs.device_type)
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"freqs_cis"}
        model = DDP(model, device_ids=[ddp_local_rank])
    print("load model success..................................")
    train_loader = _getdata(args, llamaconfig.modelArgs.max_seq_len, trainArgs.batch_size,module=args.module)
    print("dataset  prepare..................................")
    train = TrainFunction[args.module]
    print("start train..................................")
    train(max_epoch=trainArgs.max_epoch, train_loader=train_loader,config=llamaconfig,model=model,optimizer=optimizer, ddp=ddp,device=device)
    if ddp:
        destroy_process_group()

    
if __name__ == '__main__':
    main()
