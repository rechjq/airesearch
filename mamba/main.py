import torch
import argparse

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ss.models.config import MambaConfig
from transformers import AutoTokenizer, TrainingArguments,default_data_collator
import  sentencepiece  as spm
from .dataset import PretrainDataset
from trainer.mamba_trainer import MambaTrainer


def run(args):
    if args.model == "":
        #2.8b config
        mc=MambaConfig(
            d_model=2560,
            n_layer=64,
            vocab_size=64392,
            rms_norm= True,
            residual_in_fp32=True,
            fused_add_norm=True,
        pad_vocab_size_multiple=8)
        model = MambaLMHeadModel(mc, dtype=torch.bfloat16, device="cuda")
    else:
        model = MambaLMHeadModel.from_pretrained(args.model, dtype=torch.bfloat16, device="cuda")

    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    
    dataset = PretrainDataset(args.data_path, tokenizer, max_seq_length=8192,memmap=True)


    trainer = MambaTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=TrainingArguments(
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optim=args.optim,
            output_dir="mamba-chat",
            logging_steps=50,
            save_steps=500,
        ),
        data_collator=default_data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--tokenizer", type=str, default="../tokenizer/tokenizer.model")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--data_path", type=str, default="./wiki16.bin")
    parser.add_argument("--num_epochs", type=int, default=1)
    args = parser.parse_args()

    run(args)