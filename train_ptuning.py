# coding=utf-8
# email: wangzejunscut@126.com

import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    TrainingArguments,
    Trainer
)

def parse_args():
    parser = argparse.ArgumentParser(description="ChatGLM P-Tuing v2")
    parser.add_argument("--train_args_file", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="THUDM/chatglm-6b")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--eval_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_output_length", type=int, default=1024)
    parser.add_argument("--pre_seq_len", type=int, default=None)
    parser.add_argument("--prefix_projection", action="store_true")
    parser.add_argument("--ptuning_checkpoint", type=str, default=None)
    parser.add_argument("--quantization_bit", type=int, choices=[4, 8], default=None)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    args = parser.parse_args()
    return args
    
class DataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        lengths = [len(feature["input_ids"]) for feature in batch]
        longest = max(lengths)
        input_ids, labels = [], []
        for length, feature in sorted(zip(lengths, batch), key=lambda x: -x[0]):
            pad_len = longest - length
            ids = feature["input_ids"] + [self.pad_token_id] * pad_len
            label = feature["labels"] + [-100] * pad_len
            input_ids.append(torch.LongTensor(ids))
            labels.append(torch.LongTensor(label))

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        return {"input_ids": input_ids, "labels": labels}

class ModifiedTrainer(Trainer):
    def _save(self, output_dir=None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "pytorch_model.bin"))

def train(args):
    parser = HfArgumentParser(TrainingArguments)
    training_args, = parser.parse_json_file(json_file=args.train_args_file)

    # Distributed training    
    if "LOCAL_RANK" in os.environ:
        training_args.local_rank = int(os.environ["LOCAL_RANK"])

    # Set seed
    set_seed(args.seed)
    training_args.seed = args.seed
    
    # Load model and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    config.pre_seq_len = args.pre_seq_len
    config.prefix_projection = args.prefix_projection
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    if args.ptuning_checkpoint is not None:
        # Loading extra state dict of prefix encoder
        model = AutoModel.from_pretrained(args.model_name_or_path, config=config, 
                                          trust_remote_code=True)
        prefix_state_dict = torch.load(os.path.join(args.ptuning_checkpoint, 
                                       "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    else:
        model = AutoModel.from_pretrained(args.model_name_or_path, config=config, 
                                          trust_remote_code=True)
    
    if args.quantization_bit is not None:
        print(f"Quantized to {args.quantization_bit} bit")
        model = model.quantize(args.quantization_bit)
    
    if args.pre_seq_len is not None:
        # P-tuning v2
        model = model.half()
        model.transformer.prefix_encoder.float()
    else:
        # deepspeed finetune
        model = model.float()
    
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
        
    # Load dataset
    data = load_dataset(path="json", data_files=args.data_path)
    column_names = data["train"].column_names
    
    def tokenize_function(example):
        question = example["instruction"]
        if example.get("input"):
            if example["input"].strip():
                question += f"\n{example['input']}"
        answer = example["output"]
        
        q_ids = tokenizer.encode(text=question, add_special_tokens=False)
        a_ids = tokenizer.encode(text=answer, add_special_tokens=False)
        if len(q_ids) > args.max_input_length - 1:
            q_ids = q_ids[: args.max_input_length - 1]
        if len(a_ids) > args.max_output_length - 2:
            a_ids = a_ids[: args.max_output_length - 2]

        input_ids = tokenizer.build_inputs_with_special_tokens(q_ids, a_ids)
        question_length = input_ids.index(tokenizer.bos_token_id)
        labels = [-100] * question_length + input_ids[question_length: ]
        return {"input_ids": input_ids, "labels": labels}
    
    train_dataset = data["train"].map(tokenize_function, remove_columns=column_names)
    eval_dataset = None
    if args.eval_path is not None:
        eval_data = load_dataset(path="json", data_files=args.eval_path)
        eval_dataset = eval_data["train"].map(tokenize_function, remove_columns=column_names)
    
    # trainer
    trainer = ModifiedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(pad_token_id=tokenizer.pad_token_id)
    )
    
    # train model
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    # Save model & tokenizer results
    if args.pre_seq_len is not None:
        print("Saving PrefixEncoder")
        os.makedirs(training_args.output_dir, exist_ok=True)
        saved_params = {
            k: v.to("cpu") for k, v in trainer.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(training_args.output_dir, "pytorch_model.bin"))
        """
        print("Saving PrefixEncoder")
        state_dict = trainer.model.state_dict()
        filtered_state_dict = {}
        for k, v in trainer.model.named_parameters():
            if v.requires_grad:
                filtered_state_dict[k] = state_dict[k]
        trainer.model.save_pretrained(training_args.output_dir, 
            state_dict=filtered_state_dict)
        """
    else:
        trainer.save_model(training_args.output_dir)    
    #tokenizer.save_pretrained(training_args.output_dir)
    
if __name__ == "__main__":
    args = parse_args()
    train(args)

