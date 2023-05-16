# coding=utf-8

import argparse
import os
import platform
import signal
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModel, AutoTokenizer

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

def build_prompt(history):
    prompt = "欢迎使用 ChatGLM-6B 模型, 输入内容即可进行对话, clear 清空对话历史, stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户: {query}"
        prompt += f"\n\nChatGLM-6B: {response}"
    return prompt

def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="THUDM/chatglm-6b")
    
    parser.add_argument("--lora_checkpoint", type=str, default=None)
    parser.add_argument("--load_in_8bit", action="store_true")

    parser.add_argument("--ptuning_checkpoint", type=str, default=None)
    parser.add_argument("--pre_seq_len", type=int, default=None)
    parser.add_argument("--prefix_projection", action="store_true")
    
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=0.95)
 
    parser.add_argument("--quantization_bit", type=int, choices=[4, 8], default=None)
    parser.add_argument("--no_history", action="store_true")
    args = parser.parse_args()
    return args

def load_model(args):
    global model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True)

    if args.ptuning_checkpoint is not None:
        config = AutoConfig.from_pretrained(args.model_name_or_path,
            trust_remote_code=True)
        config.pre_seq_len = args.pre_seq_len
        config.prefix_projection = args.prefix_projection

        print(f"Loading prefix_encoder weight from {args.ptuning_checkpoint}")
        model = AutoModel.from_pretrained(args.model_name_or_path,
            config=config, trust_remote_code=True)
        prefix_state_dict = torch.load(os.path.join(
            args.ptuning_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
        
        if args.quantization_bit is not None:
            print(f"Quantized to {args.quantization_bit} bit")
            model = model.quantize(args.quantization_bit)

        model = model.half().cuda()
        model.transformer.prefix_encoder.float().cuda()
    
    elif args.lora_checkpoint is not None:
        if args.load_in_8bit:
            model = AutoModel.from_pretrained(args.model_name_or_path,
                load_in_8bit=True, device_map="auto", trust_remote_code=True)
        else:
            model = AutoModel.from_pretrained(args.model_name_or_path,
                trust_remote_code=True)
            model = model.half().cuda()

        model = PeftModel.from_pretrained(model.cuda(), args.lora_checkpoint)
    else:
        model = AutoModel.from_pretrained(args.model_name_or_path,
            trust_remote_code=True)
        if args.quantization_bit is not None:
            print(f"Quantized to {args.quantization_bit} bit")
            model = model.quantize(args.quantization_bit)
        model = model.half().cuda()

    model = model.eval()

def main(args):
    load_model(args)
    history = []
    global stop_stream
    print("欢迎使用 ChatGLM-6B 模型, 输入内容即可进行对话, clear 清空对话历史, stop 终止程序")
    while True:
        query = input("\n用户: ")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用 ChatGLM-6B 模型, 输入内容即可进行对话, clear 清空对话历史, stop 终止程序")
            continue
        count = 0
        if args.no_history:
            history = []
        for response, history in model.stream_chat(tokenizer, query, history=history,
            max_length=args.max_length, top_p=args.top_p, temperature=args.temperature):
            if stop_stream:
                stop_stream = False
                break
            else:
                count += 1
                if count % 8 == 0:
                    os.system(clear_command)
                    print(build_prompt(history), flush=True)
                    signal.signal(signal.SIGINT, signal_handler)
        os.system(clear_command)
        print(build_prompt(history), flush=True)

if __name__ == "__main__":
    args = parse_args()
    main(args)

