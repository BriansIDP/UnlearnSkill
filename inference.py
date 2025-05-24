import os
import random
import argparse
import math
import pickle
import time
import json
from collections import OrderedDict

from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def logging(s, logfile, logging_=True, log_=True):
    if logging_:
        print(s)
    if log_:
        with open(logfile, 'a+') as f_log:
            f_log.write(s + '\n')


def main(args):
    # Load model
    # if os.path.exists(args.model_path):
    #     with open(os.path.join(args.model_path, "model_config.json")) as fin:
    #         train_args = json.load(fin)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir="/data/milsrg1/huggingface/cache/gs534/cache")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        cache_dir="/data/milsrg1/huggingface/cache/gs534/cache",
    )
    model = model.to(device)
    model.eval()

    with open(args.testfile) as fin:
        testdata = json.load(fin)

    results = []
    for datapiece in tqdm(testdata):
        messages = [
            {"role": "system", "content": "You are helpful and faithful personal assistant"},
            {"role": "user", "content": datapiece["question"]},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        import pdb; pdb.set_trace()
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        datapiece["answer"] = response
        results.append(datapiece)

    assert args.outfile.endswith("json")
    with open(args.outfile, "w") as fout:
        json.dump(results, fout, indent=4)

if __name__ == "__main__":
    ## Parameter groups
    parser = argparse.ArgumentParser(description="LLM finetuning")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./hf_models",
        help="Path to the model file",
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="",
        help="Checkpoint of the model file",
    )
    parser.add_argument(
        "--testfile",
        type=str,
        default="dataset/gt_nbest_sel.json",
        help="Path to the model file",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default='./log.txt',
        help="Path to the log file",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default='./output.json',
        help="output file",
    )
    parser.add_argument(
        "--origmodel",
        action='store_true',
        help="Use original LLM",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=1,
        help="Number of samples to draw",
    )
    args = parser.parse_args()
    main(args)