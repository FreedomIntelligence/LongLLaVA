import torch
import os
import random
import sys
import numpy as np
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init

from argparse import ArgumentParser

from eval.utils.data_utils import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG
from eval.utils.model_utils import call_llava_engine_df, llava_image_processor
from eval.utils.eval_utils import parse_multi_choice_response, parse_open_response


from PIL import Image
import math

import shortuuid, json
import pdb


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def run_model(args, samples, model, call_model_engine_fn=None, tokenizer=None, processor=None):
    out_samples = dict()
    with torch.no_grad():
        for sample in tqdm(samples):
            response = call_model_engine_fn(args, sample, model, tokenizer, processor)

            if sample['question_type'] == 'multiple-choice':
                pred_ans = parse_multi_choice_response(response, sample['all_choices'], sample['index2ans'])
            else:  # open question
                pred_ans = response
            out_samples[sample['id']] = pred_ans
    return out_samples


def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default='llava1.5_13b_val.json',
                        help='name of saved json')
    parser.add_argument('--config_path', type=str, default="configs/llava1.5.yaml")
    parser.add_argument('--data_path', type=str, default="MMMU/MMMU") # hf dataset path.
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    # parser.add_argument('--model_path', type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--patchStrategy", type=str, default='norm')
    parser.add_argument("--chunk-idx", type=int, default=0)

    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    print('llava_initializing...')
    processor = None
    # call_model_engine = call_llava_engine_df
    # vis_process_func = llava_image_processor

    # load config and process to one value
    args.config = load_yaml(args.config_path)
    for key, value in args.config.items():
        if key != 'eval_params' and type(value) == list:
            assert len(value) == 1, 'key {} has more than one value'.format(key)
            args.config[key] = value[0]

    # run for each subject
    sub_dataset_list = []
    for subject in tqdm(CAT_SHORT2LONG.values()):
        print(f'loading {subject}')
        sub_dataset = load_dataset(args.data_path, subject, split=args.split)
        sub_dataset_list.append(sub_dataset)

    # merge all dataset
    dataset = concatenate_datasets(sub_dataset_list)


    # # load model
    # model_name = get_model_name_from_path(args.model_path)
    # tokenizer, model, vis_processors, _ = load_pretrained_model(args.model_path, None,
    #                                                             model_name)

    samples = []
    for sample in tqdm(dataset, desc='processing samples'):
        sample = process_single_sample(sample)

        sample = construct_prompt(sample, args.config)
        # pdb.set_trace()
        # if sample['image']: # PIL
        #     sample['image'] = vis_process_func(sample['image'], vis_processors).to(device)
        samples.append(sample)


    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    from visionjamba.eval.chatbot import Chatbot

    args.model_dir = args.model_path
    
    bot = Chatbot(args)

    # bot.gen_kwargs['max_new_tokens'] = 20

    questions = get_chunk(samples, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    ans_file = open(answers_file, "a")
    for line in tqdm(questions):
        bot.clear_history()
        
        idx = line["id"]
        image = line["image"].convert('RGB')
        q = line["final_input_prompt"]
        cur_prompt = q.replace('<image 1>', '').strip()
        cur_prompt = "<image>\n" + cur_prompt
        # image = Image.open(os.path.join(args.image_folder, image_file)) # ???
        
        try:
            response = bot.chat(text=cur_prompt, images=image)
        except Exception as e:
            pdb.set_trace()
            print(f'skipping {idx}. Error: {e}')
            continue
        if sample['question_type'] == 'multiple-choice':
            pred_ans = parse_multi_choice_response(response, sample['all_choices'], sample['index2ans'])
        else:  # open question
            pred_ans = response
        # out_samples[sample['id']] = pred_ans
        
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"id": idx,
                                   "prompt": cur_prompt,
                                   "raw_response": response,
                                   "parsed": pred_ans, # this is what we want
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()

    # run ex
    # out_samples = run_model(args, samples, model, call_model_engine, tokenizer, processor)
        
    

    # save_json(args.output_path, out_samples)
    # metric_dict.update({"num_example": len(out_samples)})
    # save_json(save_result_path, metric_dict)


if __name__ == '__main__':
    main()

