import json
import math
from datetime import datetime
from argparse import ArgumentParser

import json, os, re
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
import torch

import sys

from utils import get_dataloader, get_worker_class, LongContextBenchmarkDataset
# from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


import pdb


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='data/MLBench')
    parser.add_argument('--dataset_names', default='sample.json', help='Comma-separated list of dataset names')
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--bsz', default=1, type=int)
    parser.add_argument('--batch-image', default=1, type=int)
    parser.add_argument('--num_chunks', default=1, type=int)
    parser.add_argument('--chunk_idx', default=0, type=int)
    parser.add_argument('--model_configs', default='configs/model_configs.yaml')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument("--patchStrategy", type=str, default='norm')

    args = parser.parse_args()

    args.dataset_names = args.dataset_names.split(',')
    args.output_pths = {}
    for dataset_name in args.dataset_names:
        output_pth = os.path.join(args.output_dir, f"{args.model_name}/{dataset_name}/pred_{args.num_chunks}_{args.chunk_idx}.jsonl")
        args.output_pths[dataset_name] = output_pth
        os.makedirs(os.path.dirname(output_pth), exist_ok=True)

    return args


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def save(results, args, dataset_name):
    if os.path.exists(args.output_pths[dataset_name]):
        if not args.overwrite:
            print(f'{args.output_pths[dataset_name]} exists. Please pass `overwrite=True` to avoid unwanted overwriting.')
            exit(0)
    with open(args.output_pths[dataset_name], 'w') as f:
        for line in results:
            f.write(json.dumps(line) + '\n')
    # json.dump(results, open(args.output_pth, 'w'), ensure_ascii=False, indent=4)

def main(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    from visionjamba.eval.chatbot import Chatbot
    args.model_dir = args.model_path

    bot = Chatbot(args)

    ######################### Loading Data #########################
    for dataset_name in args.dataset_names:
        data_dir = args.data_dir
        # dataset_name = args.dataset_name
        dataset_dir = os.path.join(data_dir, dataset_name)
        img_dir  = os.path.join(dataset_dir, 'images')

        core_annotation = json.load(open(os.path.join(dataset_dir, f'{dataset_name}.json')))
        # split data by images number
        questions = get_chunk(core_annotation['data'], args.num_chunks, args.chunk_idx)
        # answers_file = os.path.expanduser(args.answers_file)
        # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        # ans_file = open(answers_file, "a")
        # data_dict = split_data(core_annotation['data'])
        ################################################################
        prediction_results = []
        for line in tqdm(questions):
            bot.clear_history()

            idx = line["sample_id"]
            image_file = line['task_instance']["images_path"]
            task_instruction = core_annotation['meta_data']['task_instruction'][line['task_instruction_id']]
            qs = line['task_instance']["context"]
            cleaned_text = re.sub(r'\{(image|table)#\d+\}', '<image>', qs)
            # cleaned_text = re.sub(r'\{image#\d+\}', '<image>', qs)
            context = task_instruction + '\n' + cleaned_text
            if 'choice_list' in line['task_instance'].keys():
                # choice_str = '\nChoice list: [\''+ \
                #     '\', \''.join(ann['task_instance']['choice_list']) + \
                #     '\']. \nYour answer is: '
                choice_str = '\nChoice list: \n'
                choice_str += '\n'.join([(f'{chr(65+idx)}. ' if 'GPR1200' != core_annotation['meta_data']['dataset'] else '') + f'{item}'
                    for idx, item in enumerate(line['task_instance']['choice_list'])])
                choice_str += '\nYour answer is: '
                context += choice_str

            images_path = []
            for image in image_file:
                images_path.append(os.path.join(img_dir, image))

            # qs = "<image>\n" + qs
            outputs = bot.chat(text=context, images=images_path)
            # import pdb; pdb.set_trace()

            line['pred_response'] = outputs
            line['gt_response'] = line['response']
            line['task_instance']['context'] = context
            prediction_results.append(line)

        save(prediction_results, args, dataset_name)


if __name__ == '__main__':
    args = parse_args()
    main(args)
