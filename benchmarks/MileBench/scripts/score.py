import json
import argparse
import os
import pandas as pd
from collections import defaultdict

def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = round(sum(d[key] for d in dict_list) / len(dict_list), 5)*100

    return mean_dict

def main(args):

    ########################## Set Dataset Taxonomy ##########################
    dataset_list={
        'Realistic Temporal': [
            # 'action_localization', 'action_prediction', 'action_sequence',
            # 'character_order', 'counterfactual_inference', 'egocentric_navigation',
            # 'moving_attribute', 'moving_direction', 'object_existence', 'object_interaction',
            # 'object_shuffle', 'scene_transition', 'state_change',
            "action_localization",
            "action_prediction",
            "action_sequence",
            "character_order",
            "counterfactual_inference",
            "egocentric_navigation",
            "moving_attribute",
            "moving_direction",
            "object_existence",
            "object_interaction",
            "object_shuffle",
            "scene_transition",
            "state_change"
        ],
        'Realistic Semantic': [
            'ALFRED','CLEVR-Change','DocVQA','IEdit','MMCoQA','MultiModalQA',
            'nuscenes','OCR-VQA','SlideVQA','Spot-the-Diff','TQA','WebQA','WikiVQA'
        ],
        'Diagnostic': ['TextNeedleInAHaystack','ImageNeedleInAHaystack','GPR1200',]
    }

    ########################## Collect Evaluation Result ##########################
    result_dir = args.result_dir

    result = {}
    for model_name in args.models:
        print(f'Collecting result of {model_name}...')
        # print(f"Writing {model_name} result to {result_dir}")
        model_result = {}
        for task_name, dataset_names in dataset_list.items():
            task_result = {}
            if not dataset_names:   # TODO
                continue
            for dataset in dataset_names:
                try:
                    eval_path = os.path.join(result_dir, model_name, dataset, 'eval.json')
                    if not os.path.exists(eval_path):
                        # raise ValueError(f'{model_name}--{dataset}  No evaluation file found')
                        print(f'\t{model_name}--{dataset}  No evaluation file found')
                        task_result[dataset] = {}
                        continue
                    dataset_result = json.load(open(eval_path))
                except Exception as e:
                    print(eval_path)
                task_result[dataset] = dataset_result
            # print(task_result)
            model_result[task_name] = task_result
            # task_result = dict_mean(task_result_list)
            # print(f'{model_name}--{task_name}\n{task_result}')

        # with open(os.path.join(result_dir, 'score.txt'), 'w') as f:
        #     f.write(f'{model_name}--{task}  {task_result}')
        result[model_name] = model_result

    # TODO: by image_quantity_level, by task, by...

    ########################## Save Result ##########################
    json.dump(
        result,
        open(os.path.join(result_dir, 'result.json'), 'w'),
        ensure_ascii=False, indent=4)

    # Function to parse JSON and create a dataset for DataFrame
    def parse_json_to_df(data):
        parsed_data = []
        try:
            for model, tasks in data.items():
                model_data = {'Model': model}
                for task, datasets in tasks.items():
                    for dataset, metrics in datasets.items():
                        for metric, value in metrics.items():
                            if metric not in ["image_quantity_level-Accuracy", "image_quantity_level-Result", "Diff-Accuracy"]:  # Ignore image_quantity_level-Accuracy
                                model_data[f"{dataset} ({metric})"] = round(value*100, 2)
                parsed_data.append(model_data)
        except Exception as e:
            print(e, value)
        return pd.DataFrame(parsed_data)

    # Convert JSON to DataFrame & Save to CSV
    df = parse_json_to_df(result)
    df.to_csv(os.path.join(result_dir, 'result.csv'), index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=str, required=True)
    parser.add_argument("--models", type=str, nargs='+', help="list of models")
    args = parser.parse_args()
    main(args)

'''
# score 1 model
python score_lcvlm.py \
    --result-dir outputs \
    --models llava-v1.5-7b-trim
python score_lcvlm.py \
    --result-dir outputs_combine_1 \
    --models mantis

# outputs:
python score_lcvlm.py \
    --result-dir outputs \
    --models allava yi-vl-6b cheetor qwen-vl-chat llava-v1.5-7b \
        minigpt-v2 vila llava-v1.6-vicuna-7b openflamingo llava-v1.5-13b \
        llava-v1.6-vicuna-13b video-llama2 valley video-chat2 llama-vid lwm

# outputs_combine_1:
python score_lcvlm.py \
    --result-dir outputs_combine_1 \
    --models allava yi-vl-6b cheetor qwen-vl-chat llava-v1.5-7b \
        minigpt-v2 vila llava-v1.6-vicuna-7b openflamingo llava-v1.5-13b \
        llava-v1.6-vicuna-13b video-llama2 valley video-chat2 llama-vid lwm

# outputs_adv:
python score_lcvlm.py \
    --result-dir outputs_adv_sdj \
    --models qwen-vl-chat cheetor openflamingo vila
    

python score.py \
    --result-dir /wangbenyou/xidong/VisionJamba/benchmarks/MileBench/answers \
    --models ckpt40PoolConcat2DAgain

'''

