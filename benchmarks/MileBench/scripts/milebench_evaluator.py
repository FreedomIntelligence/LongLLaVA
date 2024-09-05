import os
import json
import argparse


action_understandings = [
    "action_localization",
    "action_prediction",
    "action_sequence"
]

object_understandings = [
    "moving_attribute",
    "object_existence",
    "object_interaction",
    "object_shuffle"
]

visual_navigation = [
    'egocentric_navigation',
    'moving_direction'
]

counter_factual = [
    'character_order',
    'counterfactual_inference',
    'scene_transition',
    'state_change'
]

knowledge_grounded = [
    'MultiModalQA',
    'TQA',
    'WebQA',
    'WikiVQA'
]

text_rich = [
    'DocVQA',
    'OCR-VQA',
    'SlideVQA'
]

visual_relation = [
    'CLEVR-Change',
    'IEdit',
    'Spot-the-Diff'
]

dialogue = [
    'ALFRED',
    'MMCoQA'
]

space_understanding = [
    'nuscenes'
]

TextNeedleInAHaystack = [
    'TextNeedleInAHaystack'
]

ImageNeedleInAHaystack = [
    'ImageNeedleInAHaystack'
]

GPR1200 = [
    'GPR1200'
]

all_test_tasks = action_understandings + object_understandings + visual_navigation + counter_factual + knowledge_grounded + text_rich + visual_relation + dialogue + space_understanding + TextNeedleInAHaystack + ImageNeedleInAHaystack + GPR1200
realistic_tasks = action_understandings + object_understandings + visual_navigation + counter_factual + knowledge_grounded + text_rich + visual_relation + dialogue + space_understanding
diagnostic_tasks = TextNeedleInAHaystack + ImageNeedleInAHaystack + GPR1200

task_groups = {
    "action_understandings": action_understandings,
    "object_understandings": object_understandings,
    "visual_navigation": visual_navigation,
    "counter_factual": counter_factual,
    "knowledge_grounded": knowledge_grounded,
    "text_rich": text_rich,
    "visual_relation": visual_relation,
    "dialogue": dialogue,
    "space_understanding": space_understanding,
    "TextNeedleInAHaystack": TextNeedleInAHaystack,
    "ImageNeedleInAHaystack": ImageNeedleInAHaystack,
    "GPR1200": GPR1200
}

def get_score(data_dir, dataset_name):
    path = os.path.join(data_dir, dataset_name, 'eval.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")
    with open(path, 'r') as f:
        data = json.load(f)
    if "Accuracy" in data:
        return data['Accuracy']
    elif "Rouge-L f" in data:
        return data['Rouge-L f']
    else:
        raise ValueError(f"Score not found in {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='outputs')
    args = parser.parse_args()

    scores = {}

    for task in all_test_tasks:
        score = get_score(args.data_dir, task)
        scores[task] = score
    
    task_groups_scores = {}

    for group, tasks in task_groups.items():
        group_score = sum(scores[task] for task in tasks) / len(tasks)
        task_groups_scores[group] = group_score
    
    total_avg_score = sum(task_groups_scores.values()) / len(task_groups_scores)

    real_avg_score = sum(scores[group] for group in realistic_tasks) / len(realistic_tasks)
    diag_avg_score = sum(scores[group] for group in diagnostic_tasks) / len(diagnostic_tasks)

    with open(args.output_path, 'a') as f:
        f.write(f"MileBench (Total Acc)\t{total_avg_score}\n")
        f.write(f"MileBench (Realistic Tasks Acc)\t{real_avg_score}\n")
        f.write(f"MileBench (Diagnostic Tasks Acc)\t{diag_avg_score}\n")