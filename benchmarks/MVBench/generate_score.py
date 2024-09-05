import re
import argparse
import json
import os
from collections import defaultdict

def extract_and_choose_answer(pattern, model_answer):
    matches = re.findall(pattern, model_answer)
    option_count = {}
    for match in matches:
        option_count[match.upper()] = option_count.get(match.upper(), 0) + 1

    if not option_count:
        loose_pattern = r'[A-F]'
        if pattern == loose_pattern:
            return None
        else:
            return extract_and_choose_answer(loose_pattern, model_answer)
        
    max_count = max(option_count.values())
    max_options = [option for option, count in option_count.items() if count == max_count]
    return max_options[0]

def generate_score(result_path, score_path):
    json_objects = []
    for filename in os.listdir(result_path):
        print(filename)
        if filename.endswith('.json'):
            file_path = os.path.join(result_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = [json.loads(line) for line in file]
                json_objects.extend(data)
        
    all = defaultdict(int)
    right = defaultdict(int)
    accuracy_dict = defaultdict(float)
    incorrect_items = []
    
    print(f'****Total:{len(json_objects)}****')
    debug = True
    for item in json_objects:
        source = item["source"]
        answer = item["model_answer"][0]
        all[source] += 1  
        pattern = r'[（\(]([A-Fa-f])[）\)]'
        extract_answer = extract_and_choose_answer(pattern, answer)
        if debug:
            debug = False
            print(f'extract_answer:{extract_answer}')
            print(answer)
            right_answer = item['answer_index']
            print(f'right_answer:{right_answer}')
        if extract_answer:
            if item['answer_index'] == extract_answer:
                right[source] += 1
            else:
                incorrect_items.append(item)
        else:
            if item['answer'] in answer:
                right[source] += 1
            else:
                incorrect_items.append(item)
                
    print(f'all:{all}')
    print(f'right:{right}')        
                
    total_accuracy = 0.0
    for key in right:
        accuracy_dict[key] = right[key] / all[key]
        total_accuracy += accuracy_dict[key]
        
    overall_average = total_accuracy / len(right) if right else 0.0

    with open(score_path, "w", encoding="utf8") as f:
        json.dump({"individual_accuracies": accuracy_dict, "overall_average": overall_average}, f, indent=4)
    
    print(f'***********score_result save in {score_path}*************')

    incorrect_path = os.path.join(os.path.dirname(score_path), "incorrect_items.json")
    with open(incorrect_path, "w", encoding="utf8") as f:
        json.dump(incorrect_items, f, indent=4)
    
    print(f'***********incorrect_items save in {incorrect_path}*************')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", '-a', type=str, help="path to the output data")
    parser.add_argument("--score_path", '-o', type=str, help="path to the score")
    args = parser.parse_args()
    generate_score(args.output_path, args.score_path)
