import re
from rouge import Rouge
import argparse
import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdb

class Eval:

    def __init__(self):
        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def char(self, index):
        if index < 26:
            return chr(index+65)
        elif index < 52:
            return 'A'+chr(index+65-26)
        else:
            return 'B'+chr(index+65-26-26)

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def process(self, answer):
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = self.processPunctuation(answer)
        answer = answer.strip('\'')
        answer = answer.strip('\"')
        answer = answer.strip().lower()
        return answer

    def get_image_quantity_level(self, sample):
        # 2-5 6-31 32-109
        image_num = len(sample['image'])
        if image_num < 6:
            return 'Few'
        elif image_num > 31:
            return 'Many'
        else:
            return 'Medium'

    def evaluate_rouge(self, predictions, core_json):
        ####################
        # get image_quantity_level
        if len(predictions) != len(core_json['data']):
            raise ValueError(f'There is prediction absent.')
        new_pres = {d['sample_id']: d for d in predictions}
        for sample in core_json['data']:
            # pdb.set_trace()
            new_pres[int(sample['sample_id'])]['image_quantity_level'] = sample['image_quantity_level']
            new_pres[int(sample['sample_id'])]['image'] = sample['task_instance']['images_path']
        for pre in new_pres.values():
            assert 'image_quantity_level' in pre.keys()

        rouge = Rouge()
        acc = {'f': []}
        eval_list = []
        image_quantity_level_cnt = {'Few': [], 'Medium': [], 'Many': []}
        for i, res in enumerate(predictions):
            sample_id = res['sample_id']
            gt_ans = self.process(res["gt_response"])
            pred_ans = self.process(res["pred_response"])
            assert gt_ans != ''
            if pred_ans == '':
                score = 0
            else:
                score = rouge.get_scores(pred_ans, gt_ans)[0]['rouge-l']['f']
            acc['f'].append(score)
            image_quantity_level_cnt[self.get_image_quantity_level(res)].append(score)
            eval_list.append({'id':str(sample_id),'score':str(round(score,3))})
        return {
            'Rouge-L f': np.mean(acc['f']),
            'image_quantity_level-Accuracy': {k: np.mean(v) if len(v)!=0 else 0 for k, v in image_quantity_level_cnt.items()},
            'image_quantity_level-Result': {k: [sum(v), len(v)] for k, v in image_quantity_level_cnt.items()}}, eval_list

    def match_choice(self, text, option):
        '''Return: A B C D...'''

        def preprocess_option_string(option_string):
            # First, preprocess the option text to normalize it
            processed_option = self.process(option_string)

            # Then, escape any special regex characters in the processed option text
            # List of regex special characters that need to be escaped
            special_chars = ["\\", ".", "^", "$", "*", "+", "?", "{", "}", "[", "]", "|", "(", ")"]
            # Escape the special characters by prefixing them with a backslash
            for char in special_chars:
                if char in processed_option:
                    processed_option = processed_option.replace(char, "\\" + char)
            # escaped_option = escape_special_chars(processed_option)
            return processed_option
        # import pdb; pdb.set_trace()
        if text == "":
            return 'C'
        try:
            # Maybe start from the head
            # 1. Char+Choice: `A. Blastomycosis`
            option_str = "|".join([preprocess_option_string(f"{k} {v}")for k,v in option.items()])
            option_pattern = rf'({option_str})'
            option_res = re.search(option_pattern, text, re.S)   # NOTE we dont use match_all
            if option_res:
                return (option_res.group(0)[0]).upper()

            # 2. Choice: `Blastomycosis`
            option_str = "|".join([preprocess_option_string(v).replace(' ', '') for k,v in option.items()])
            option_pattern = rf'({option_str})'
            option_res = re.search(option_pattern, text.replace(' ', ''), re.S)   # NOTE we dont use match_all
            if option_res:
                for k, v in option.items():
                    if option_res[0].strip() == preprocess_option_string(v).replace(' ', ''):
                        return k.upper()
            
            # 3. Char: `A` `AB`
            if len(text) in [1,2] and text.upper() in option.keys():
                return text.upper()

            # use gpt extract

        except Exception as e:
            print(f"something wrong during match_choice {text}: {e}")
            return text
        return "".join([i.upper() for i in text if i.upper() in option])

    def judge_multi_choice(self, sample):
        ####################
        sample_id = sample['sample_id']
        gt_ans = sample["gt_response"]
        pred_ans = sample["pred_response"]
        choice_list = sample['choice_list']
        assert gt_ans in choice_list
        # Convert choice_list to a dictionary format expected by match_choice
        option_dict = {self.char(i): choice for i, choice in enumerate(choice_list)}

        # Use match_choice to determine the selected answer from pred_ans
        selected_answer = self.match_choice(pred_ans, option_dict)

        # Check if the selected answer matches the ground truth
        gt_ans_chr = self.char(choice_list.index(sample["gt_response"]))
        if selected_answer == gt_ans_chr:
            return 1, selected_answer
        else:
            return 0, selected_answer

    # NOTE: TFIDF evaluator
    # def judge_multi_choice(self, sample):
    #     sample_id = sample['sample_id']
    #     gt_ans = sample["gt_response"]
    #     pred_ans = sample["pred_response"]
    #     choice_list = sample['choice_list']
    #     if gt_ans not in choice_list:
    #         print(gt_ans)
    #         print(choice_list)
    #     assert gt_ans in choice_list
    #     try:
    #         vectorizer = TfidfVectorizer()
    #         texts = [pred_ans] + choice_list
    #         tfidf_matrix = vectorizer.fit_transform(texts)
    #         cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    #         most_similar_index = cosine_similarities.argmax()
    #         if choice_list[most_similar_index] == gt_ans:
    #             return 1
    #         else:
    #             return 0
    #     except:
    #         if pred_ans == gt_ans:
    #             return 1
    #         else:
    #             return 0

    def process_sample(self, sample):
        sample["gt_response"] = self.process(sample["gt_response"])
        sample["pred_response"] = self.process(sample["pred_response"])
        for i in range(len(sample['choice_list'])):
            sample["choice_list"][i] = self.process(sample["choice_list"][i])

    def evaluate_multichoice(self, predictions, core_json):
        '''
        predictions: raw prediction file output by models
        '''
        # get choice_list & image_quantity_level
        if len(predictions) != len(core_json['data']):
            raise ValueError(f'There is prediction absent. {len(predictions)}!={len(core_json["data"])}')
        new_pres = {d['sample_id']: d for d in predictions}
        for sample in core_json['data']:
            new_pres[int(sample['sample_id'])]['choice_list'] = sample['task_instance']['choice_list']
            new_pres[int(sample['sample_id'])]['image_quantity_level'] = sample['image_quantity_level']
            new_pres[int(sample['sample_id'])]['image'] = sample['task_instance']['images_path']
        for pre in new_pres.values():
            assert 'choice_list' in pre.keys()
            assert 'image_quantity_level' in pre.keys()
        
        correct = 0
        eval_list = []
        image_quantity_level_cnt = {'Few': [], 'Medium': [], 'Many': []}
        for i, sample in enumerate(predictions):
            # Process string
            self.process_sample(sample)
            # Score
            
            score, extracted_answer = self.judge_multi_choice(sample)
            sample['extracted'] = extracted_answer
            sample['result'] = score
            eval_list.append({'id':str(sample['sample_id']), 'score': str(score)})
            correct += score
            image_quantity_level_cnt[self.get_image_quantity_level(sample)].append(score)
        return predictions, {
            'Accuracy': correct/len(predictions),
            'image_quantity_level-Accuracy': {k: np.mean(v) if len(v)!=0 else 0 for k, v in image_quantity_level_cnt.items()},
            'image_quantity_level-Result': {k: [sum(v), len(v)] for k, v in image_quantity_level_cnt.items()}}, eval_list

    def evaluate_needle(self, predictions, core_json, needle=True):
        # get choice_list & image_quantity_level
        if len(predictions) != len(core_json['data']):
            raise ValueError(f'There is prediction absent. {len(predictions)}!={len(core_json["data"])}')
        new_pres = {d['sample_id']: d for d in predictions}
        for sample in core_json['data']:
            new_pres[int(sample['sample_id'])]['image_quantity_level'] = sample['image_quantity_level']
            new_pres[int(sample['sample_id'])]['image'] = sample['task_instance']['images_path']
        for pre in new_pres.values():
            assert 'image_quantity_level' in pre.keys()
        
        correct = 0
        eval_list = []
        image_quantity_level_cnt = {'Few': [], 'Medium': [], 'Many': []}
        for i, sample in enumerate(predictions):
            # Process string
            sample_id = sample['sample_id']
            gt_ans = self.process(sample["gt_response"])
            pred_ans = self.process(sample["pred_response"])
            
            # Score
            if needle:
                score = 1 if gt_ans in pred_ans.split() else 0
            else:
                score = 1 if gt_ans in pred_ans else 0

            sample['result'] = score
            eval_list.append({'id':str(sample['sample_id']), 'score': str(score)})
            correct += score
            image_quantity_level_cnt[self.get_image_quantity_level(sample)].append(score)
        return {
            'Accuracy': correct/len(predictions),
            'image_quantity_level-Accuracy': {k: np.mean(v) if len(v)!=0 else 0 for k, v in image_quantity_level_cnt.items()},
            'image_quantity_level-Result': {k: [sum(v), len(v)] for k, v in image_quantity_level_cnt.items()}}, eval_list


def main(args):

    dataset = args.dataset
    result_dir = args.result_dir
    model_name = result_dir.split('/')[-1]

    core_annotation = json.load(open(os.path.join(args.data_dir, dataset, f'{dataset}-adv.json' if args.adv else f'{dataset}.json')))
    question_type = core_annotation['meta_data']['question_type']

    # Load predictions
    output_dir = os.path.join(result_dir, dataset)
    pred_file_path = os.path.join(output_dir, f'pred_{args.img_num}.json') if args.img_num > 0 else os.path.join(output_dir, 'pred.jsonl')
    if not os.path.exists(pred_file_path):
        print(pred_file_path)
        raise ValueError(f'{model_name}--{dataset} No prediction file found')
    preds = []

    for line in open(pred_file_path):
        preds.append(json.loads(line))
    # preds = json.load(open(pred_file_path))
    # preds = json.load(open(os.path.join(output_dir, 'pred.json')))
    assert preds and len(preds) != 0

    # Get scores
    scorer = Eval()
    if 'NeedleInAHaystack' in dataset or 'MMCoQA' in dataset:
        eval_result, eval_list = \
            scorer.evaluate_needle(preds, core_annotation, needle='NeedleInAHaystack' in dataset)
    elif question_type == 'open-ended':
        eval_result,eval_list = scorer.evaluate_rouge(preds, core_annotation)
    elif question_type == 'multi-choice':
        predictions_with_extracted_answers, eval_result, eval_list = scorer.evaluate_multichoice(preds, core_annotation)
        predictions_with_extracted_answers_path = os.path.join(output_dir, f'pred_with_extracted_{args.img_num}.json') if args.img_num > 0 else os.path.join(output_dir, f'pred_with_extracted.json')
        json.dump(predictions_with_extracted_answers, open(predictions_with_extracted_answers_path,'w'), indent=4)

    else:
        raise ValueError('Dataset not supported')

    print(f"{model_name}:  {dataset}:  {eval_result}")
    eval_output_path = os.path.join(output_dir, f'eval_{args.img_num}.json') if args.img_num > 0 else os.path.join(output_dir, f'eval.json')
    eval_score_output_path = os.path.join(output_dir, f'eval_score_{args.img_num}.json') if args.img_num > 0 else os.path.join(output_dir, f'eval_score.json')

    json.dump(eval_result, open(eval_output_path,'w'))
    json.dump(eval_list, open(eval_score_output_path,'w'), indent=4)
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--result-dir', type=str, required=True)
    parser.add_argument('--adv', action='store_true', help='Use adversarial data for evaluation.')
    parser.add_argument('--img-num', type=int, default=0)
    args = parser.parse_args()

    main(args)

'''
# MC
python evaluate_lcvlm.py \
    --data-dir data/MLBench \
    --dataset DocVQA \
    --result-dir outputs/llama-vid

# OE
python evaluate_lcvlm.py \
    --data-dir data/MLBench \
    --dataset ALFRED \
    --result-dir outputs/llava-v1.5-7b

# OE Needle
python evaluate_lcvlm.py \
    --data-dir data/MLBench \
    --dataset ImageNeedleInAHaystack \
    --result-dir outputs/allava
python evaluate_lcvlm.py \
    --data-dir data/MLBench \
    --dataset MMCoQA \
    --result-dir outputs/allava


# All
EVAL_SCRIPT_PATH=evaluate_lcvlm.py
DATA_DIR=data/MLBench
for model in gpt-4-turbo gemini-1.5-pro-latest-reformat gemini-1.0-pro-vision-001 claude-3-opus-20240229; do
# for model in qwen-vl-chat minigpt-v2 cheetor allava llava-v1.5-7b llama-vid openflamingo video-llama2 valley video-chat2; do
    for dataset_name in GPR1200; do
    # for dataset_name in ALFRED CLEVR-Change DocVQA IEdit MMCoQA MultiModalQA nuscenes OCR-VQA SlideVQA Spot-the-Diff TQA WebQA WikiVQA GPR1200 ImageNeedleInAHaystack TextNeedleInAHaystack; do
        python ${EVAL_SCRIPT_PATH} \
            --data-dir ${DATA_DIR} \
            --dataset ${dataset_name} \
            --result-dir outputs/${model} \
            # >> logs/${model}/${dataset_name}.log
        python ${EVAL_SCRIPT_PATH} \
            --data-dir ${DATA_DIR} \
            --dataset ${dataset_name} \
            --result-dir outputs_combine_1/${model} \
            # >> logs/${model}/${dataset_name}_combine_1.log
    done
done
'''
