import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer
import random
import statistics

IMAGE_TOKEN = 144
MAX_LENTH = 128000

with open('VQA.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

random.shuffle(data)

FinalList = []
TmpDict = {
    'image': [],
    'conversations': []
}
ALL_TEXT = True
TmpLength = 0
tokenizer = AutoTokenizer.from_pretrained('./ckpt/Jamba-v0.1', trust_remote_code=True)
EOS_TOKEN = tokenizer.eos_token
ALL_Lenths = []
Skip_item = 0
Skip_Alltext = 0
BeforeItemNum = len(data)
for item in tqdm(data):
    if 'image' in item:
        ALL_TEXT = False
        item_lenth = 0
        for conversation in item['conversations']:
            textvalue = conversation['value']
            textvalue = textvalue.replace('<image>', '')
            item_lenth += len(tokenizer.encode(textvalue))
            item_lenth += len(item['image']) * IMAGE_TOKEN
        if TmpLength + item_lenth > MAX_LENTH:
            if not ALL_TEXT:
                if len(TmpDict['image']) > 0 and len(TmpDict['conversations']) > 0:
                    FinalList.append(TmpDict)
                    ALL_Lenths.append(TmpLength)
                else:
                    Skip_item += 1
            else:
                Skip_Alltext += 1
            TmpDict = {
                'image': [],
                'conversations': []
            }
            ALL_TEXT = True
            TmpLength = 0
        else:
            TmpDict['image'].extend(item['image'])
            item['conversations'][-1]['value'] += EOS_TOKEN
            TmpDict['conversations'].extend(item['conversations'])
            TmpLength += item_lenth
    else:
        item_lenth = 0
        for conversation in item['conversations']:
            textvalue = conversation['value']
            item_lenth += len(tokenizer.encode(textvalue))
        if TmpLength + item_lenth > MAX_LENTH:
            if not ALL_TEXT and len(TmpDict['image'])> 0 and len(TmpDict['conversations']) > 0:
                FinalList.append(TmpDict)
                ALL_Lenths.append(TmpLength)
            else:
                Skip_Alltext += 1
            TmpDict = {
                'image': [],
                'conversations': []
            }
            ALL_TEXT = True
            TmpLength = 0
        else:
            item['conversations'][-1]['value'] += EOS_TOKEN
            TmpDict['conversations'].extend(item['conversations'])
            TmpLength += item_lenth

if not ALL_TEXT and len(TmpDict['image'])> 0:
    FinalList.append(TmpDict)      


longest = max(ALL_Lenths)
shortest = min(ALL_Lenths)
median_value = statistics.median(ALL_Lenths)
average = statistics.mean(ALL_Lenths)
AftItemNum = len(ALL_Lenths)

Statistic = {
    "original_num": BeforeItemNum,
    "longest": longest,
    "shortest": shortest,
    "median": median_value,
    "average": average,
    "Skip_item": Skip_item,
    "Concat_num": AftItemNum,
    "Skip_Alltext": Skip_Alltext
}

with open(f'./data/VQAStatistic{MAX_LENTH}_{IMAGE_TOKEN}.json', 'w', encoding='utf-8') as file:
    json.dump(Statistic, file, ensure_ascii=False, indent=4)

with open(f'./VQA{MAX_LENTH}_{IMAGE_TOKEN}.json', 'w', encoding='utf-8') as file:
    json.dump(FinalList, file, ensure_ascii=False, indent=4)
        
    