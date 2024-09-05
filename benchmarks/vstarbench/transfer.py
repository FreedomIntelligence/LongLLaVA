import json
import os
import cv2
from PIL import Image
from tqdm import tqdm
        

file_path = './benchmarks/vstarbench/test_questions.jsonl'

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析每一行的 JSON 数据
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data


data = read_jsonl(file_path)

# 遍历数据并修改每个项
for item in tqdm(data):
    image = item["image"]
    item['image'] = f'./benchmarks/vstarbench/{image}'
    item['question'] = item['text']
    item['source'] = item["category"]
    item['answer_index'] = item["label"]


with open('./benchmarks/vstarbench/vstartest.json', 'w', encoding='utf-8') as output:
    json.dump(data, output, ensure_ascii=False, indent=4)

