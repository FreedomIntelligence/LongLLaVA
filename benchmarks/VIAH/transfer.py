import json
import os
import cv2
from PIL import Image
from tqdm import tqdm

def extract_frames(video, t=1.0):
    path_parts = video.split('/')
    base_name = os.path.splitext(path_parts[-1])[0]
    output_dir = "./benchmarks/VIAH/images"
    subdir = os.path.join(output_dir, base_name)
    os.makedirs(subdir, exist_ok=True)  

    try:
        cap = cv2.VideoCapture(video)
    except Exception as e:
        print("-" * 50)
        print(f"Error opening video file {video}: {e}")
        return []

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            cap.release()
            return []
    except Exception as e:
        print("-" * 50)
        print(f"Error getting FPS or frame count from {video}: {e}")
        cap.release()
        return []

    try:
        frame_interval = max(int(fps * t), 1)
        frameList = []
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                split_path = os.path.join(subdir, f'{count}.jpg')
                cv2.imwrite(split_path, frame) 
                frameList.append(split_path)
            count += 1
        cap.release()

    except Exception as e:
        print("-" * 50)
        print(f"Error extracting keyframes from {video}: {e}")

    return frameList

        
        

file_path = './VIAH/VNBench-main-4try.json'


with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

choice_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

source_dict = {}


for item in tqdm(data):
    video_id = item['video']
    item['video'] = f'./benchmarks/VIAH/video/{video_id[1:]}'
    item['video'] = extract_frames(item['video'])
    new_question = "You must choose your answer from the Choice List.\n" + item['question'] + "\nChoice list: \n"
    for idx, candidate in enumerate(item['options']):
        new_question += f"{choice_list[idx]}. {candidate}\n"
    new_question += "Your answer is: "
    item['ori_question'] = item['question']
    item['question'] = new_question
    item['source'] = item["type"]
    item['answer_index'] = item["gt_option"]


with open('./benchmarks/VIAH/VNBench-test.json', 'w', encoding='utf-8') as output:
    json.dump(data, output, ensure_ascii=False, indent=4)

