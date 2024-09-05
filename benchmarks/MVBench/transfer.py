import json
import os
import cv2
from PIL import Image
from tqdm import tqdm

def extract_frames(video, t=0.25):
    path_parts = video.split('/')
    base_name = os.path.splitext(path_parts[-1])[0]
    output_dir = "/wangbenyou/xidong/VisionJamba/benchmarks/MVBench/images"
    subdir = os.path.join(output_dir, base_name)
    os.makedirs(subdir, exist_ok=True)  # 确保子目录存在

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
                cv2.imwrite(split_path, frame)  # 直接使用 cv2.imwrite 保存帧
                frameList.append(split_path)
            count += 1
        cap.release()

    except Exception as e:
        print("-" * 50)
        print(f"Error extracting keyframes from {video}: {e}")
        
    return frameList
        
        
def merge_json_files(file_path, output_file):
    merged_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    for item in tqdm(data):                     
        if 'fps' not in item:
            video = item['video']
            if type(video) is list:
                if len(video) == 1:
                    print(video)
                    video = video[0]
                else:
                    continue
            item['video'] = extract_frames(video)    
        merged_data.extend(data)
        
    with open(output_file, 'w', encoding='utf-8') as output:
        json.dump(merged_data, output, ensure_ascii=False, indent=4)
        
        
# 使用示例
file_path = '/wangbenyou/xidong/VisionJamba/benchmarks/MVBench/merged_new.json'
output_file = '/wangbenyou/xidong/VisionJamba/benchmarks/MVBench/merged_new_path.json'
merge_json_files(file_path, output_file)