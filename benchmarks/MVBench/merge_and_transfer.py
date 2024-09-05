base_dir = '/wangbenyou/xidong/VisionJamba/benchmarks/MVBench'
data_list_all = {
    "action_sequence.json": f"{base_dir}/video/star/Charades_v1_480/", # has start & end
    "action_prediction.json": f"{base_dir}/video/star/Charades_v1_480/", # has start & end
    "action_antonym.json": f"{base_dir}/video/ssv2_video/",
    "fine_grained_action.json": f"{base_dir}/video/Moments_in_Time_Raw/videos/",
    "unexpected_action.json": f"{base_dir}/video/FunQA_test/test/",
    "object_existence.json": f"{base_dir}/video/clevrer/video_validation/",
    "object_interaction.json": f"{base_dir}/video/star/Charades_v1_480/", # has start & end
    "object_shuffle.json": f"{base_dir}/video/perception/videos/",
    "moving_direction.json": f"{base_dir}/video/clevrer/video_validation/",
    "action_localization.json": f"{base_dir}/video/sta/sta_video/",  # has start & end
    "scene_transition.json": f"{base_dir}/video/scene_qa/video/",
    "action_count.json": f"{base_dir}/video/perception/videos/",
    "moving_count.json": f"{base_dir}/video/clevrer/video_validation/",
    "moving_attribute.json": f"{base_dir}/video/clevrer/video_validation/",
    "state_change.json": f"{base_dir}/video/perception/videos/",
    "fine_grained_pose.json": f"{base_dir}/video/nturgbd/",
    "character_order.json": f"{base_dir}/video/perception/videos/",
    "egocentric_navigation.json": f"{base_dir}/video/vlnqa/",
    "episodic_reasoning.json": f"{base_dir}/video/tvqa/frames_fps3_hq/",  # has start & end, read frame
    "counterfactual_inference.json": f"{base_dir}/video/clevrer/video_validation/",
}

import os
import json
import cv2  # 用于视频处理
import shutil

def extract_frames(video, t=0.5):
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


def is_video_file(path):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']
    return any(path.lower().endswith(ext) for ext in video_extensions)

def get_files_from_directory(directory):
    files = [os.path.join(directory, f) for f in sorted(os.listdir(directory)) if os.path.isfile(os.path.join(directory, f))]
    return files

def sample_files(files, sample_size=200):
    if len(files) <= sample_size:
        return files
    interval = len(files) / sample_size
    sampled_files = [files[int(i * interval)] for i in range(sample_size)]
    return sampled_files

choice_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# 处理视频裁剪或图片范围
processed_path = '/wangbenyou/xidong/VisionJamba/benchmarks/MVBench/video/processed'
os.makedirs(processed_path, exist_ok=True)
                        
                        
def trim_video(video_path, start_time, end_time, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video file: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1
    
    cap.release()
    out.release()
    print(f"Video saved to: {output_path}")

def merge_json_files(input_folder, output_file):
    merged_data = []

    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                for item in data:
                    item['source'] = filename
                    video_name = item["video"]
                    item["video"] = data_list_all[item['source']] + video_name
                    
                    if 'start' in item and 'end' in item and 'fps' not in item:
                        start_time = item['start']
                        end_time = item['end']
                        output_video_path = os.path.join(processed_path, f'processed_{video_name}')
                        trim_video(item["video"], start_time, end_time, output_video_path)
                        item["ori_video"] = item['video']
                        item["video"] = output_video_path
                    elif 'start' in item and 'end' in item and 'fps' in item:
                        start_frame = int(item['start'] * item['fps'])
                        end_frame = int(item['end'] * item['fps'])
                        dir = item["video"]
                        print(f'processing {dir}')
                        image_files = get_files_from_directory(item["video"])
                        selected_images = image_files[start_frame:end_frame + 1]
                        item["video"] = selected_images
                        
                    if 'fps' not in item:
                        item['video'] = extract_frames(item['video'])
                    
                    new_question = "You must choose your answer from the Choice List.\n" + item['question'] + "\nChoice list: \n"
                    for idx, candidate in enumerate(item['candidates']):
                        new_question += f"{choice_list[idx]}. {candidate}\n"
                    new_question += "Your answer is: "
                    
                    answer_index = item['candidates'].index(item['answer'])
                    new_answer = f"{choice_list[answer_index]}"
                    
                    item['question'] = new_question.strip()
                    item['answer_index'] = new_answer
                    
                merged_data.extend(data)
        
    with open(output_file, 'w', encoding='utf-8') as output:
        json.dump(merged_data, output, ensure_ascii=False, indent=4)

# 使用示例
input_folder = '/wangbenyou/xidong/VisionJamba/benchmarks/MVBench/json'
output_file = '/wangbenyou/xidong/VisionJamba/benchmarks/MVBench/merged_new.json'
merge_json_files(input_folder, output_file)
