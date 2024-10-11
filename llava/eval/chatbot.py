import argparse
import torch

from PIL import Image

from PIL import Image
import cv2
import transformers
import numpy as np
import os
import uuid
if transformers.__version__ > '4.36':
    truncate_inputs = False

from llava.model import *
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_anyres_image, get_model_name_from_path, process_images

class Chatbot():
    def __init__(self, config) -> None:

        # self.gen_kwargs = getattr(config, 'gen_kwargs', {None})

        self.gen_kwargs = {
            'do_sample': False,
            'max_new_tokens': 768,
            'min_new_tokens': 1,
            'temperature': .0,
        }

        self.device = 'cuda:0'
        self.config = config
        self.init_components()

        self.history = []
        self.images = []

        self.debug = True

    def init_components(self):
        d = self.config.model_dir
        model_name = get_model_name_from_path(d)
        tokenizer, model, image_processor, context_len = load_pretrained_model(d, None, model_name, False, False)
        self.model = model
        self.conv_mode = "jamba"
        self.jamba_process_images = process_images
        self.jamba_tokenizer_image_token = tokenizer_image_token
        self.truncate_input = True
        self.jamba_conv_templates = conv_templates
        eos_token_id = tokenizer.eos_token_id
        self.gen_kwargs['eos_token_id'] = eos_token_id
        self.gen_kwargs['pad_token_id'] = tokenizer.pad_token_id if tokenizer.pad_token_id else eos_token_id
        print(f'setting eos_token_id to {eos_token_id}')
        


        model.eval()
        self.tokenizer = tokenizer
        self.processor = image_processor

    def clear_history(self,):
        self.images = []
        self.history = []


    def tokenizer_image_token(self, prompt, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None): # copied from llava
        prompt_chunks = [self.tokenizer(chunk, add_special_tokens=False).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == self.tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    def get_conv(self, text):
        ret = []
        if self.history is None:
            self.history = []

        for conv in self.history:
            ret.append({'from': 'human', 'value': conv[0]})
            ret.append({'from': 'gpt', 'value': conv[1]})

        ret.append({'from': 'human', 'value': text})
        ret.append({'from': 'gpt', 'value': None})

        return ret


    def get_image_tensors(self, images):
        list_image_tensors = []
        crop_size = self.processor.crop_size
        processor = self.processor
        for fp in images:
            # if fp is None and self.data_args.is_multimodal: # None is used as a placeholder
            if fp is None: # None is used as a placeholder
                list_image_tensors.append(torch.zeros(3, crop_size['height'], crop_size['width']).to(self.device))
                continue
            elif isinstance(fp, str):
                image = Image.open(fp).convert('RGB')
            elif isinstance(fp, Image.Image):
                image = fp # already an image
            else:
                raise TypeError(f'Unsupported type {type(fp)}')

            if True or self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0] # a tensor
            list_image_tensors.append(image.to(self.device))
        return list_image_tensors


    def chat_with_jamba(self, text: str, media=None, isVideo=False, t=1.0, frameNum=128, patchside_length=336):
        def extract_frames(video, t=1.0, frameNum=128):
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
                        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        frameList.append(pil_img)
                    count += 1
                cap.release()

                # If the number of keyframes exceeds 240, sample it down to 240
                if len(frameList) > frameNum:
                    indices = np.linspace(0, len(frameList) - 1, frameNum, dtype=int)
                    frameList = [frameList[i] for i in indices]

            except Exception as e:
                print("-" * 50)
                print(f"Error extracting keyframes from {video}: {e}")

            return frameList

        def check_image_path_valid(image):
            try:
                Image.open(image).convert('RGB') # make sure that the path exists
            except:
                print(f'invalid images in {image}')
                return False
            return True

        def is_video_file(path):
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']
            return any(path.lower().endswith(ext) for ext in video_extensions)

        def insert_image_placeholder_for_video(t, num_images, placeholder='<img><image></img>', tag='<t>'):
            result = '<vid>'
            for _ in range(num_images):
                result += f"{placeholder}{tag}"
            result = result.rstrip(tag) + '</vid>'
            result = result + t
            return result

        def processForBestFitPatch(text, images, output_dir='./LongLLaVA/data/TestBestFit', patchside_length=336):
            side_length = patchside_length
            placeholder_count = text.count('<image>')
            if placeholder_count != len(images):
                raise ValueError("The number of <image> placeholders does not match the number of images.")
            
            new_image_paths = []
            os.makedirs(output_dir, exist_ok=True)

            for idx, image_path in enumerate(images):
                if isinstance(image_path, str) and os.path.isfile(image_path):
                    image = cv2.imread(image_path)
                elif isinstance(image_path, Image.Image):  # Assuming image_path is a numpy array representing an image
                    image = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
                    random_filename = str(uuid.uuid4()) + '.jpg'
                    random_path = os.path.join(output_dir, 'ori', 'images', random_filename)
                    cv2.imwrite(random_path, image)
                    image_path = random_path
                    
                if image is None:
                    raise FileNotFoundError(f"Image not found: {image_path}")

                height, width = image.shape[:2]

                new_height = ((height + side_length - 1) // side_length) * side_length
                new_width = ((width + side_length - 1) // side_length) * side_length
                pad_height = (new_height - height) // 2
                pad_width = (new_width - width) // 2

                padded_image = cv2.copyMakeBorder(image, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT, value=[255, 255, 255])

                split_images = [image_path]
                path_parts = image_path.split('/')
                base_name = os.path.splitext(path_parts[-1])[0]
                if len(path_parts) >= 3:
                    subdir = os.path.join(output_dir, path_parts[-3], path_parts[-2])
                else:
                    subdir = os.path.join(output_dir, path_parts[-2])

                os.makedirs(subdir, exist_ok=True)

                for i in range(0, new_height, side_length):
                    for j in range(0, new_width, side_length):
                        split_img = padded_image[i:i+side_length, j:j+side_length]
                        split_path = os.path.join(subdir, f'{base_name}_{i//side_length}_{j//side_length}.jpg')
                        if not os.path.exists(split_path):
                            cv2.imwrite(split_path, split_img)
                        split_images.append(split_path)
                
                row_count = new_height // side_length
                col_count = new_width // side_length
                
                replace_str = '<image>\n' + '\n'.join(['<img>' + '</img><img>'.join(['<image>' for _ in range(col_count)]) + '</img>' for _ in range(row_count)])
                text = text.replace('<image>', replace_str, 1)
                
                new_image_paths.extend(split_images)

            final_placeholder_count = text.count('<image>')
            if final_placeholder_count != len(new_image_paths):
                print(new_image_paths)
                print(placeholder_count)
                raise ValueError("The number of processed <image> placeholders does not match the number of split images.")
            
            return text, new_image_paths

        if text == '':
            return 'Please type in something'

        if isinstance(media, str) or isinstance(media, Image.Image) or isinstance(media, cv2.VideoCapture) or media is None:
            media = [media]

        images = []
        VideoFLAG = False
        media = [item for item in media if item is not None]
        for mediaItem in media:
            if not mediaItem:
                continue
            if isinstance(mediaItem, Image.Image):
                # media is an image object
                images.append(mediaItem)
            elif isinstance(mediaItem, cv2.VideoCapture):
                # media is a video object
                images.extend(extract_frames(mediaItem, t, frameNum))
                VideoFLAG = True
            elif os.path.isfile(mediaItem):
                if is_video_file(mediaItem):
                    # media is a video file path
                    images.extend(extract_frames(mediaItem, t, frameNum))
                    VideoFLAG = True
                elif check_image_path_valid(mediaItem):
                    # media is an image file path
                    images.append(mediaItem)
                else:
                    print("The provided path does not exist.")
                    continue
            else:
                print(f"The provided mediaItem is neither a recognized path nor a media object.\n mediaItem:{mediaItem}")
                continue


        # assert len(images) < self.max_images_per_round, f'at most {self.max_images_per_round} images'

        if VideoFLAG or isVideo:
            if len(images) > frameNum:
                indices = np.linspace(0, len(images) - 1, frameNum, dtype=int)
                images = [images[i] for i in indices]
            
            text = insert_image_placeholder_for_video(text, len(images))
        
        
                
        if '</img>' not in text:
            text = text.replace('<image>', '<img><image></img>')

        if len(images):
            if 'bestFit' in self.config.patchStrategy:
                text, images = processForBestFitPatch(text, images, patchside_length=patchside_length)
            elif 'norm'!=self.config.patchStrategy:
                print('Error: patchStrategy is not Impplmented')


        if images == []  and self.images == []:
            self.images = [None]
        self.images.extend(images)

        # make conv
        conv = self.jamba_conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # make input ids
        input_ids = self.jamba_tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        if self.images != [None]:
            lenth = len(images)
            image_tensors = self.jamba_process_images(self.images, self.processor, self.model.config).to(self.device, dtype=torch.float16)
        else:
            image_tensors = None

        output_ids = self.model.generate(
                input_ids,
                images=image_tensors,
                use_cache=True,
                **self.gen_kwargs)

        try:
            answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            if self.debug:
                print(f'input_ids:{input_ids}')
                input = self.tokenizer.decode(input_ids[input_ids != -200])
                print(f'input:{input}')
                print(f'output_ids:{output_ids}')
                print(f'answer:{answer}')
                self.debug=False
        except:
            raise ValueError('Shouldn\'t be an error here!')

        return answer

    def chat(self, text: str, images: list[str]=None, isVideo=False, t=1.0, frameNum=128, patchside_length=336):
        '''
        images: list[str], images for this round
        text: str
        '''

        return self.chat_with_jamba(text, images, isVideo, t, frameNum, patchside_length)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_dir", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--patchStrategy", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    bot = Chatbot(args)
    