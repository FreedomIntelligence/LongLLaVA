from argparse import ArgumentParser
import json


parser = ArgumentParser()
# parser.add_argument('--output_path', type=str, default='llava1.5_13b_val.json',
#                     help='name of saved json')
# parser.add_argument('--config_path', type=str, default="configs/llava1.5.yaml")
parser.add_argument('--input', type=str, default="") # hf dataset path.
parser.add_argument('--output', type=str, default="")
# parser.add_argument('--split', type=str, default='validation')
# parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()


with open(args.input) as f:
    lines = [json.loads(line) for line in f.readlines()]

new = {}
for line in lines:
    new[line['id']] = line['parsed']


with open(args.output, 'w') as f:
    json.dump(new, f, indent=2, ensure_ascii=False)

print(f'converted to {args.output}')

