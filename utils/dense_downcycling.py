import shutil
import argparse
import torch

from Jamba.configuration_jamba import JambaConfig
from Jamba.modeling_jamba import JambaForCausalLM
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="Path to the input model")
parser.add_argument("--output_path", type=str, help="Path to save the output model")
# parser.add_argument("--expert_ids", type=int, nargs='+', help="Expert IDs to use")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

sparse_config = JambaConfig.from_pretrained(args.model_path, trust_remote_code=True).to_dict()

# Modify the config to set dense model
sparse_config["num_experts"] = 1
sparse_config["num_experts_per_tok"] = 1

def name_mapping(name):
    """
    Maps the given name to a list of names based on certain conditions.

    Args:
        name (str): The name to be mapped.

    Returns:
        tuple: A tuple containing a list of mapped names and a boolean value indicating whether the name contains "down_proj" or not.
    """
    if "experts" in name and (int(name.split(".")[2]) - sparse_config["expert_layer_offset"]) % sparse_config["expert_layer_period"] == 0:
        if "down_proj" in name:
            return [name.replace("experts.0", f"experts.{i}") for i in range(sparse_config["num_experts"])], True
        return [name.replace("experts.0", f"experts.{i}") for i in range(sparse_config["num_experts"])], False
    return [name], False

sparse_model = JambaForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)

dense_config = JambaConfig(**dict(sparse_config))
dense_model = JambaForCausalLM(dense_config)

param_dict = dict(sparse_model.named_parameters())

for name, param in dense_model.named_parameters():
    names, scale = name_mapping(name)
    print(names)
    if scale:
        # Scale down_proj for numerical stability
        param.data.copy_(torch.stack([param_dict[name].data for name in names]).mean(dim=0) * 0.045)
    else:
        if 'feed_forward.gate_proj.weight' in names[0]:
            if names[0] not in param_dict:
                expert = int(name.split(".")[2])
                names[0] = f'model.layers.{expert-1}.feed_forward.gate_proj.weight'
                print(names)
        param.data.copy_(torch.stack([param_dict[name].data for name in names]).mean(dim=0))
        
dense_model.to(dtype=torch.bfloat16)
tokenizer.save_pretrained(args.output_path)
dense_model.save_pretrained(args.output_path)
shutil.copy("Jamba/configuration_jamba.py", args.output_path)
shutil.copy("Jamba/modeling_jamba.py", args.output_path)

"""
python dense_downcycling.py --model_path ./ckpts/Jamba --output_path ./ckpts/Jamba-9B
"""
