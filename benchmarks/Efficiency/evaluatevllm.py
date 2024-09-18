import time
import sys
import json
from vllm import LLM, SamplingParams
import os
os.environ['VLLM_FUSED_MOE_CHUNK_SIZE']='32768'


# Load prompts from a JSON file
with open('./benchmarks/Efficiency/Llama_3_8B_100k.json', 'r') as file:
    json_prompts = json.load(file)


modelName = '/sds_wangby/models/Llama-2-13b-hf'
prompts = json_prompts[0][0]
tokenizer = AutoTokenizer.from_pretrained(modelName)
input_ids = tokenizer(prompts, return_tensors='pt')["input_ids"][:, :100000]

prompts = tokenizer.decode(input_ids)
llm = LLM(model=modelName)
# llm = LLM(model="/wangbenyou/xidong/VisionJamba/ckpts/Jamba-v1.5", quantization="experts_int8", max_model_len=100*1024)

# Define a function to generate text and measure time and memory.
def generate_and_measure(prompts, max_tokens):
    sampling_params = SamplingParams(sample=False, max_tokens=max_tokens, min_tokens=max_tokens)
    
    start_time = time.time()  # Start timing
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()  # End timing
    
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    
    # Calculate total memory used by the generated outputs
    total_memory = sum(sys.getsizeof(output.outputs[0].text) for output in outputs)
    
    # Print the outputs along with the stats
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    
    print(f"Elapsed Time: {elapsed_time:.4f} seconds")
    print(f"Total Memory Used: {total_memory} bytes")

# Generate and measure for 1 token length
print("Generating 1 Token Length:")
generate_and_measure(prompts, max_tokens=1)

# Generate and measure for 500 tokens length
print("\nGenerating 1000 Tokens Length:")
generate_and_measure(prompts, max_tokens=1000)
