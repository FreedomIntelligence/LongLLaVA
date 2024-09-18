import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True, )



# Load prompts from a JSON file
with open('./benchmarks/Efficiency/Llama_3_8B_100k.json', 'r') as file:
    json_prompts = json.load(file)

# Extract the first prompt from each list
prompts = json_prompts[0][0]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the Llama2 tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('./ckpts/Jamba-v1.5')
model = AutoModelForCausalLM.from_pretrained('./ckpts/Jamba-v1.5', quantization_config=quantization_config)

# Tokenize the prompts
input_ids = tokenizer(prompts, return_tensors='pt')["input_ids"][:, :100000].to(device)
# print(input_ids)

# Define a function to generate text and measure time and GPU memory.
def generate_and_measure(input_ids, max_tokens):
    torch.cuda.empty_cache()  # Clear GPU memory cache
    
    # Reset memory statistics and start timing
    torch.cuda.reset_max_memory_allocated()
    start_time = time.time()
    
    # Generate text
    outputs = model.generate(input_ids, max_new_tokens=max_tokens, min_new_tokens=max_tokens)
    
    # Measure elapsed time and memory usage
    end_time = time.time()
    max_memory_used = torch.cuda.max_memory_allocated()
    elapsed_time = end_time - start_time
    
    # Decode and print the generated text (optional)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Elapsed Time: {elapsed_time:.4f} seconds")
    print(f"Max GPU Memory Used: {max_memory_used / 1024 ** 2:.2f} MB")
    # print(f"Generated Text: {generated_text[:200]}...")  # Print the first 200 characters

# Test with different token lengths
print("Generating 1 Token Length:")
generate_and_measure(input_ids, max_tokens=1)

print("Generating 1000 Tokens Length:")
generate_and_measure(input_ids, max_tokens=256)
