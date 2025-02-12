import json
import os
from openai import OpenAI
from tqdm import tqdm
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REWARD_MODEL = "skywork"
assert REWARD_MODEL in ["skywork", "dt_skywork", "nemotron"]
REGEN_COMPLETION=True

generate_client = OpenAI(
    api_key=os.environ["TOGETHER_API_KEY"], base_url="https://api.together.xyz/v1"
)

def load_skywork():
    # Load model and tokenizer
    device = "cuda:0"
    model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
        num_labels=1,
    )
    rm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    return rm, rm_tokenizer, device

def load_dt_skywork():
    model_name = "Decision-Tree-Reward-Llama-3.1-8B" # Another choice is "Decision-Tree-Reward-Gemma-2-27B" 
    repo_id = f"RLHFlow/{model_name}"
    device = "cuda"
    # Initialize the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
    # Load the decision tree
    model.load_decision_tree(repo_id, filename="decision_tree.pkl")
    return model, tokenizer, device

def load_nemotron():
    rw_model_client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API"],
    )
    return rw_model_client

def evaluate_skywork(item, prompt_type, completion, device, rm, rm_tokenizer):
    conv = [{"role": "user", "content": item['original_prompt']}, 
            {"role": "assistant", "content": completion}]
    conv_tokenized = rm_tokenizer.apply_chat_template(conv, tokenize=True, return_tensors="pt").to(device)

    # Get the reward scores
    with torch.no_grad():
        score = rm(conv_tokenized).logits[0][0].item()
    
    result["rewards"][prompt_type] = f"score:{score}"
    return result

def evaluate_dt_skywork(item, prompt_type, completion, device, rm, rm_tokenizer):
    prompt = item['original_prompt']
    original_completion = item['completions']['original']

    # Compare the two responses
    output = rm.compare(prompt, original_completion, completion, rm_tokenizer, device)
    evaluation = ""
    for idx, attr in enumerate(output["attributes"]):
        evaluation += f"{attr}:{output['rewards'][1][idx]}," #1 ~ response of completion
    
    result["rewards"][prompt_type] = evaluation[:-1]
    return result

def generate_completion():
    with open("outputs/compressed_results.json", "r") as f:
        forward_data = json.load(f)

    completion_results = []

    for item in tqdm(forward_data):
        result = {
            "original_prompt": item["original"],
            "completions": {},
        }
        for prompt_type, prompt_text in item.items():
            completion = generate_client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K",
                messages=[{"role": "user", "content": prompt_text}],
            )
            result["completions"][prompt_type] = completion.choices[0].message.content
        completion_results.append(result)

        with open(f"outputs/{REWARD_MODEL}_completion_results.json", "w") as f:
            json.dump(completion_results, f, indent=4)
    return completion_results

def load_completion_results():
    if REGEN_COMPLETION:
        completion_results = generate_completion()
    else:
        with open(f"outputs/{REWARD_MODEL}_completion_results.json", "r") as f:
            completion_results = json.load(f)
    return completion_results


if REWARD_MODEL in ["skywork"]:
    rm, rm_tokenizer, device = load_skywork()
elif REWARD_MODEL in ["dt_skywork"]:
    rm, rm_tokenizer, device = load_dt_skywork()
elif REWARD_MODEL == "nemotron":
    rw_model_client = load_nemotron()

if __name__ == "__main__":
    reward_results = []
    completion_results = load_completion_results()
    for item in tqdm(completion_results):
        result = {
            "original_prompt": item["original_prompt"],
            "rewards": {},
        }
        original_prompt = item["original_prompt"]
        completions = item["completions"]
        for prompt_type, completion in completions.items():
            if REWARD_MODEL == "skywork":
                result = evaluate_skywork(item, prompt_type, completion, device, rm, rm_tokenizer)
            elif REWARD_MODEL == "dt_skywork":
                result = evaluate_dt_skywork(item, prompt_type, completion, device, rm, rm_tokenizer)
            elif REWARD_MODEL == "nemotron":
                response = rw_model_client.chat.completions.create(
                    model="nvidia/nemotron-4-340b-reward",
                    messages=[
                        {"role": "user", "content": original_prompt},
                        {"role": "assistant", "content": completion},
                    ],
                )
                result["rewards"][prompt_type] = response.choices[0].message[0].content
            
            print(result["rewards"])
        reward_results.append(result)

        with open(f"outputs/{REWARD_MODEL}_reward_results.json", "w") as f:
            json.dump(reward_results, f, indent=4)
