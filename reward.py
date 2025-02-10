import json
import os
from openai import OpenAI
from tqdm import tqdm

rw_model_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API"],
)
generate_client = OpenAI(
    api_key=os.environ["TOGETHER_API_KEY"], base_url="https://api.together.xyz/v1"
)


with open("compressed_results.json", "r") as f:
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

    with open("completion_results.json", "w") as f:
        json.dump(completion_results, f, indent=4)


reward_results = []

for item in completion_results:
    result = {
        "original_prompt": item["original_prompt"],
        "rewards": {},
    }
    original_prompt = item["original_prompt"]
    completions = item["completions"]
    for prompt_type, completion in completions.items():
        response = rw_model_client.chat.completions.create(
            model="nvidia/nemotron-4-340b-reward",
            messages=[
                {"role": "user", "content": original_prompt},
                {"role": "assistant", "content": completion},
            ],
        )
        print(prompt_type, response.choices[0].message[0].content)
        result["rewards"][prompt_type] = response.choices[0].message[0].content
    reward_results.append(result)

    with open("reward_results.json", "w") as f:
        json.dump(reward_results, f, indent=4)
