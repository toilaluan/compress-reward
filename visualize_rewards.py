import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the reward results
with open("reward_results.json", "r") as f:
    reward_data = json.load(f)

# Initialize data structures to store parsed rewards
reward_aspects = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
prompt_types = set()
parsed_rewards = {aspect: {} for aspect in reward_aspects}

# Parse the reward strings and organize data
for item in reward_data:
    for prompt_type, reward_str in item["rewards"].items():
        prompt_types.add(prompt_type)
        # Parse the reward string
        aspects = reward_str.split(",")
        for aspect in aspects:
            name, value = aspect.split(":")
            if prompt_type not in parsed_rewards[name]:
                parsed_rewards[name][prompt_type] = []
            parsed_rewards[name][prompt_type].append(float(value))

# Calculate means for each aspect and prompt type
mean_rewards = {
    aspect: {pt: np.mean(values) for pt, values in prompt_data.items()}
    for aspect, prompt_data in parsed_rewards.items()
}

# Create figure
fig = plt.figure(figsize=(15, 12))

# Create subplots for error bar plots
fig, axs = plt.subplots(len(reward_aspects), 1, figsize=(10, 15))
plt.subplots_adjust(hspace=0.4)

# Calculate means and standard errors for each aspect and prompt type
prompt_types = list(prompt_types)
for i, aspect in enumerate(reward_aspects):
    means = [np.mean(parsed_rewards[aspect][pt]) for pt in prompt_types]
    sems = [
        np.std(parsed_rewards[aspect][pt]) / np.sqrt(len(parsed_rewards[aspect][pt]))
        for pt in prompt_types
    ]

    # Plot with error bars in respective subplot
    axs[i].errorbar(means, prompt_types, xerr=sems, fmt="o", capsize=5, color=f"C{i}")

    # Customize each subplot
    axs[i].set_xlabel("Score")
    axs[i].set_title(f"{aspect.capitalize()}")
    axs[i].grid(True, linestyle="--", alpha=0.7)

    # Set consistent x-axis limits
    max_score = max([max(parsed_rewards[aspect][pt]) for pt in prompt_types])
    min_score = min([min(parsed_rewards[aspect][pt]) for pt in prompt_types])
    margin = (max_score - min_score) * 0.1
    axs[i].set_xlim(min_score - margin, max_score + margin)

# Only show y-axis labels (prompt types) for the middle subplot
for ax in axs:
    ax.set_ylabel("Prompt Type")

# Add overall title
fig.suptitle("Reward Analysis by Aspect", fontsize=16, y=0.95)

# Save the figure
plt.savefig("reward_analysis_error_bars.png", bbox_inches="tight", dpi=300)
plt.close()

# Print numerical summary
print("\nMean Scores by Prompt Type:")
for prompt_type in prompt_types:
    print(f"\n{prompt_type}:")
    for aspect in reward_aspects:
        mean_score = mean_rewards[aspect][prompt_type]
        print(f"{aspect.capitalize()}: {mean_score:.3f}")
