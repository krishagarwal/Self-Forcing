import os
import shutil
import random
import pandas as pd
from pathlib import Path
import numpy as np
import argparse
import json
from collections import defaultdict
import matplotlib.pyplot as plt

def load_prompts(prompt_dir: str) -> dict:
    """
    Load prompts from directory containing txt files and create a mapping dictionary.
    Each txt file may contain multiple prompt rows.
    
    Args:
        prompt_dir: Path to the directory containing prompt txt files
        
    Returns:
        Dictionary mapping video names to their prompts
    """
    prompts = {}
    prompt_files = list(Path(prompt_dir).glob("*.txt"))
    
    # Process all prompt files (no limit here, limit will be applied later)
    # max_prompts = getattr(load_prompts, 'max_prompts', 256)  # Default to 256 if not set
    # if len(prompt_files) > max_prompts:
    #     prompt_files = random.sample(prompt_files, max_prompts)
    
    for prompt_file in prompt_files:
        with open(prompt_file, 'r') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines):
            prompt = line.strip()
            if prompt:  # Skip empty lines
                # Video name is generated from first 200 chars of prompt
                video_name = prompt + '-0.mp4'
                prompts[video_name] = prompt
    
    print(f"Loaded {len(prompts)} prompts from {prompt_dir}")
    return prompts

def organize_evaluation_pairs(full_folder: str, sparse_folder: str, output_base: str, prompt_file: str, num_groups: int = 4, max_videos: int = 256):
    """
    Organize videos into evaluation pairs and generate necessary files.
    
    Args:
        full_folder: Path to folder containing full attention videos
        sparse_folder: Path to folder containing sparse attention videos
        output_base: Base path for output folders
        prompt_file: Path to the prompts text file
        num_groups: Number of groups to split videos into
    """
    # Load prompts
    prompts = load_prompts(prompt_file)
    print(f"Loaded {len(prompts)} prompts from {prompt_file}")
    
    # Get all video files
    full_videos = {f.stem: f for f in Path(full_folder).glob("*-0.mp4")}
    sparse_videos = {f.stem: f for f in Path(sparse_folder).glob("*-0.mp4")}

    print(sorted(list(prompts.keys()))[0])
    print(sorted(list(prompts.keys()))[0])

    full_videos = {k : v for k, v in full_videos.items() if any(f"{x}-0.mp4" == k for x in prompts.keys())}
    sparse_videos = {k : v for k, v in sparse_videos.items() if any(f"{x}-0.mp4" == k for x in prompts.keys())}

    # Find matching video names
    matching_names = sorted(list(set(full_videos.keys()) & set(sparse_videos.keys())))
    
    # Limit total pairs to max_videos
    if len(matching_names) > max_videos:
        matching_names = matching_names[:max_videos]
        print(f"Limited to {max_videos} videos as requested")
    
    total_pairs = len(matching_names)
    
    print(f"Total number of video pairs: {total_pairs}")
    print(f"Splitting into {num_groups} groups")
    
    # Calculate pairs per group
    pairs_per_group = total_pairs // num_groups
    remaining_pairs = total_pairs % num_groups
    
    print(f"Pairs per group: {pairs_per_group} (with {remaining_pairs} extra pairs distributed)")
    
    # Create mapping dictionary to store video information
    video_mapping = {
        "full_folder": full_folder,
        "sparse_folder": sparse_folder,
        "groups": {}
    }
    
    # Create output structure
    for group_idx in range(num_groups):
        # Calculate start and end indices for this group
        start_idx = group_idx * pairs_per_group
        end_idx = start_idx + pairs_per_group + (1 if group_idx < remaining_pairs else 0)
        
        # Get pairs for this group
        group_pairs = matching_names[start_idx:end_idx]
        
        print(f"\nProcessing group {group_idx+1} with {len(group_pairs)} pairs")
        
        # Create group folder
        group_folder = os.path.join(output_base, f"group_{group_idx+1}")
        os.makedirs(group_folder, exist_ok=True)
        
        # Initialize group mapping
        video_mapping["groups"][f"group_{group_idx+1}"] = {}
        
        # Create pairs
        for pair_idx, video_name in enumerate(group_pairs, 1):
            # Create pair folder
            pair_folder = os.path.join(group_folder, f"pair_{pair_idx:03d}")
            os.makedirs(pair_folder, exist_ok=True)
            
            # Randomly decide which video goes first
            if random.random() < 0.5:
                first_video = full_videos[video_name]
                second_video = sparse_videos[video_name]
                first_type = "full"
                second_type = "sparse"
            else:
                first_video = sparse_videos[video_name]
                second_video = full_videos[video_name]
                first_type = "sparse"
                second_type = "full"
            
            # Copy videos
            shutil.copy2(first_video, os.path.join(pair_folder, "video1.mp4"))
            shutil.copy2(second_video, os.path.join(pair_folder, "video2.mp4"))
            
            # Add prompt file
            # Remove the first 4 characters (e.g., "001_") from video name
            # video_name_without_prefix = video_name[4:] + '.mp4'
            video_name_without_prefix = video_name + '.mp4'
            prompt = prompts.get(video_name_without_prefix, "")
            if not prompt:
                print(f"Warning: No prompt found for video {video_name}")
            with open(os.path.join(pair_folder, "prompt.txt"), "w") as f:
                f.write(prompt)
            
            # Record mapping information
            video_mapping["groups"][f"group_{group_idx+1}"][f"pair_{pair_idx:03d}"] = {
                "video_name": video_name,
                "prompt": prompt,
                "video1": {
                    "type": first_type,
                    "original_path": str(first_video)
                },
                "video2": {
                    "type": second_type,
                    "original_path": str(second_video)
                }
            }
        
        # Create evaluation CSV
        df = pd.DataFrame({
            'Pair': [f"pair_{i:03d}" for i in range(1, len(group_pairs) + 1)],
            'WIN': [0] * len(group_pairs),
            'TIE': [0] * len(group_pairs),
            'LOSE': [0] * len(group_pairs)
        })
        df.to_csv(os.path.join(group_folder, "evaluation.csv"), index=False)
        
        # Create README
        readme_content = f"""# Group {group_idx+1} Evaluation

This folder contains {len(group_pairs)} video pairs for evaluation.

## Structure
- Each pair is in its own folder (pair_001, pair_002, etc.)
- Each pair folder contains:
  - video1.mp4
  - video2.mp4
  - prompt.txt (contains the original prompt used to generate these videos)

## Evaluation
- Use the evaluation.csv file to record your evaluation
- For each pair, mark one of WIN, TIE, or LOSE based on video1's quality relative to video2
- WIN: video1 is better than video2
- TIE: video1 and video2 are of similar quality
- LOSE: video1 is worse than video2

## Evaluation Criteria
Please evaluate the videos based on:
1. Visual quality (clarity, stability, consistency)
2. Prompt adherence (how well the video matches the given prompt)
3. Overall coherence and naturalness of the generated content

## Note
Please evaluate the videos based on their visual quality and prompt adherence, without any prior knowledge of their generation method.
"""
        with open(os.path.join(group_folder, "README.md"), "w") as f:
            f.write(readme_content)
    
    # Save mapping information
    mapping_file = os.path.join(output_base, "video_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(video_mapping, f, indent=2)
    
    print(f"\nMapping information saved to: {mapping_file}")

def plot_evaluation_results(evaluation_base: str):
    """
    读取 evaluation_results.csv 并画出 Win/Tie/Loss 的横向堆叠条形图
    """
    results_file = os.path.join(evaluation_base, 'evaluation_results.csv')
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return

    df = pd.read_csv(results_file)
    labels = df['Model'].tolist()
    win = df['Wins'].tolist()
    tie = df['Ties'].tolist()
    loss = df['Losses'].tolist()
    total = [w + t + l for w, t, l in zip(win, tie, loss)]
    win_pct = [w / tot * 100 if tot > 0 else 0 for w, tot in zip(win, total)]
    tie_pct = [t / tot * 100 if tot > 0 else 0 for t, tot in zip(tie, total)]
    loss_pct = [l / tot * 100 if tot > 0 else 0 for l, tot in zip(loss, total)]

    fig, ax = plt.subplots(figsize=(6.75, 2.2))
    y = range(len(labels))
    ax.barh(y, win_pct, label='Win', color='#7b9fe0')
    ax.barh(y, tie_pct, left=win_pct, label='Tie', color='#b5cbe8')
    ax.barh(y, loss_pct, left=[i+j for i, j in zip(win_pct, tie_pct)], label='Loss', color='#e8ecf4')

    for i in y:
        ax.text(win_pct[i]/2, i, f"{win_pct[i]:.1f}%", va='center', ha='center', color='white', fontsize=11)
        ax.text(win_pct[i]+tie_pct[i]/2, i, f"{tie_pct[i]:.1f}%", va='center', ha='center', color='black', fontsize=11)
        ax.text(win_pct[i]+tie_pct[i]+loss_pct[i]/2, i, f"{loss_pct[i]:.1f}%", va='center', ha='center', color='black', fontsize=11)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=12)

    ax.set_xlim(0, 100)
    # ax.set_xlabel('Percentage (%)', fontsize=12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))
    ax.tick_params(axis='x', labelsize=12)
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.6), fontsize=12)
    plt.tight_layout()
    save_path = os.path.join('finetune_scripts/human_eval/evaluation_results.png')
    plt.savefig(save_path, dpi=200)
    print(f"Plot saved to: {save_path}")
    plt.show()

def analyze_evaluation_results(evaluation_base: str, mapping_file: str):
    """
    Analyze evaluation results from all groups and determine which model performs better.
    
    Args:
        evaluation_base: Base path containing all evaluation groups
        mapping_file: Path to the video mapping JSON file
    """
    # Load mapping information
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    
    # Initialize counters
    results = {
        'full': {'win': 0, 'tie': 0, 'lose': 0},
        'sparse': {'win': 0, 'tie': 0, 'lose': 0},
        'total_pairs': 0
    }
    
    # Process each group
    for group_name in mapping['groups']:
        group_path = os.path.join(evaluation_base, group_name)
        eval_csv = os.path.join(group_path, 'evaluation.csv')
        
        if not os.path.exists(eval_csv):
            print(f"Warning: Evaluation file not found for {group_name}")
            continue
        
        # Load evaluation results
        eval_df = pd.read_csv(eval_csv)
        
        # Process each pair
        for _, row in eval_df.iterrows():
            pair_name = row['Pair']
            pair_info = mapping['groups'][group_name][pair_name]
            
            # Determine which model is video1 and which is video2
            video1_type = pair_info['video1']['type']
            video2_type = pair_info['video2']['type']
            
            # Get evaluation result
            if row['WIN'] == 1:
                results[video1_type]['win'] += 1
                results[video2_type]['lose'] += 1
            elif row['TIE'] == 1:
                results[video1_type]['tie'] += 1
                results[video2_type]['tie'] += 1
            elif row['LOSE'] == 1:
                results[video1_type]['lose'] += 1
                results[video2_type]['win'] += 1
            
            results['total_pairs'] += 1
    
    # Calculate statistics
    total = results['total_pairs']
    full_win_rate = (results['full']['win'] + 0.5 * results['full']['tie']) / total
    sparse_win_rate = (results['sparse']['win'] + 0.5 * results['sparse']['tie']) / total
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    print(f"Total pairs evaluated: {total}")
    print("\nFull Attention Model:")
    print(f"Wins: {results['full']['win']} ({results['full']['win']/total*100:.1f}%)")
    print(f"Ties: {results['full']['tie']} ({results['full']['tie']/total*100:.1f}%)")
    print(f"Losses: {results['full']['lose']} ({results['full']['lose']/total*100:.1f}%)")
    # print(f"Win rate (including ties): {full_win_rate*100:.1f}%")
    
    print("\nSparse Attention Model:")
    print(f"Wins: {results['sparse']['win']} ({results['sparse']['win']/total*100:.1f}%)")
    print(f"Ties: {results['sparse']['tie']} ({results['sparse']['tie']/total*100:.1f}%)")
    print(f"Losses: {results['sparse']['lose']} ({results['sparse']['lose']/total*100:.1f}%)")
    # print(f"Win rate (including ties): {sparse_win_rate*100:.1f}%")
    
    print("\nConclusion:")
    if full_win_rate > sparse_win_rate:
        print(f"Full Attention model performs better by {abs(full_win_rate - sparse_win_rate)*100:.1f}%")
    elif sparse_win_rate > full_win_rate:
        print(f"Sparse Attention model performs better by {abs(full_win_rate - sparse_win_rate)*100:.1f}%")
    else:
        print("Both models perform equally well")
    
    # Save detailed results to CSV
    results_df = pd.DataFrame({
        'Model': ['ViSA\nvs\nSVG'],
        'Wins': [results['sparse']['win']],
        'Ties': [results['sparse']['tie']],
        'Losses': [results['sparse']['lose']],
        'Win Rate': [sparse_win_rate]
    })
    
    results_file = os.path.join(evaluation_base, 'evaluation_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to: {results_file}")

    # 新增：画图
    plot_evaluation_results(evaluation_base)

if __name__ == "__main__":
    # python local_scripts/human_eval/generate_human_eval.py --full_folder ../finetrainer-sparse-ft/human_eval_video/Wan-14B-full-720/ --sparse_folder human_eval_video/Wan2.1-VSA-T2V-14B-720P-Diffusers-g5.0-f5.0-renamed/ --output_base human_eval_video/evaluation_groups_14B --prompt_dir ./assets/human_eval_14B_split_64 --num_groups 2 --analyze

    # python local_scripts/human_eval/generate_human_eval.py --full_folder human_eval_video/Wan2.1-T2V-1.3B-Diffusers-DMD --sparse_folder human_eval_video/FastWan2.1-T2V-1.3B-Diffusers --output_base human_eval_video/evaluation_groups_1.3B_Distill --prompt_dir ./assets/human_eval_1.3B_split_64 --num_groups 2 --analyze
    parser = argparse.ArgumentParser(description='Organize videos into evaluation pairs')
    parser.add_argument('--num_groups', type=int, default=4, help='Number of groups to split videos into')
    parser.add_argument('--max_videos', type=int, default=100, help='Maximum number of videos to process (default: 256)')
    parser.add_argument('--full_folder', type=str, default="human_eval_video/SVG-0.85",
                      help='Path to folder containing full attention videos')
    parser.add_argument('--sparse_folder', type=str, default="human_eval_video/Wan-1.3B-sparse-topk32-g4.0-s5",
                      help='Path to folder containing sparse attention videos')
    parser.add_argument('--output_base', type=str, default="human_eval_video/evaluation_groups_1.3B",
                      help='Base path for output folders')
    parser.add_argument('--prompt_dir', type=str, default="./assets/MG_256",
                      help='Path to the directory containing prompt txt files')
    parser.add_argument('--analyze', action='store_true',
                      help='Analyze evaluation results instead of organizing videos')
    
    args = parser.parse_args()
    
    if args.analyze:
        mapping_file = os.path.join(args.output_base, "video_mapping.json")
        analyze_evaluation_results(args.output_base, mapping_file)
    else:
        organize_evaluation_pairs(
            full_folder=args.full_folder,
            sparse_folder=args.sparse_folder,
            output_base=args.output_base,
            prompt_file=args.prompt_dir,
            num_groups=args.num_groups,
            max_videos=args.max_videos
        )
