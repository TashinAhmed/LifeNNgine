import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

def generate_report():
    if not os.path.exists('comparative_experiments/heatmap_results.json'):
        print("No results found.")
        return
    if not os.path.exists('comparative_experiments/heatmap_results.jsonl'):
        print("No results found.")
        return
    data = []
    with open('comparative_experiments/heatmap_results.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    df = pd.DataFrame(data)

    # >>> Generate Heatmaps <<<
    # One heatmap per Rule, axes: Degree vs Width (averaged over Depth?) or 
    # Maybe Degree vs Width for best Depth?
    # Or Depth is small (2,3), maybe just pick Depth=2?
    # Pivoted to: Rule x Degree x Width (take max success rate over depths? or average?)
    # "populate success rate heat maps across depths and widths"
    
    # QUESTION from tashinahmed: Maybe fix Degree and plot Depth vs Width?
    # Or multiple heatmaps.
    
    os.makedirs('comparative_experiments/plots', exist_ok=True)
    rules = df['rule'].unique()
    degrees = df['degree'].unique()
    with open('comparative_experiments/results.md', 'w') as f:
        f.write("# Comparative CA Analysis Results\n\n")
        for rule in rules:
            f.write(f"## Rule: {rule}\n\n")
            rule_df = df[df['rule'] == rule]
            # Heatmap: Width v/s Depth for each Degree
            for deg in degrees:
                sub_df = rule_df[rule_df['degree'] == deg]
                if sub_df.empty: continue
                pivot = sub_df.pivot(index='depth', columns='width', values='success_rate')
                plt.figure(figsize=(6, 4))
                sns.heatmap(pivot, annot=True, cmap='RdYlGn', vmin=0, vmax=1)
                plt.title(f"Success Rate: {rule} (Degree {deg})")
                plt.xlabel("Width")
                plt.ylabel("Depth")
                plt.tight_layout()
                img_path = f"plots/heatmap_{rule.replace(' ', '_')}_deg{deg}.png"
                plt.savefig(f"comparative_experiments/{img_path}")
                plt.close()
                f.write(f"### Degree {deg}\n\n")
                f.write(f"![Heatmap]({img_path})\n\n")
        f.write("## Learned Functions\n\n")
        f.write("Below are visualizations of the learned activation functions and kernels for selected successful runs.\n\n")
        plots = [p for p in os.listdir('comparative_experiments/plots') if 'heatmap' not in p]
        plots.sort()
        for p in plots:
            f.write(f"![{p}](plots/{p})\n\n")
    print("Report generated at comparative_experiments/results.md")

if __name__ == "__main__":
    generate_report()
