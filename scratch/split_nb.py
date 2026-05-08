import json
import os
from pathlib import Path

def extract_notebooks(ipynb_path, output_dir):
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    stages = [
        ("01_sft_mini.py", "Stage from `notebooks/01_sft_mini.py`"),
        ("02_preference_data.py", "Stage from `notebooks/02_preference_data.py`"),
        ("03_dpo_train.py", "Stage from `notebooks/03_dpo_train.py`"),
        ("04_compare_and_eval.py", "Stage from `notebooks/04_compare_and_eval.py`"),
        ("05_merge_deploy_gguf.py", "Stage from `notebooks/05_merge_deploy_gguf.py`"),
        ("06_benchmark.py", "Stage from `notebooks/06_benchmark.py`"),
    ]

    current_stage_idx = -1
    stage_cells = {name: [] for name, _ in stages}

    for cell in nb['cells']:
        source_text = "".join(cell['source'])
        
        # Check if this cell marks the start of a new stage
        found_new_stage = False
        for i, (name, marker) in enumerate(stages):
            if marker in source_text:
                current_stage_idx = i
                found_new_stage = True
                break
        
        if current_stage_idx != -1 and not found_new_stage:
            # Skip the setup/install cells at the very beginning of the stitched notebook
            # and only start collecting after the first stage marker.
            stage_cells[stages[current_stage_idx][0]].append(cell)

    header = "# ---\n# jupyter:\n#   jupytext:\n#     formats: py:percent\n# ---\n\n"

    for name, cells in stage_cells.items():
        content = [header]
        for cell in cells:
            if cell['cell_type'] == 'markdown':
                content.append("# %% [markdown]\n")
                for line in cell['source']:
                    # Jupytext markdown cells use '#' at start of each line
                    content.append(f"# {line}")
                if not cell['source'][-1].endswith('\n'):
                    content.append('\n')
                content.append("\n")
            elif cell['cell_type'] == 'code':
                content.append("# %%\n")
                content.append("".join(cell['source']))
                if not "".join(cell['source']).endswith('\n'):
                    content.append('\n')
                content.append("\n")
        
        target_path = Path(output_dir) / name
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write("".join(content))
        print(f"Written {target_path}")

if __name__ == "__main__":
    ipynb = "/home/cuong/Desktop/python/VinUni/D22/Day22-Track3-DPO-Alignment-Lab/colab/Lab22_DPO_T4.ipynb"
    out_dir = "/home/cuong/Desktop/python/VinUni/D22/Day22-Track3-DPO-Alignment-Lab/notebooks"
    extract_notebooks(ipynb, out_dir)
