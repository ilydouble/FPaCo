
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configs
METHODS_ORDER = [
    'BioMedCLIP', 
    'CoOp', 
    'Tip-Adapter (Zero-Shot)', 'Tip-Adapter (Training-Free)', 'Tip-Adapter (Fine-Tuned)',
    'TDA', 
    'DPE', 
    'ce_loss', 'focal_loss', 
    'paco', 'gpaco', 'bpaco_original', 
    'FPaCo (NoHeat)',
    'FPaCo (Heat)'
]

DATASETS = ['aptos', 'fingerA', 'fingerB', 'fingerC', 'mias', 'octa', 'oral_cancer']
BASE_PATH = '/Users/liruirui/Documents/code/study/FPaCo'
OUTPUT_MD = 'results.md'
OUTPUT_IMG = 'results_chart.png'

def read_json_metric(path, key_metric):
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Helper to extract max of a list or value
            def get_val(d, k):
                return d.get(k)

            # Check if data contains lists (history style) or values (snapshot style)
            acc_val = get_val(data, 'acc')
            
            if isinstance(acc_val, list) and acc_val:
                # History style: Find index of max accuracy
                # We prioritize accuracy for the table
                max_acc = max(acc_val)
                # handle multiple max? just pick first or whatever
                idx = acc_val.index(max_acc)
                
                f1_list = data.get('f1', [])
                auc_list = data.get('auc', [])
                
                # Careful if lists are different lengths (shouldn't be, but good to be safe)
                f1 = f1_list[idx] if idx < len(f1_list) else None
                auc = auc_list[idx] if idx < len(auc_list) else None
                
                return {
                    'Accuracy': max_acc,
                    'F1 Score': f1,
                    'AUC': auc
                }

            elif isinstance(acc_val, (int, float)):
                 # Snapshot style (FPaCo, BioMedCLIP, etc.)
                 return {
                    'Accuracy': acc_val,
                    'F1 Score': get_val(data, 'f1'),
                    'AUC': get_val(data, 'auc')
                }
                
            return None
            
        except Exception as e:
            # print(f"Error reading {path}: {e}")
            pass
    return None

def main():
    results = []

    # 1. Iterate Datasets
    for ds in DATASETS:
        # --- Standard History JSON Methods ---
        # ce_loss, focal_loss, paco, gpaco, bpaco
        # Paths: method/results/ds/history.json
        # Handle 'fingerA' mapping if needed. Previous logic: if fingerA -> finger
        folder_name = ds
        # if ds in ['fingerA', 'fingerB', 'fingerC']: folder_name = 'finger' # But previous run script used 'fingerB' etc?
        # Let's check logic:
        # Step 670 user edit: fingerB -> output "fingerB". fingerC -> "fingerC".
        # So we trust folder_name = ds for new ones.
        # But for 'fingerA', user mapped to 'finger' in script? No, Step 670 changed finger -> fingerA.
        # Wait, Step 670 changed input dataset path, output dir "finger".
        # Step 670 snippet: 
        #   # 4. Fingerprint -> dataset fingerA -> output "finger". (Original was finger->finger)
        #   # 5. FingerprintB -> dataset fingerB -> output "fingerB" (New?) - Wait, duplicate key #4.
        
        # Assumption:
        # fingerA -> results stored in 'finger' folder? Or 'fingerA'?
        # Let's try both.
        
        std_methods = ['ce_loss', 'focal_loss', 'paco', 'gpaco', 'bpaco_original']
        for m in std_methods:
            # Try ds name
            path = os.path.join(BASE_PATH, m, 'results', ds, 'history.json')
            metrics = read_json_metric(path, 'acc')
            
            # Fallback for fingerA -> finger
            if not metrics and ds == 'fingerA':
                path = os.path.join(BASE_PATH, m, 'results', 'finger', 'history.json')
                metrics = read_json_metric(path, 'acc')

            if metrics:
                results.append({'Method': m, 'Dataset': ds, **metrics})
            else:
                 # Add N/A row? User wants comparison. Best to add empty.
                results.append({'Method': m, 'Dataset': ds, 'Accuracy': None})

        # --- Multimodal (FPaCo) ---
        heat_ds_name = ds
        if ds == 'fingerA': heat_ds_name = 'finger'
        
        path = os.path.join(BASE_PATH, 'fpaco_noheat', 'results', heat_ds_name, 'results.json')
        # Fallback to direct name if mapped path doesn't exist (though mapped is likely correct for fingerA)
        if not os.path.exists(path) and heat_ds_name != ds:
             path = os.path.join(BASE_PATH, 'fpaco_noheat', 'results', ds, 'results.json')

        metrics = read_json_metric(path, 'acc')
        results.append({'Method': 'FPaCo (NoHeat)', 'Dataset': ds, **(metrics or {'Accuracy': None})})

        # --- BioMedCLIP ---
        # biomedclip/results_ds.json
        path = os.path.join(BASE_PATH, 'biomedclip', f'results_{ds}.json')
        if not os.path.exists(path) and ds == 'fingerA': path = os.path.join(BASE_PATH, 'biomedclip', 'results_finger.json')
        
        # BioMed is list of dicts.
        if os.path.exists(path):
            try:
                with open(path) as f: d = json.load(f)[0]
                results.append({
                    'Method': 'BioMedCLIP', 'Dataset': ds,
                    'Accuracy': d.get('acc'), 'F1 Score': d.get('f1'), 'AUC': d.get('auc')
                })
            except: results.append({'Method': 'BioMedCLIP', 'Dataset': ds, 'Accuracy': None})
        else: results.append({'Method': 'BioMedCLIP', 'Dataset': ds, 'Accuracy': None})

        # --- CoOp ---
        path = os.path.join(BASE_PATH, 'coop', 'results', f'results_{ds}.json')
        if not os.path.exists(path) and ds == 'fingerA': path = os.path.join(BASE_PATH, 'coop', 'results', 'results_finger.json')
        metrics = read_json_metric(path, 'acc') # CoOp json isn't list? Check Step 202 code... "data = json.load(f)". Yes dict.
        results.append({'Method': 'CoOp', 'Dataset': ds, **(metrics or {'Accuracy': None})})

        # --- DPE ---
        path = os.path.join(BASE_PATH, 'dpe', 'results', f'results_{ds}.json')
        if not os.path.exists(path) and ds == 'fingerA': path = os.path.join(BASE_PATH, 'dpe', 'results', 'results_finger.json')
        metrics = read_json_metric(path, 'acc')
        results.append({'Method': 'DPE', 'Dataset': ds, **(metrics or {'Accuracy': None})})

        # --- FPaCo (Heat) ---
        path_heat = os.path.join(BASE_PATH, 'fpaco', 'results', ds, 'results.json')
        if not os.path.exists(path_heat) and ds == 'fingerA':
            path_heat = os.path.join(BASE_PATH, 'fpaco', 'results', 'finger', 'results.json')
        
        metrics_heat = read_json_metric(path_heat, 'acc')
        results.append({'Method': 'FPaCo (Heat)', 'Dataset': ds, **(metrics_heat or {'Accuracy': None})})

        # --- Tip-Adapter ---
        def parse_tip_adapter_txt(fpath):
            parsed = {
                'Tip-Adapter (Zero-Shot)': {},
                'Tip-Adapter (Training-Free)': {},
                'Tip-Adapter (Fine-Tuned)': {}
            }
            if not os.path.exists(fpath):
                # Try finger fallback
                if 'finger' in fpath and 'fingerA' not in fpath and 'fingerB' not in fpath and 'fingerC' not in fpath:
                     pass # Already generic
                elif 'finger' in fpath:
                     # Try generic 'finger'
                     parent = os.path.dirname(fpath)
                     generic_path = os.path.join(parent, 'finger_biomed_tipadapter.txt')
                     if os.path.exists(generic_path):
                         fpath = generic_path
            
            if not os.path.exists(fpath):
                return parsed
                
            try:
                with open(fpath, 'r') as f:
                    for line in f:
                        if ':' not in line: continue
                        k, v = line.split(':', 1)
                        k = k.strip()
                        v = v.strip()
                        try:
                            val = float(v.rstrip('%')) / 100.0 if '%' in v else float(v)
                        except:
                            continue
                            
                        # Determine variant
                        target = None
                        if 'Zero-Shot' in k: target = 'Tip-Adapter (Zero-Shot)'
                        elif 'Tip-Adapter-F' in k: target = 'Tip-Adapter (Fine-Tuned)'
                        elif 'Tip-Adapter' in k: target = 'Tip-Adapter (Training-Free)'
                        
                        if target:
                            if 'Acc' in k: parsed[target]['Accuracy'] = val
                            elif 'F1' in k: parsed[target]['F1 Score'] = val
                            elif 'AUC' in k: parsed[target]['AUC'] = val
            except Exception as e:
                print(f"Error parsing Tip-Adapter file {fpath}: {e}")
                
            return parsed

        txt_fname = f'{ds}_biomed_tipadapter.txt'
        # Handle mapping if needed (fingerA -> finger)
        if ds == 'fingerA': txt_fname = 'finger_biomed_tipadapter.txt'
        
        txt_path = os.path.join(BASE_PATH, 'tipadapter', 'results', txt_fname)
        
        tip_data = parse_tip_adapter_txt(txt_path)
        for variant, metrics in tip_data.items():
            results.append({'Method': variant, 'Dataset': ds, **metrics})

    # 2. DataFrame Processing
    df = pd.DataFrame(results)
    
    # Fill missing cols
    for c in ['Accuracy', 'F1 Score', 'AUC']:
        if c not in df.columns: df[c] = np.nan
        
    # Order Methods
    df['Method'] = pd.Categorical(df['Method'], categories=METHODS_ORDER, ordered=True)
    df.sort_values(['Dataset', 'Method'], inplace=True)

    # 3. Formatting and Bolding
    # We create a display copy
    df_disp = df.copy()
    
    # Bold max Accuracy per Dataset
    for ds in DATASETS:
        # Get subset indices
        mask = df_disp['Dataset'] == ds
        if not mask.any(): continue
        
        subset = df.loc[mask]
        
        # Find Max Accuracy
        # Ensure numeric
        accs = pd.to_numeric(subset['Accuracy'], errors='coerce')
        if accs.notna().any():
            max_val = accs.max()
            # Highlight all matching max
            is_max = (accs == max_val) & accs.notna()
            
            # Apply formatting
            for idx in is_max[is_max].index:
                val = df_disp.at[idx, 'Accuracy']
                # Format number then bold
                if isinstance(val, (int, float)):
                    df_disp.at[idx, 'Accuracy'] = f"**{val:.4f}**"
    
    # Format others regular
    for col in ['Accuracy', 'F1 Score', 'AUC']:
        df_disp[col] = df_disp[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and "**" not in str(x) else x)
        
    # 4. Generate Chart
    # Pivot for chart: X=Dataset, Y=Accuracy, Hue=Method
    pivot_chart = df.pivot(index='Dataset', columns='Method', values='Accuracy')
    # Filter only present datasets and methods
    pivot_chart = pivot_chart.dropna(how='all', axis=0).dropna(how='all', axis=1)
    
    if not pivot_chart.empty:
        plt.figure(figsize=(15, 8))
        pivot_chart.plot(kind='bar', figsize=(15, 8), width=0.8)
        plt.title('Accuracy Comparison by Method and Dataset')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(OUTPUT_IMG)
        plt.close()

    # 5. Save Markdown
    with open(OUTPUT_MD, 'w') as f:
        f.write("# Experiment Results\n\n")
        
        if os.path.exists(OUTPUT_IMG):
            f.write(f"![Accuracy Chart]({OUTPUT_IMG})\n\n")
            
        f.write("## Detailed metrics\n\n")
        try:
            f.write(df_disp.to_markdown(index=False))
        except:
            f.write(df_disp.to_string(index=False))
            
    print(f"Done. Saved to {OUTPUT_MD} and {OUTPUT_IMG}")

if __name__ == '__main__':
    main()
