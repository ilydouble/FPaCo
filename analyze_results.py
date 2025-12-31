
import json
import os
import pandas as pd

methods = ['ce_loss', 'focal_loss', 'paco', 'gpaco', 'bpaco_original']
# Updated datasets list including new variants
datasets = ['aptos', 'fingerA', 'fingerB', 'fingerC', 'mias', 'octa', 'oral_cancer']
base_path = '/Users/liruirui/Documents/code/study/FPaCo'
output_file = 'results.md'

results_data = []

# 1. Base Methods
for dataset in datasets:
    # Map dataset name to folder name if needed
    # Standard names: fingerA, fingerB, fingerC are top level folders or handled by script?
    # The shell scripts output to "$RESULTS_ROOT/finger" or "fingerB"
    # We need to map `dataset` to `result_folder_name`
    
    res_folder = dataset
    if dataset == 'fingerA': res_folder = 'finger' # Original scripts mapped fingerA -> finger (based on user edit Step 671)
    
    for method in methods:
        json_path = os.path.join(base_path, method, 'results', res_folder, 'history.json')
        
        metrics = {'Method': method, 'Dataset': dataset}
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    history = json.load(f)
                
                # Taking Max
                if 'acc' in history and history['acc']: metrics['Accuracy'] = max(history['acc'])
                else: metrics['Accuracy'] = 'N/A'
                    
                if 'f1' in history and history['f1']: metrics['F1 Score'] = max(history['f1'])
                else: metrics['F1 Score'] = 'N/A'
                
                if 'auc' in history and history['auc']: metrics['AUC'] = max(history['auc'])
                else: metrics['AUC'] = 'N/A'
                    
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
                metrics.update({'Accuracy': 'Error', 'F1 Score': 'Error', 'AUC': 'Error'})
        else:
            metrics.update({'Accuracy': 'N/A', 'F1 Score': 'N/A', 'AUC': 'N/A'})
            
        results_data.append(metrics)

# 2. BioMedCLIP
for dataset in datasets:
    # BioMedCLIP results might be named differently?
    # biomedclip/results_dataset.json
    # Assume file is results_fingerA.json? Or still results_finger.json?
    # Let's check logic: usually logic maps 'finger' -> 'fingerprint'
    
    # We try exact name match first
    biomed_json = os.path.join(base_path, 'biomedclip', f'results_{dataset}.json')
    # Fallback for fingerA if results_finger.json exists?
    if not os.path.exists(biomed_json) and dataset == 'fingerA':
         biomed_json = os.path.join(base_path, 'biomedclip', 'results_finger.json')

    if os.path.exists(biomed_json):
        try:
            with open(biomed_json, 'r') as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    entry = data[0]
                    results_data.append({
                        'Method': 'BioMedCLIP',
                        'Dataset': dataset,
                        'Accuracy': entry.get('acc', 'N/A'),
                        'F1 Score': entry.get('f1', 'N/A'),
                        'AUC': entry.get('auc', 'N/A')
                    })
        except Exception: pass

# 3. Tip-Adapter
for dataset in datasets:
    # tipadapter/results/dataset_biomed_tipadapter.txt
    tip_txt = os.path.join(base_path, 'tipadapter', 'results', f'{dataset}_biomed_tipadapter.txt')
    if not os.path.exists(tip_txt) and dataset == 'fingerA':
        tip_txt = os.path.join(base_path, 'tipadapter', 'results', 'finger_biomed_tipadapter.txt')
        
    if os.path.exists(tip_txt):
        try:
            with open(tip_txt, 'r') as f: lines = f.readlines()
            
            # Parsing logic (simplified)
            tip_res = {k: {'Accuracy': 'N/A', 'F1 Score': 'N/A', 'AUC': 'N/A'} 
                       for k in ['Tip-Adapter (Zero-Shot)', 'Tip-Adapter (Training-Free)', 'Tip-Adapter (Fine-Tuned)']}
            
            for line in lines:
                if ':' not in line: continue
                k, v = line.split(':', 1)
                k = k.strip()
                try: v = float(v.strip().rstrip('%'))/100.0 if '%' in v else float(v.strip())
                except: v = 'N/A'
                
                if 'Zero-Shot' in k: 
                    m = 'Tip-Adapter (Zero-Shot)'
                    if 'Acc' in k: tip_res[m]['Accuracy'] = v
                    elif 'F1' in k: tip_res[m]['F1 Score'] = v
                    elif 'AUC' in k: tip_res[m]['AUC'] = v
                elif 'Tip-Adapter-F' in k:
                    m = 'Tip-Adapter (Fine-Tuned)'
                    if 'Acc' in k: tip_res[m]['Accuracy'] = v
                    elif 'F1' in k: tip_res[m]['F1 Score'] = v
                    elif 'AUC' in k: tip_res[m]['AUC'] = v
                elif 'Tip-Adapter' in k: # Training-free default
                    m = 'Tip-Adapter (Training-Free)'
                    if 'Acc' in k: tip_res[m]['Accuracy'] = v
                    elif 'F1' in k: tip_res[m]['F1 Score'] = v
                    elif 'AUC' in k: tip_res[m]['AUC'] = v

            for m_name, m_vals in tip_res.items():
                if m_vals['Accuracy'] != 'N/A':
                    results_data.append({'Method': m_name, 'Dataset': dataset, **m_vals})
        except Exception: pass

# 4. DPE
for dataset in datasets:
    dpe_json = os.path.join(base_path, 'dpe', 'results', f'results_{dataset}.json')
    if not os.path.exists(dpe_json) and dataset == 'fingerA':
         dpe_json = os.path.join(base_path, 'dpe', 'results', 'results_finger.json')
         
    if os.path.exists(dpe_json):
        try:
            with open(dpe_json, 'r') as f: data = json.load(f)
            results_data.append({
                'Method': 'DPE', 'Dataset': dataset,
                'Accuracy': data.get('acc', 'N/A'), 'F1 Score': data.get('f1', 'N/A'), 'AUC': data.get('auc', 'N/A')
            })
        except Exception: pass

# 5. CoOp
for dataset in datasets:
    coop_json = os.path.join(base_path, 'coop', 'results', f'results_{dataset}.json')
    if not os.path.exists(coop_json) and dataset == 'fingerA':
         coop_json = os.path.join(base_path, 'coop', 'results', 'results_finger.json')
         
    if os.path.exists(coop_json):
        try:
            with open(coop_json, 'r') as f: data = json.load(f)
            results_data.append({
                'Method': 'CoOp', 'Dataset': dataset,
                'Accuracy': data.get('acc', 'N/A'), 'F1 Score': data.get('f1', 'N/A'), 'AUC': data.get('auc', 'N/A')
            })
        except Exception: pass

# Process Output
df = pd.DataFrame(results_data)
if not df.empty:
    # Formatting
    df['Dataset'] = pd.Categorical(df['Dataset'], categories=datasets, ordered=True)
    df = df.sort_values(['Dataset', 'Method'])
    for col in ['Accuracy', 'F1 Score', 'AUC']:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

    # Write Markdown
    with open(output_file, 'w') as f:
        f.write("# Experiment Results Summary\n\n")
        f.write("## Overall Comparison\n\n")
        
        # Use pandas to_markdown if available, else manual
        try:
            f.write(df.to_markdown(index=False))
        except ImportError:
            # Simple fallback
            f.write(df.to_string(index=False))
            
        f.write("\n\n## Metric-Specific Comparisons\n")
        for metric in ['Accuracy', 'F1 Score', 'AUC']:
            f.write(f"\n### {metric}\n\n")
            try:
                pivoted = df.pivot(index='Dataset', columns='Method', values=metric)
                f.write(pivoted.to_markdown())
            except:
                f.write("Could not generate pivot table.")
            f.write("\n")

    print(f"Results saved to {output_file}")
else:
    print("No results found.")
