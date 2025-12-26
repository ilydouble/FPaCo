import json
import os
import pandas as pd

methods = ['ce_loss', 'focal_loss', 'paco', 'gpaco', 'bpaco_original']
datasets = ['aptos', 'finger', 'mias', 'octa', 'oral_cancer']
base_path = '/Users/liruirui/Documents/code/study/FPaCo'

results_data = []

for dataset in datasets:
    for method in methods:
        json_path = os.path.join(base_path, method, 'results', dataset, 'history.json')
        
        metrics = {'Method': method, 'Dataset': dataset}
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    history = json.load(f)
                    
                # Extract max values for interest metrics
                # We assume the lists are of the same length and correspond to epochs.
                # However, for a comparison table of "best results", taking the max of each valid metric is a common practice.
                # Alternatively, we could pick the epoch with max F1 or max AUC. 
                # Let's take the max of each individual metric for now to show potential.
                
                if 'acc' in history and history['acc']:
                    metrics['Accuracy'] = max(history['acc'])
                else:
                    metrics['Accuracy'] = 'N/A'
                    
                if 'f1' in history and history['f1']:
                    metrics['F1 Score'] = max(history['f1'])
                else:
                     metrics['F1 Score'] = 'N/A'
                
                if 'auc' in history and history['auc']:
                    metrics['AUC'] = max(history['auc'])
                else:
                    metrics['AUC'] = 'N/A'
                    
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
                metrics['Accuracy'] = 'Error'
                metrics['F1 Score'] = 'Error'
                metrics['AUC'] = 'Error'
        else:
            metrics['Accuracy'] = 'N/A'
            metrics['F1 Score'] = 'N/A'
            metrics['AUC'] = 'N/A'
            
        results_data.append(metrics)

# Add BioMedCLIP results
for dataset in datasets:
    biomed_json = os.path.join(base_path, 'biomedclip', f'results_{dataset}.json')
    if os.path.exists(biomed_json):
        try:
            with open(biomed_json, 'r') as f:
                data = json.load(f)
                # Ensure data is a list and take the first item as expected
                if isinstance(data, list) and len(data) > 0:
                    entry = data[0]
                    results_data.append({
                        'Method': 'BioMedCLIP',
                        'Dataset': dataset,
                        'Accuracy': entry.get('acc', 'N/A'),
                        'F1 Score': entry.get('f1', 'N/A'),
                        'AUC': entry.get('auc', 'N/A')
                    })
        except Exception as e:
            print(f"Error reading {biomed_json}: {e}")

# Add Tip-Adapter results
for dataset in datasets:
    tip_txt = os.path.join(base_path, 'tipadapter', 'results', f'{dataset}_biomed_tipadapter.txt')
    if os.path.exists(tip_txt):
        try:
            with open(tip_txt, 'r') as f:
                lines = f.readlines()
            
            # Temporary storage to group metrics by method
            tip_results = {
                'Tip-Adapter (Zero-Shot)': {'Accuracy': 'N/A', 'F1 Score': 'N/A', 'AUC': 'N/A'},
                'Tip-Adapter (Training-Free)': {'Accuracy': 'N/A', 'F1 Score': 'N/A', 'AUC': 'N/A'},
                'Tip-Adapter (Fine-Tuned)': {'Accuracy': 'N/A', 'F1 Score': 'N/A', 'AUC': 'N/A'}
            }
            
            for line in lines:
                line = line.strip()
                if not line: continue
                
                parts = line.split(':')
                if len(parts) < 2: continue
                
                key = parts[0].strip()
                val_str = parts[1].strip()
                
                # Helper to parse value
                def parse_val(s):
                    if s.endswith('%'):
                        return float(s.rstrip('%')) / 100.0
                    try:
                        return float(s)
                    except:
                        return 'N/A'

                val = parse_val(val_str)
                
                # Check key mapping
                if key == 'Zero-Shot Acc':
                    tip_results['Tip-Adapter (Zero-Shot)']['Accuracy'] = val
                elif key == 'Zero-Shot F1':
                    tip_results['Tip-Adapter (Zero-Shot)']['F1 Score'] = val
                elif key == 'Zero-Shot AUC':
                    tip_results['Tip-Adapter (Zero-Shot)']['AUC'] = val
                    
                elif key == 'Tip-Adapter Acc':
                    tip_results['Tip-Adapter (Training-Free)']['Accuracy'] = val
                elif key == 'Tip-Adapter F1':
                    tip_results['Tip-Adapter (Training-Free)']['F1 Score'] = val
                elif key == 'Tip-Adapter AUC':
                    tip_results['Tip-Adapter (Training-Free)']['AUC'] = val
                    
                elif key == 'Tip-Adapter-F Acc':
                    tip_results['Tip-Adapter (Fine-Tuned)']['Accuracy'] = val
                elif key == 'Tip-Adapter-F F1':
                    tip_results['Tip-Adapter (Fine-Tuned)']['F1 Score'] = val
                elif key == 'Tip-Adapter-F AUC':
                    tip_results['Tip-Adapter (Fine-Tuned)']['AUC'] = val

            # Append to results_data
            for method, metrics in tip_results.items():
                # Only add if we have at least an Accuracy (meaning the method was run)
                 # Or actually, we should add it if the file exists, as defaults are N/A.
                 # Let's check if 'Accuracy' is not N/A just to be safe, or just add all since file exists
                 if metrics['Accuracy'] != 'N/A':
                    results_data.append({
                        'Method': method,
                        'Dataset': dataset,
                        'Accuracy': metrics['Accuracy'],
                        'F1 Score': metrics['F1 Score'],
                        'AUC': metrics['AUC']
                    })
                    
        except Exception as e:
             print(f"Error reading {tip_txt}: {e}")

# Add DPE results
for dataset in datasets:
    dpe_json = os.path.join(base_path, 'dpe', 'results', f'results_{dataset}.json')
    if os.path.exists(dpe_json):
        try:
            with open(dpe_json, 'r') as f:
                data = json.load(f)
                # DPE json structure: {'dataset': '...', 'acc': ..., 'f1': ..., 'auc': ..., 'zs_acc':..., 'zs_f1':..., 'zs_auc':...}
                
                # Add DPE (TTA)
                results_data.append({
                    'Method': 'DPE',
                    'Dataset': dataset,
                    'Accuracy': data.get('acc', 'N/A'),
                    'F1 Score': data.get('f1', 'N/A'),
                    'AUC': data.get('auc', 'N/A')
                })
                
                # Add DPE (Zero-Shot) if available
                if 'zs_acc' in data:
                    results_data.append({
                        'Method': 'DPE (Zero-Shot)',
                        'Dataset': dataset,
                        'Accuracy': data.get('zs_acc', 'N/A'),
                        'F1 Score': data.get('zs_f1', 'N/A'),
                        'AUC': data.get('zs_auc', 'N/A')
                    })
        except Exception as e:
            print(f"Error reading {dpe_json}: {e}")

    # 4. CoOp Results
    coop_dir = os.path.join(workspace, 'coop', 'results')
    if not os.path.exists(coop_dir):
        coop_dir = os.path.join(workspace, 'coop', 'results_coop')
        
    if os.path.exists(coop_dir):
        coop_json = os.path.join(coop_dir, f'results_{dataset}.json')
        if os.path.exists(coop_json):
            try:
                with open(coop_json, 'r') as f:
                    data = json.load(f)
                    results_data.append({
                        'Method': 'CoOp',
                        'Dataset': dataset,
                        'Accuracy': data.get('acc', 'N/A'),
                        'F1 Score': data.get('f1', 'N/A'),
                        'AUC': data.get('auc', 'N/A')
                    })
            except Exception as e:
                print(f"Error reading {coop_json}: {e}")

# Create DataFrame
df = pd.DataFrame(results_data)

# Sort for better readability
df['Dataset'] = pd.Categorical(df['Dataset'], categories=datasets, ordered=True)
df = df.sort_values(['Dataset', 'Method'])

# Format values to 4 decimal places
for col in ['Accuracy', 'F1 Score', 'AUC']:
    df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

print("## Comparison of Methods across Datasets")
print(df.to_markdown(index=False))

# Also try a pivoted view
pivot_df = df.pivot(index='Dataset', columns='Method', values=['Accuracy', 'F1 Score', 'AUC'])
# print("\n## Pivoted Comparison")
# print(pivot_df.to_markdown()) 

for metric in ['Accuracy', 'F1 Score', 'AUC']:
    print(f"\n### {metric} Comparison")
    # For pivot to work with duplicate entries (if any error), we assume unique dataset-method pairs
    try:
        metric_pivot = df.pivot(index='Dataset', columns='Method', values=metric)
        print(metric_pivot.to_markdown())
    except Exception as e:
        print(f"Could not pivot for {metric}: {e}")
