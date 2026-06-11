import os
import json

def extract_metrics(base_dir):
    results = []
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return results
    
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        
        history_path = os.path.join(subdir_path, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            best_val_acc = max(history.get('val_acc', [0]))
            best_val_f1 = max(history.get('val_f1', [0]))
            
            res = {
                'config': subdir,
                'best_val_acc': best_val_acc,
                'best_val_f1': best_val_f1
            }
            # Parse hyperparameters from folder name if possible
            parts = subdir.split('_')
            for part in parts:
                if 'sigma' in part:
                    res['sigma'] = part.replace('sigma', '')
                elif 'lr' in part:
                    res['lr'] = part.replace('lr', '')
                elif 'beta' in part:
                    res['beta'] = part.replace('beta', '')
                elif 'q' in part and part.startswith('q'):
                    res['queue_size'] = part.replace('q', '')
                elif 't' in part and part.startswith('t'):
                    res['temp'] = part.replace('t', '')
            
            results.append(res)
    return results

tuning_dir = '/Users/liruirui/Documents/code/study/finger/multimodal_classification/results/bpaco_tuning'
advanced_dir = '/Users/liruirui/Documents/code/study/finger/multimodal_classification/results/bpaco_tuning_advanced'
refined_dir = '/Users/liruirui/Documents/code/study/finger/multimodal_classification/results/bpaco_refined_tuning'

tuning_results = extract_metrics(tuning_dir)
advanced_results = extract_metrics(advanced_dir)
refined_results = extract_metrics(refined_dir)

def print_results(title, results):
    print(f"### {title}")
    if not results:
        print("No results found.")
        return
    
    # Sort by best_val_acc
    results.sort(key=lambda x: x['best_val_acc'], reverse=True)
    
    # Get all keys for header
    keys = set()
    for r in results:
        keys.update(r.keys())
    keys = sorted(list(keys))
    
    # Move 'config', 'best_val_acc', 'best_val_f1' to front
    priority = ['config', 'best_val_acc', 'best_val_f1']
    for p in reversed(priority):
        if p in keys:
            keys.remove(p)
            keys.insert(0, p)
    
    header = " | ".join(keys)
    separator = " | ".join(["---"] * len(keys))
    print(header)
    print(separator)
    
    for r in results:
        row = " | ".join([str(r.get(k, '')) for k in keys])
        print(row)

print_results("BPaCo Tuning Results", tuning_results)
print("\n")
print_results("BPaCo Tuning Advanced Results", advanced_results)
print("\n")
print_results("BPaCo Refined Tuning Results", refined_results)
