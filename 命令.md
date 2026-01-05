# MIAS
python fpaco/generate_offline_detections.py --dataset /Users/liruirui/Documents/code/study/FPaCo/datasets/mias_classification_dataset --limit 3
python fpaco/visualize_detections.py --dataset /Users/liruirui/Documents/code/study/FPaCo/datasets/mias_classification_dataset --limit 3 --output-dir /Users/liruirui/Documents/code/study/FPaCo/fpaco/vis_results

# Oral
python fpaco/generate_offline_detections.py --dataset /Users/liruirui/Documents/code/study/FPaCo/datasets/oral_cancer_classification_dataset --limit 3
python fpaco/visualize_detections.py --dataset /Users/liruirui/Documents/code/study/FPaCo/datasets/oral_cancer_classification_dataset --limit 3 --output-dir /Users/liruirui/Documents/code/study/FPaCo/fpaco/vis_results

# APTOS
python fpaco/generate_offline_detections.py --dataset /Users/liruirui/Documents/code/study/FPaCo/datasets/aptos_classification_dataset --limit 3
python fpaco/visualize_detections.py --dataset /Users/liruirui/Documents/code/study/FPaCo/datasets/aptos_classification_dataset --limit 3 --output-dir /Users/liruirui/Documents/code/study/FPaCo/fpaco/vis_results

# Fingerprint
python fpaco/generate_offline_detections.py --dataset /Users/liruirui/Documents/code/study/FPaCo/datasets/fingerA --limit 3
python fpaco/visualize_detections.py --dataset /Users/liruirui/Documents/code/study/FPaCo/datasets/fingerA --limit 3 --output-dir /Users/liruirui/Documents/code/study/FPaCo/fpaco/vis_results

# OCTA
python fpaco/generate_offline_detections.py --dataset /Users/liruirui/Documents/code/study/FPaCo/datasets/octa_classification_dataset --limit 3
python fpaco/visualize_detections.py --dataset /Users/liruirui/Documents/code/study/FPaCo/datasets/octa_classification_dataset --limit 3 --output-dir /Users/liruirui/Documents/code/study/FPaCo/fpaco/vis_results





# Create a directory for Gemini test outputs to avoid overwriting Florence files
mkdir -p fpaco/gemini_vis_test

# 1. MIAS
echo "Running Gemini on MIAS..."
python fpaco/generate_gemini_heatmaps.py --dataset /Users/liruirui/Documents/code/study/FPaCo/datasets/mias_classification_dataset --output-dir fpaco/gemini_vis_test/mias --limit 3 --model gemini-3-flash-preview
python fpaco/visualize_detections.py --dataset fpaco/gemini_vis_test/mias --limit 3 --output-dir fpaco/gemini_vis_result_imgs

# 2. Oral
echo "Running Gemini on Oral..."
python fpaco/generate_gemini_heatmaps.py --dataset /Users/liruirui/Documents/code/study/FPaCo/datasets/oral_cancer_classification_dataset --output-dir fpaco/gemini_vis_test/oral --limit 3 --model gemini-3-flash-preview
python fpaco/visualize_detections.py --dataset fpaco/gemini_vis_test/oral --limit 3 --output-dir fpaco/gemini_vis_result_imgs

# 3. APTOS
echo "Running Gemini on APTOS..."
python fpaco/generate_gemini_heatmaps.py --dataset /Users/liruirui/Documents/code/study/FPaCo/datasets/aptos_classification_dataset --output-dir fpaco/gemini_vis_test/aptos --limit 3 --model gemini-3-flash-preview
python fpaco/visualize_detections.py --dataset fpaco/gemini_vis_test/aptos --limit 3 --output-dir fpaco/gemini_vis_result_imgs

# 4. Fingerprint A
echo "Running Gemini on Fingerprint A..."
python fpaco/generate_gemini_heatmaps.py --dataset /Users/liruirui/Documents/code/study/FPaCo/datasets/fingerA --output-dir fpaco/gemini_vis_test/fingerA --limit 3 --model gemini-3-flash-preview
python fpaco/visualize_detections.py --dataset fpaco/gemini_vis_test/fingerA --limit 3 --output-dir fpaco/gemini_vis_result_imgs

# 5. OCTA
echo "Running Gemini on OCTA..."
python fpaco/generate_gemini_heatmaps.py --dataset /Users/liruirui/Documents/code/study/FPaCo/datasets/octa_classification_dataset --output-dir fpaco/gemini_vis_test/octa --limit 3 --model gemini-3-flash-preview
python fpaco/visualize_detections.py --dataset fpaco/gemini_vis_test/octa --limit 3 --output-dir fpaco/gemini_vis_result_imgs