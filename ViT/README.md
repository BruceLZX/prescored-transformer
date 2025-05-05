# Prescored Transformer - ViT

This folder contains the code to test our prescored method on Vision Transformer (ViT) attention mechanisms. The implementation evaluates the effectiveness of the prescored approach in improving attention performance.

## Features
- Replace standard ViT self‑attention with a lightweight k‑means sampling (“prescored”) variant  
- Script A: classification evaluation on ImageNet–1k (reports loss & accuracy)  
- Script B: attention‑metric analysis (reports heavy‐attention capture % and overlap ratios)
## Files
- **`bench_monkey_patch.py`**  
  - Loads a pretrained ViT (e.g. `vit_large_patch16_224`)  
  - Applies `CustomKMeansAttention` (configurable clusters/samples/iterations)  
  - Runs a single‑pass evaluation on the ImageNet validation split  
  - Prints test loss and top‑1 accuracy  

- **`heavy_coverage_test.py`**  
  - Loads the same prescored‑attention ViT (e.g. `vit_base_patch16_224`)  
  - For each block, computes:  
    - **Overall Heavy Attention Capture %**: fraction of full‑attention’s high‑weight entries covered by sampled keys  
    - **Granular Overlap Ratio**: per‑head match rate between sampled indices and top‑weighted positions  
  - Outputs mean, median, min, and max statistics across the validation set  

## Usage
1. Clone the repository.
2. Navigate to this folder.
3. Install dependencies.
4. Run the provided scripts to test the prescored method (change parameters as needed).

## Requirements
- Python 3.x
- PyTorch
- Additional dependencies listed in `requirements.txt`.

# License
The code is licensed under the Apache 2.0 license.
