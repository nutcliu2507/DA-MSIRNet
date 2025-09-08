# Image Inpainting with PyTorch

## Environment
- Python: 3.8
- PyTorch: 1.8.1
- GPU: Single RTX 3090 (all experiments are conducted on a single GPU)

Make sure to install compatible CUDA and CuDNN versions for PyTorch 1.8.1.  
It is recommended to use a virtual environment.

---

## Visualization (Visdom)
Before training, start the Visdom server (default port: 8097):
```
python -m visdom.server
```
Once launched, you can monitor training at:  
http://localhost:8097

---

## Training
Basic training command (replace {} with your settings):
```
python train.py --name {your_name} --img_file {your_image_path} --niter {your_niter} --mask_type [2,4] --batchSize 4 --lr 1e-4 --gpu_ids 0 --no_augment --no_flip --no_rotation
```
Experimental Settings:
- Batch Size: 4
- Optimizer: AdamW (β1 = 0.5, β2 = 0.9)
- Learning Rate: 1e-4
- Input Resolution: 256 × 256
- Training niter:
  - CelebA-HQ: 650,000
  - Places2: 1,800,000
  - FFHQ: 1,700,000
  - Paris: 260,700

Mask Types (--mask_type):
- 0: center mask
- 1: random regular mask
- 2: random irregular mask
- 3: external irregular mask
- 4: random freeform mask

Resume Training:
To continue training, use the same --name and add --continue_train:

---

## Testing / Inference
```
python test.py --name {your_name} --checkpoints_dir ./checkpoints/ --mask_type [3] --gpu_ids 0 --img_file {your_image_path} --mask_file {your_mask_path} --batchSize 1 --results_dir {your_image_result_path}
```
The option --no_shuffle can be used to keep a fixed testing order for reproducibility.
---

## License
This project is based on the original work by huangwenwenlili (2023), licensed under the MIT License.  
This project itself is also released under the MIT License.

- Original Copyright (c) 2023 huangwenwenlili
- Modifications Copyright (c) 2024 Chun-Chieh Chang
