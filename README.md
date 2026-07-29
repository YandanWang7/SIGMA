# SIGMA

Official implementation of Supervision Intensity-Guided Adaptive One-to-Many Matching for Long-Tailed Object Detection.

## Environment

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Build the multi-scale deformable attention CUDA extension:

```bash
cd models/dino/ops
python setup.py build install
cd ../../..
```

Use a PyTorch/CUDA version compatible with GPU driver. The original experiments used CUDA GPUs and distributed training.

## Dataset

Prepare LVIS/COCO-format data outside the repository. A typical layout is:

```text
/path/to/lvis/
  annotations/
    lvis_v1_train.json
    lvis_v1_val.json
  train2017/
  val2017/
```

Pass the dataset root with `--coco_path`.

## Checkpoint

The released checkpoint is provided at:

```text
checkpoints/checkpoint_best_regular.pth
```

This file is larger than GitHub's regular file-size limit and should be tracked with Git LFS.

## Evaluation

Run evaluation with the default configuration:

```bash
python main.py \
  --eval \
  --resume checkpoints/checkpoint_best_regular.pth \
  --coco_path /path/to/lvis \
  --output_dir outputs/eval_final
```

For distributed evaluation:

```bash
torchrun --nproc_per_node=8 main.py \
  --eval \
  --resume checkpoints/checkpoint_best_regular.pth \
  --coco_path /path/to/lvis \
  --output_dir outputs/eval_final \
  --num_workers 2
```

## Training

Run training with the default configuration:

```bash
torchrun --nproc_per_node=8 main.py \
  --dataset_file lvis \
  --coco_path /path/to/lvis \
  --output_dir outputs/train_final \
  --num_workers 4
```
