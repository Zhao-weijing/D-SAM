### Overview

This repository provides code to train CIFAR models with **Sharpness-Aware Minimization (SAM)** and its variants (especially **D-SAM**), and to evaluate their robustness on the **CIFAR-10-C** corruption benchmark.

- **Training script**: `example_cifar.py`  
- **Corruption robustness evaluation**: `test_robustness_cifar10c.py`  
  (test accuracy on CIFAR-10-C under different corruption types and severities)

---

### Train with SAM / D-SAM (`example_cifar.py`)

`example_cifar.py` trains a model on CIFAR10 / CIFAR100 / SVHN / ImageNet and saves the **best checkpoint** on the validation set.

Key arguments (only the most useful ones):

- `--dataset` : `CIFAR10` / `CIFAR100` / `SVHN` / `ImageNet` (default: `CIFAR10`)
- `--model` : model name, e.g. `ResNet18`
- `--optimizer` : optimizer, important options:
  - `SAM` : standard SAM
  - `D_SAM` : D-SAM optimizer defined in `D_SAM.py`
  - **Note: For the best performance, it is highly recommended to use the D_FriendlySAM variant with --rho 5.**
- `--lr` : learning rate (e.g. `0.1`)
- `--momentum` : momentum (default `0.9`)
- `--weight_decay` : weight decay (default `5e-4`)
- `--rho` : SAM / D-SAM radius (default `0.5`)
- `--batch_size` : batch size (default `128`)
- `--epochs` : training epochs (default `200`)
- `--checkpoint_dir` : directory to save checkpoints and training history

#### Example: train CIFAR-10 with D-SAM

```bash
python example_cifar.py \
  --dataset CIFAR10 \
  --model ResNet18 \
  --optimizer D_SAM \
  --lr 0.1 \
  --momentum 0.9 \
  --weight_decay 5e-4 \
  --rho 0.5 \
  --batch_size 128 \
  --epochs 200 \
  --checkpoint_dir 
```

#### Example (recommended): train CIFAR-100 with D_FriendlySAM

```bash
python example_cifar.py \
  --dataset CIFAR100 \
  --model ResNet18 \
  --optimizer D_FriendlySAM \
  --lr 0.1 \
  --momentum 0.9 \
  --weight_decay 5e-4 \
  --rho 5 \
  --batch_size 128 \
  --epochs 200 \
  --checkpoint_dir 
```

The script automatically saves the best model as:

```text
best_{dataset}_{model}_batch_{...}_{optimizer}_lr_{...}_wd_{...}_mom_{...}_rho_{...}_epochs_{...}_seed_{...}_noise_{...}.pth
```

---

### CIFAR-10-C robustness evaluation (`test_robustness_cifar10c.py`)

`test_robustness_cifar10c.py` evaluates a trained CIFAR-10 model on the **CIFAR-10-C** corruption dataset.

#### Prepare CIFAR-10-C

1. Download CIFAR-10-C.  
2. Unzip it to a folder containing:
   - multiple corruption `.npy` files (e.g. `gaussian_noise.npy`, `fog.npy`, `jpeg_compression.npy`, …)
   - `labels.npy`
3. Set `--data_root` to this folder (default in the script: `/data/CIFAR-10-C`).

#### Main arguments

- `--checkpoint_path` : path to the trained `.pth` model (e.g. best checkpoint from `example_cifar.py`)
- `--data_root` : CIFAR-10-C root directory
- `--batch_size` : batch size (default `128`)
- `--num_workers` : dataloader workers (default `2`)
- `--output_dir` : directory to save CSV results
- `--seed` : random seed (default `0`)

The script:

- tests 15 corruption types (`gaussian_noise`, `shot_noise`, `impulse_noise`, `defocus_blur`, `glass_blur`, `motion_blur`, `zoom_blur`, `snow`, `frost`, `fog`, `brightness`, `contrast`, `elastic_transform`, `pixelate`, `jpeg_compression`);
- evaluates all 5 severities (1–5) for each type;
- prints per-severity and per-type accuracies;
- computes the overall mean accuracy;
- saves all results as a CSV file.

#### Example: evaluate a D-SAM CIFAR-10 model on CIFAR-10-C

```bash
python test_robustness_cifar10c.py \
  --checkpoint_path  \
  --data_root /data/weking/CIFAR-10-C \
  --batch_size 128 \
  --num_workers 2 \
  --output_dir 
```

This produces a CSV file in `output_dir`, e.g.:

```text
robustness_results_best_CIFAR10_ResNet18_batch_128_D_SAM_lr_0.1_wd_0.0005_mom_0.9_rho_0.5_epochs_200_seed_0_noise_0.0.csv
```

The CSV contains:

- accuracy for each corruption type and severity 1–5;
- average accuracy per type;
- overall average accuracy (last row).

