# One Neuron to Fool Them All

## Prerequisites

- Install [this](https://github.com/iamgroot42/robustness) fork of the robustness package `pip install -e robustness`
- If you are going to run experiments for Imagenet, modify `IMAGENET_PATH` in `utils.py` accordingly
- Download pretrained models and pre-computed statistics:
  `wget https://www.dropbox.com/s/rsxzw30fdmle2qu/data.tar.gz?dl=1 -O data.tar.gz`
- Extract files
  `tar -xf data.tar.gz`

## Pre-Computing Statistics (skip if downloaded file above)

### Generating feature-wise statistics ($\mu$, $\sigma$)
- Given any model and dataset, calculate the feature-wise mean and standard deviation across the dataset. Computes across training and validation data for CIFAR10, and only validation data for Imagenet
- `python act_values.py <model_arch> <model_type/path> <output_prefix_path_> <cifar10/imagenet>`

### Generating $\Delta(i,x)$ values
- Given a model (assumes positive range of features, true for all architectures in codebase via ReLU) and dataset, computes the $\Delta(i,x)$ $\forall i,x$ and saves them for later use (generating attack seeds)
- `python multiclass_deltas.py <model_arch> <model_type/path> <cifar10/imagenet> <output_file_path>`

## Neuron-sensitivity Attack

### Generating adversarial examples using sensitive neurons
- Given a model and corresponding feature statistics and $\Delta(i,x)$ values (computed above), find adversarial seeds within specific perturbation budgets
- `python optimal_impostor.py`

## Training for Sensitivity

### Training using proposed regularization term
- Much faster than adversarial training
- Logs $L_2$ PGD attack success rates on validation set while training (to monitor robustness)
- `python sensitivity_training.py --output_dir <output_model_path>`

### Pruning neurons from trained model, based on sensitivity
- Given a trained model and dataset, use $\Delta(i,x)$ values to identify and prune weights that correspond to specific features
- Can prune `random` (randomly sample), `least` (zero out least sensitive first), or `most` (zero out most sensitive first)
- Prune `N` neurons (from features layer)
`python delta_defense.py <model_arch> <model_type/path> <random/most/least> <N> <output_model_path`
