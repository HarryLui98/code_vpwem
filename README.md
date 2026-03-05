# VPWEM: Non-Markovian Visuomotor Policy with Working and Episodic Memory


## Installation

#### 1. Create and activate conda environment
```bash
$ conda create -n vpwem python==3.9
$ conda activate vpwem
```
#### 2. Install PyTorch
Install `torch` that is compatible with your CUDA version:
```bash
$ pip install torch==2.6.0+cu118 torchaudio==2.6.0+cu118 torchvision==0.21.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```
#### 3. Install MuJoCo
* Install the dependencies related to the mujoco-py environment with `apt`: 
```bash
$ sudo apt-get install libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev patchelf
```
* or with `conda`:
```bash
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
```
then add `export CPATH=$CONDA_PREFIX/include` to `~/.bashrc` and install patchelf with:

```bash
source ~/.bashrc
pip install patchelf
```

* Following https://github.com/openai/mujoco-py#install-mujoco:

  * Download the MuJoCo version 2.1 binaries for
    [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or
    [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz).
  * Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.

* The following environment paths should be specify in the `~/.bashrc`:
```bash
export MUJOCO_PATH=~/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```
* Install `mujoco-py`:
```bash
source ~/.bashrc
pip install "Cython<3" # mujoco-py is incompatible with cython 3
pip install mujoco-py==2.1.2.14
```
#### 4. Install Robosuite and Robomimic

```bash
pip install robosuite @ https://github.com/cheng-chi/robosuite/archive/277ab9588ad7a4f4b55cf75508b44aa67ec171f0.tar.gz 
pip install robomimic==0.2.0 # robomimic version matters in reproducing results
```
#### 5. Install Gym

```bash
pip install setuptools==65.5.0 pip==21  # gym 0.21 installation is broken with more recent versions
pip install wheel==0.38.0
pip install gym==0.21.0 gym-notices==0.0.8
```
#### 6. Install CleanDiffuser and VPWEM
For users who need to run `pipelines` and reproduce the results of the paper, they will need to install RL simulators.

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Repository Structure

```
vpwem/
├── cleandiffuser/          # Core library
│   ├── dataset/            # Dataset loaders (Robomimic, MoMaRT, MIKASA, etc.)
│   ├── diffusion/          # Diffusion models (DDPM, DDIM, EDM, etc.)
│   ├── env/                # Environment wrappers (Robomimic, MIKASA, etc.)
│   ├── nn_condition/       # Condition networks (image observation encoders)
│   ├── nn_diffusion/       # Diffusion backbone networks (ChiTransformer, etc.)
│   ├── nn_memory/          # Memory modules (Q-Former, KMeans, AdjSim, etc.)
│   └── utils/              # Utility functions
├── configs/dp/             # Hydra YAML config files
│   ├── robomimic_multi_modal/  # Robomimic benchmark configs
│   ├── momart/                 # MoMaRT benchmark configs
│   └── mikasa/                 # MIKASA benchmark configs
├── pipelines/              # Training and evaluation scripts
├── exp_scripts/            # Shell scripts for running experiments
└── requirements.txt
```

### Config Organization

Configs are organized by **benchmark** and **method**:

| Config Directory              | Method                          | Pipeline Prefix    |
| ----------------------------- | ------------------------------- | ------------------ |
| `chi_transformer/`            | Diffusion Policy (DP) | `dp_*`             |
| `chi_transformer_ptp/`        | DP with  Past Token Prediction (DP-PTP)                         | `dpptp_*`          |
| `mail/`                       | DP with Mamba Policy(MAIL)               | `mail_*`           |
| `chi_transformer_ptp_longmem/`| **VPWEM (Ours)**                | `longmemdpptp_*`   |

Pipeline scripts follow a naming convention:
- `*.py`: training scripts
- `*_emb.py`: use pre-computed observation embeddings from short-context policy (faster training)
- `*_eval.py` / `*_emb_eval.py` : evaluation scripts


### Pre-compute Observation Embeddings (Optional)

To speed up training, you can pre-compute observation embeddings using a trained short-context policy, which follows [Long-context Diffusion Policy](https://github.com/long-context-dp/ldp):

```bash
python pipelines/rewrite_with_embeddings_*.py
```

### Training

* Before running, update the following fields in the corresponding YAML config file under `configs/*/`:
- `dataset_path`: path to your dataset
- `save_path`: path to save model checkpoints

* You may also directly override any config parameter via command-line arguments:

```bash
python pipelines/longmemdpptp_robomimic_image_emb.py \
    --config-name=longsquare_abs_emb \
    seed=100 \
    mem_compress_length=2
```

### Evaluation

Evaluation scripts share the same config as training. Make sure `work_dir` points to the directory containing the saved model checkpoints.

```bash
# Evaluate VPWEM on Robomimic (LongSquare)
python pipelines/longmemdpptp_robomimic_lhsq_image_emb_eval.py \
    --config-name=longsquare_abs \
    seed=100
```

### Running with Shell Scripts

We provide example shell scripts in `exp_scripts/` that run training and evaluation across multiple seeds

## Acknowledgement

- [CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser)
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
- [Long-context Diffusion Policy](https://github.com/long-context-dp/ldp)
- [MaIL](https://github.com/ALRhub/MaIL)
