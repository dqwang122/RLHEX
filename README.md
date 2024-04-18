# MolExplainer
This is the repository for the paper: *Global Human-guided Counterfactual Explanations for Molecular Properties via Reinforcement Learning*.

## Setup

We recommend using a conda environment to set up.
```shell
$ conda create -n rlhex python=3.9
$ conda activate rlhex
$ pip install -r requirements.txt
$ pip install -e .
```

### Download Checkpoints

* To use PS-VAE as the backbone, download checkpoints from [PS-VAE](https://github.com/THUNLP-MT/PS-VAE) and put it under the *PS-VAE/ckpts* directory.
* To re-use the GNN classifier trained by GCFExplainer, download checkpoints from [GCFExplainer](https://github.com/mertkosan/GCFExplainer) and put it under the *data* directory.

Note: You can change the default checkpoint path by editing the config files under *configs/\*.yaml*


## Baselines

We integrate several baselines into our repo for reproduction.

For PSVAE and PSVAE-SA, you can directly use the following commands:

``` shell
# PSVAE
$ python sample.py --config configs/baseline.yaml  # sample CF candidates
$ python summary.py --config configs/summary.yaml  # summarize the top-k candidates from the candidate set

# PSVAE-SA on the aids dataset
$ python sample.py --config configs/config_aids.yaml
$ python summary.py --config configs/summary_aids.yaml
```

The first time you run these commands, it will process the *data/\*.smi* and save the *.pt* files to the *data/*.

We add the validity check to the original GCFExplainer for a fair comparison. You can use the code under *baselines/GCFExplainer* to run the experiments:

``` shell
$ cd baselines/GCFExplainer
$ python vrrw.py --dataset aids # random walk to get the CF candidates
$ python summary.py --dataset aids  # summarize the top-k candidates from the candidate set
```


### Training

we use the [PPO framework](https://github.com/ericyangyu/PPO-for-Beginners/tree/master) for our RLHEX. The main code lies under the *src/* directory. To get a better understanding of the PPO framework, please refer to the original repo.


There are some extra requirements for the PPO training. To set up:

``` shell
$ conda activate rlhex
$ cd src
$ pip install -r requirements.txt
```


The main entry is *src/main.py* and the options are listed in *src/arguments.py*. Please check the meaning of different options before running the experiment.

For training on the aids, a typical command is like:
``` shell
$ python main.py --dataset aids \
    --temp 1 --beam_size 10 \
    --lr 2e-4 --batch_size 64 \
    --eval_every 10 --max_timesteps_per_episode 1 \
    --obs_type emb --action_type z --ucb \
    --cov_weight 1 --pred_weight 1 \
    --wandb --name train_on_aids
```

### Inference

For evaluation, the command is like:
``` shell
$ python main.py \
    --dataset aids \
    --mode test \
    --temp 1 --beam_size 10 \
    --lr 2e-4 --batch_size 64 \
    --eval_every 10 --max_timesteps_per_episode 1 \
    --obs_type emb --action_type z \
    --ucb --cov_weight 1 --pred_weight 1 \
    --ckpt_dir <path-to-the-checkpoint>
```
The result file will be put under *results/aids/test_runs/*.
