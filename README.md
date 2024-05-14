
______________________________________________________________________

<div align="center">

# Profam


</div>

## Description

training a MoE model on CATH/FunFams/TED/UniRef50
[(Docs)](https://docs.google.com/document/d/1UptsPFMFTVyTEu-Ve75NfNpVNrzrWPJlyWvfhi2nsw4/edit)

 

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/alex-hh/profam.git
cd profam

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/alex-hh/profam.git
cd profam

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
