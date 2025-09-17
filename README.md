# Overview
This is the code for the Densely Crowded Holstein-Friesian Cow Re-Identification pipeline. It consists of two parts. More info can be found [here](https://phoenix4582.github.io/dazzlecows.github.io/)

# AutoMaskPrompt
This folder contains the pipeline using off-the-shelf OWL-v2 and SAM2 for automated cow data acquisition.
## Setup
The libraries `transformers` from **HuggingFace** and `Ultralytics` from **Ultralytics** are the core prerequisites for customised dataset generation.

# SelfReID
This folder holds the code of the unsupervised contrastive learner built on PyTorch and PyTorch-Lightning.
## Setup
We strongly recommend using Anaconda for setting up the environment. Once you have installed Anaconda, simply type:
```
conda env create --name your_envname --file=lightning_id.yaml
```
to build up the environment

# Additional resources

For the inference of SOTA models from HuggingFace and Ultralytics, additional resources can be found at:

HuggingFace: https://huggingface.co/docs/transformers/en/installation

SAM2: https://docs.ultralytics.com/models/sam-2/

For the contrastive learning module based on PyTorch and PyTorch Lightning modules, additional documentation and tutorials can be found at:

PyTorch Vision: https://pytorch.org/vision/stable/index.html

PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html
