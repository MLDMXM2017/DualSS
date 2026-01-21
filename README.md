# Introduction
This repository contains the official implementation for research on developing DualSS, A Dual-Diffusion Synthetic Sampling Framework for Screening Fatty Liver Disease Severity with Class-Imbalanced Tongue Images. The proposed DualSS first trains a disease-agnostic dual-diffusion generator (DDG) on tongue-image data, leveraging abundant negative and unlabeled samples to learn unified tongue representations. Building on DDG, we design a latent space synthetic sampling pipeline that synthesizes realistic positive-class images to construct a more balanced augmented training dataset. We then couple this pipeline with a noise-robust training strategy on the augmented dataset to further enhance the learning of downstream diagnostic models.

*Some components, configurations, and scripts may be subject to further refactoring or extension in future updates.

# Usage

    conda env create -f environment.yaml
    pip install -r requirements.txt
    pip install -e ./src/clip/
    pip install -e ./src/taming-transformers/
    pip install -e .
    python setup.py install

    python main_train_generator.py --base configs/autoencoder/RDVAE_vq8192_32x32x12_2decoder.yaml -t --gpus 0
    python main_train_generator.py --base configs/autoencoder/CADM_vq8192_32x32x12_2decoder.yaml -t --gpus 0
    python main_sample.py
    python main_train_diagnosis.py

# Reference
[https://github.com/CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)  
[https://github.com/xingjunm/dimensionality-driven-learning](https://github.com/xingjunm/dimensionality-driven-learning)

