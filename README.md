# Hierarchical Contrast for Unsupervised Skeleton-based Action Representation Learning
This is a repository contains the implementation of our AAAI'23 paper:

```
@inproceedings{xxx,
  title={ Hierarchical Contrast for Unsupervised Skeleton-based Action Representation Learning},
  author={xxx},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={1},
  number={1},
  pages={1--1},
  year={2023}
}
```
![image](./fig/hico.png)

## Requirements
![.](https://img.shields.io/badge/Python-3.9-yellow) ![.](https://img.shields.io/badge/Pytorch-1.12.1-yellow)  
You can use the following instructions to create the corresponding conda environment. 
```
conda create -n hico python=3.9 anaconda
conda activate hico
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch
pip3 install tensorboard
```

## Data Preprocessing
- Download raw [NTU-RGB+D 60 and 120](https://github.com/shahroudy/NTURGB-D).
- Follow the [skeleton-contrast](https://github.com/fmthoker/skeleton-contrast) data prepreprocessing instructions.
- Replace the  data location in the option files.
