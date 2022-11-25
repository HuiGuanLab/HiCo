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
- Replace the [data_path](https://github.com/HuiGuanLab/HiCo/blob/081f97dd341e6e1a5884d7e75a9189aa233e96a3/options/options_pretraining.py#L17) with your processed data location in the option files.

## Pretraining and Evaluation
Since our proposed HiCo consume less (due to smaller encoder and queue), we only implemented single GPU training.
```
CUDA_VISIBLE_DEVICES=0 python pretraining.py \
  --lr 0.01 \
  --batch-size 64 \
  --hico-t 0.2  --hico-k 2048 \
  --checkpoint-path ./checkpoints/ntu60_xview_joint \
  --schedule 351  --epochs 451  --pre-dataset ntu60 --protocol cross_view \
  --skeleton-representation joint
```
