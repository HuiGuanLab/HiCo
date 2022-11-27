# Hierarchical Contrast for Unsupervised Skeleton-based Action Representation Learning
This is a repository contains the implementation of our AAAI'23 paper [Hierarchical Contrast for Unsupervised Skeleton-based Action Representation Learning]().

![image](./fig/hico.png)

## Requirements
![.](https://img.shields.io/badge/Python-3.9-yellow) ![.](https://img.shields.io/badge/Pytorch-1.12.1-yellow)  
Use the following instructions to create the corresponding conda environment. 
```
conda create -n hico python=3.9 anaconda
conda activate hico
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch
pip3 install tensorboard
```

## Data Preparation
- Download raw [NTU-RGB+D 60 and 120 skeleton data](https://github.com/shahroudy/NTURGB-D) and save to ./data folder.
- Preprocess data with with `data_gen/ntu_gendata.py`.
- For convenience, we also provide preprocessed data in [Google Drive]().
<!--- - After preprocessing, replace the [data_path](https://github.com/HuiGuanLab/HiCo/blob/081f97dd341e6e1a5884d7e75a9189aa233e96a3/options/options_pretraining.py#L17) with your data location in the option files (`option_pretraining.py`, `option_classification.py` and `option_retrieval.py`). -->

## Pretraining and Evaluation
HiCo consumes less (due to smaller encoders and queues), so we only implemented single GPU training.
#### Unsupervised Pretraining
- eg. Train on NTU-RGB+D 60 cross-view joint stream.
```
CUDA_VISIBLE_DEVICES=0 python pretraining.py \
  --lr 0.01 \
  --batch-size 64 \
  --hico-t 0.2  --hico-k 2048 \
  --checkpoint-path ./checkpoints/ntu60_xview_joint \
  --schedule 351  --epochs 451  --pre-dataset ntu60 --protocol cross_view \
  --skeleton-representation joint
```
#### Downstream Task Evaluation
- eg. Skeleton-based action recognition. Train a linear classifier on pretrained query encoder.
```
CUDA_VISIBLE_DEVICES=0 python action_classification.py \
  --lr 2 \
  --batch-size 1024 \
  --pretrained  ./checkpoints/ntu60_xview_joint/checkpoint_450.pth.tar \
  --finetune-dataset ntu60 --protocol cross_view \
  --finetune_skeleton_representation joint
```
- eg. Skeleton-based action retrieval. Apply a KNN classifier on on pretrained query encoder.
```
CUDA_VISIBLE_DEVICES=0 python action_retrieval.py \
  --knn-neighbours 1 \
  --pretrained  ./checkpoints/ntu60_xview_joint/checkpoint_0450.pth.tar \
  --finetune-dataset ntu60 --protocol cross_view  \
  --finetune-skeleton-representation joint
```
## Pretrained Models
We release several pretrained models:
- HiCo-GRU, NTU-60 and NTU-120: [released_models]()
- HiCo-LSTM, NTU-60 and NTU-120: [released_models]()
- HiCo-Transformer, NTU-60 and NTU-120: [released_models]()  

Expected performance on skeleton-based action recognition:  

|     Model        | NTU 60 xsub (%) | NTU 60 xview (%) |   NTU 120 xsub (%)   |   NTU 120 xset (%)   |
| :--------------: | :-------------: | :--------------: | :-----------------:  | :-----------------:  |
| HiCo-GRU         |      80.6      |      88.6         |       72.5           |      73.8            |
| HiCo-LSTM        |      81.4      |      88.8         |       73.7           |      74.5            |
| HiCo-Transformer |      81.1      |      88.6         |       72.8           |      74.1            | 

## Citation
If you find this repository useful, please consider citing our paper:
```
@inproceedings{hico2023,
  title={ Hierarchical Contrast for Unsupervised Skeleton-based Action Representation Learning},
  author={Jianfeng Dong and Shengkai Sun and Zhonglin Liu and Shujie Chen and Baolong Liu and Xun Wang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```

## Visualization
We utilize t-SNE to visualize the learned action representation obtained by our HiCo-Transformer models with different granularities on NTU-60 xsub.


## Acknowledgement
The framework of our code is based on [skeleton-contrast](https://github.com/fmthoker/skeleton-contrast).  
  
This work was supported by the NSFC (61902347,
62002323, 61976188), the Public Welfare Technology Research
Project of Zhejiang Province (LGF21F020010), the
Open Projects Program of the National Laboratory of Pattern
Recognition, the Fundamental Research Funds for the
Provincial Universities of Zhejiang.
