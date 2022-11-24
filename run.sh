 CUDA_VISIBLE_DEVICES=0 python knn.py \
  --knn-neighbours 1 \
  --pretrained  ./checkpoints/test/checkpoint_0000_loss_33.657_acc_1.661.pth.tar \
  --finetune-dataset ntu60 --protocol cross_view  --finetune-skeleton-representation joint