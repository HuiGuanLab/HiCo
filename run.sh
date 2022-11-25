cuda_device=$1
test_name=$2
representation=$3

if [ ! -d ./checkpoints/${test_name} ];then
    mkdir -p ./checkpoints/${test_name}
fi

CUDA_VISIBLE_DEVICES=${cuda_device} python pretraining.py \
  --lr 0.01 \
  --batch-size 64 \
  --hico-t 0.2   --hico-k 2048 \
  --checkpoint-path ./checkpoints/${test_name} \
  --schedule 351  --epochs 451  --pre-dataset ntu60 --protocol cross_view \
  --skeleton-representation ${representation} | tee -a ./checkpoints/${test_name}/pretraining.log
 
CUDA_VISIBLE_DEVICES=${cuda_device} python action_classification.py \
  --lr 2 \
  --batch-size 1024 \
  --pretrained  ./checkpoints/${test_name}/checkpoint_450.pth.tar \
  --finetune-dataset ntu60 --protocol cross_view \
  --finetune_skeleton_representation ${representation} | tee -a ./checkpoints/${test_name}/classification.log

CUDA_VISIBLE_DEVICES=${cuda_device} python action_retrieval.py \
  --knn-neighbours 1 \
  --pretrained  ./checkpoints/${test_name}/checkpoint_0450.pth.tar \
  --finetune-dataset ntu60 --protocol cross_view  \
  --finetune-skeleton-representation ${representation} | tee -a ./checkpoints/${test_name}/retrieval.log