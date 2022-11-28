cuda_device=$1
test_name=$2
dataset=$3
protocol=$4
representation=$5

if [ ! -d ./checkpoints/${test_name} ];then
    mkdir -p ./checkpoints/${test_name}
fi

CUDA_VISIBLE_DEVICES=${cuda_device} python pretraining.py \
  --lr 0.01 \
  --batch-size 64 \
  --hico-t 0.2   --hico-k 2048 \
  --checkpoint-path ./checkpoints/${test_name} \
  --schedule 351  --epochs 451  --pre-dataset ${dataset} --protocol ${protocol} \
  --skeleton-representation ${representation} | tee -a ./checkpoints/${test_name}/pretraining.log
