cuda_device=$1
test_name=$2
dataset=$3
protocol=$4
representation=$5

if [ ! -d ./checkpoints/${test_name} ];then
    mkdir -p ./checkpoints/${test_name}
fi

CUDA_VISIBLE_DEVICES=${cuda_device} python action_retrieval.py \
  --knn-neighbours 1 \
  --pretrained  ./checkpoints/${test_name}/checkpoint_0450.pth.tar \
  --finetune-dataset ${dataset} --protocol  ${protocol}  \
  --finetune-skeleton-representation ${representation} | tee -a ./checkpoints/${test_name}/retrieval.log