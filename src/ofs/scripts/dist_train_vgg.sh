MODEL_NAME=vgg

GPUS=$1
PORT=${PORT:-4321}

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \ 
CUDA_VISIBLE_DEVICES=1,2
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    ofs/train.py \
    --output_dir output \
    --model_name $MODEL_NAME \
    --batch_size 8 \
    --epochs 300 \
    --optim SGD \
    --lr 0.005 \
    --model_select zero \
    --launcher pytorch \
    