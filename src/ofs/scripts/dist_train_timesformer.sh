MODEL_NAME=timesformer
FRAME_SELECT=None
NUMFRAME=8

GPUS=$1
PORT=${PORT:-4321}

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \ 
CUDA_VISIBLE_DEVICES=0,1,2
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    ofs/train.py \
    --name 01_${MODEL_NAME}_${FRAME_SELECT}_numframe${NUMFRAME} \
    --output_dir output \
    --model_name $MODEL_NAME \
    --batch_size 8 \
    --epochs 300 \
    --optim SGD \
    --lr 0.005 \
    --num_worker 2 \
    --frame_select $FRAME_SELECT \
    --launcher pytorch \
    