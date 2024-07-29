MODEL_NAME=vgg
FRAME_SELECT=all
NUMFRAME=16

GPUNUM=$1

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \ 
export CUDA_VISIBLE_DEVICES=$GPUNUM \

python3 ofs/train.py \
    --name 01_${MODEL_NAME}_${FRAME_SELECT}_numframe${NUMFRAME}\
    --output_dir output \
    --model_name $MODEL_NAME \
    --batch_size 8 \
    --epochs 300 \
    --optim SGD \
    --lr 0.005 \
    --frame_select $FRAME_SELECT \
    