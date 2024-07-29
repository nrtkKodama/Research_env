MODEL_NAME=vgg

GPUNUM=$1

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \ 
export CUDA_VISIBLE_DEVICES=$GPUNUM \ 
python3 ofs/test.py \
    --output_dir output \
    --load_dir output/vgg/01_vgg_random_numframe1/20240710_065544 \
    --use_metrics mse+js+balanced_acc \
    --model_name $MODEL_NAME \
    --batch_size 16 \