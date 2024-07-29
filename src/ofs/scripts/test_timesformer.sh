MODEL_NAME=timesformer

GPUNUM=$1

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \ 
export CUDA_VISIBLE_DEVICES=$GPUNUM
python3 ofs/test.py \
    --output_dir output \
    --load_dir output/timesformer/01_timesformer_None_numframe8/20240716_154746 \
    --use_metrics mse+js+balanced_acc \
    --num_frame 8 \
    --model_name $MODEL_NAME \
    --batch_size 16 \