# 概要
  venvを利用することを前提とした研究環境
# 手順
## 1 docker 関連
### environments/docker/dockerfile
change these variables
- ARG PROJECT_NAME
- ARG USER_NAME
- ARG GROUP_NAME

### environments/docker/docker-compose.yaml
change these variables
- volumes
- BASE_IMAGE
  - check your environment's gpu
    ```
    $ nvidia-smi
    ```
  - check optimal cuda version for your gpu ([site](https://en.wikipedia.org/wiki/CUDA#GPUs_supported))
  - search base image ([site](https://hub.docker.com/r/nvidia/cuda/tags))

## 2　venv
```
$ python3 -m venv environments/work
$ source environments/work/bin/activate
(work)$ pip3 install -r requirements.txt
```
## 3 確認 | cudaと表示されればok
```
(work)$ python src/startup.py
cuda
```