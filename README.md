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
- "SERVICE_NAME" (make sure to use small letter)
 ```
 services:
  SERVICE_NAME:
 ```
- volumes
- BASE_IMAGE
  - check your environment's gpu
    ```
    $ nvidia-smi
    ```
  - check optimal cuda version for your gpu ([site](https://en.wikipedia.org/wiki/CUDA#GPUs_supported))
  - search base image ([site](https://hub.docker.com/r/nvidia/cuda/tags))

### environments/docker/Makefile
- change "SERVICE_NAME"

### build and start docker container
```
$cd {PROJECT_NAME}/environments/docker
$make all #build and exec
```

## 2　venv
```
$ python3 -m venv environments/{set your own env.(work)}
$ source environments/{set your own env.(work)}/bin/activate
(work)$ pip3 install -r requirements.txt
```

## 3 PyTorch関連のインストール
- CUDAのバージョンに合ったpytorchをインストールする
- CUDAのバージョンの確認
```
$ nvcc -V
```
- pytorchの[サイト](https://pytorch.org/get-started/previous-versions/)からちょうど良いやつを探してpipインストール
- ### pipでインストールする時には必ずvenv内で！

## 4 確認 | cudaと表示されればok
```
(work)$ python src/startup.py
cuda
```