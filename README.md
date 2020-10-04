# Mask Detection

## Prerequisites

1. [docker-ce](https://docs.docker.com/v17.09/engine/installation/linux/docker-ce/ubuntu/#install-docker-ce-1)
2. [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
3. [caffe-ssd 1.0.0-rc3'](https://github.com/weiliu89/caffe/tree/master)



## Configure Caffe Docker
1. Download [u16.04_x86_64_cv3.4_py3_caffessd.tar](https://fazendo_upload) or install [caffe-ssd](https://github.com/weiliu89/caffe/tree/master).
2. Run the following commands:
```
docker load --input /path/to/u16.04_x86_64_cv3.4_py3_caffessd.tar
```
```
git clone https://github.com/brunafrade/mask-detection.git
```

```
nvidia-docker run -it -v your_git_clone_path:/home/ --name $containername u16.04_x86_64_cv3.4_py3_caffessd:v2.3.0 bash

```
3. Install the text editor of your preference, e.g. `sudo apt install nano`.
4. The docker contains a version of Caffe that was compiled to work with a GeForce GTX 1050, which has computability of 6.1. If your GPU is not this one, run:
```
nvidia-smi --query-gpu=name --format=csv,noheader
```
5. This command will display the GPU you have. Look this GPU computability at [CUDA GPUs](https://en.wikipedia.org/wiki/CUDA#GPUs_supported).
6. If it is different from 6.1, use the text editor you installed to edit `CUDA_ARCH` attribute in `/home/caffe-ssd/Makefile.config` to the computability of your GPU, e.g. for a GeForce GTX 1050 the computability is 6.1, so `CUDA_ARCH := -gencode arch=compute_61,code=sm_61` .


8. Run the following commands:
```
cd /home/caffe-ssd/build/
```
```
rm -r *
```
```
cmake .. && make -j8
```
9. Run:
```
cd /home/
```

### Observation
1. If you need to start the container, you should run:
```
docker exec -it container_name bash
```

### Train
1. Dowload dataset [here](http://):


2. Edit files models/ssd_face/ssd_face_train_tunning.prototxt and models/ssd_face/ssd_face_test.prototxt and change dir:
```
/home//train_val_data_mask/VOC0712/lmdb/VOC0712_trainval_lmdb
```

```
labelmap/labelmap.prototxt
```

3. Run:
```
train.sh
```
