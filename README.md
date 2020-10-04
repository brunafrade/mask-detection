# Mask Detection

## Prerequisites

1. [docker-ce](https://docs.docker.com/v17.09/engine/installation/linux/docker-ce/ubuntu/#install-docker-ce-1)
2. [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)



## Configure Caffe Docker
1. Download [polsec_caffe-v1.0.0.tar](https://polsecai.sharepoint.com/:u:/s/ai/EaLhOJLNV7VPtYOCvE5wcaEBIufqp1M1X93KSumOUrUEzQ?e=WLTTpq).
2. Run the following commands:
```
docker load --input /path/to/u16.04_x86_64_cv3.4_py3_caffessd.tar
```
```
git clone https://gitlab.com/polsec-ai/aipolsec/licence-plate-module.git
```
```
cd license-plate-module

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

