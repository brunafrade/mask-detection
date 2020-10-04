# License Plate Module

## Prerequisites

1. [docker-ce](https://docs.docker.com/v17.09/engine/installation/linux/docker-ce/ubuntu/#install-docker-ce-1)
2. [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
3. [NCSDK v1](https://github.com/movidius/ncsdk)


## Configure Caffe Docker
1. Download [polsec_caffe-v1.0.0.tar](https://polsecai.sharepoint.com/:u:/s/ai/EaLhOJLNV7VPtYOCvE5wcaEBIufqp1M1X93KSumOUrUEzQ?e=WLTTpq). If you're using the analytical server, this file is stored in `docker_images`.
2. Run the following commands:
```
docker load --input /path/to/polsec_caffe-v1.0.0.tar
```
```
git clone https://gitlab.com/polsec-ai/aipolsec/licence-plate-module.git
```
```
cd license-plate-module
```
```
sudo chmod +x scripts/*.sh
```
```
./scripts/run_docker.sh /path/to/license-plate-module/ your_name
```
3. Install the text editor of your preference, e.g. `apt install nano`.
4. The docker contains a version of Caffe that was compiled to work with a GeForce GTX 1050, which has computability of 6.1. If your GPU is not this one, run:
```
nvidia-smi --query-gpu=name --format=csv,noheader
```
5. This command will display the GPU you have. Look this GPU computability at [CUDA GPUs](https://en.wikipedia.org/wiki/CUDA#GPUs_supported).
6. If it is different from 6.1, use the text editor you installed to edit `CUDA_ARCH` attribute in `/home/caffe-ssd/Makefile.config` to the computability of your GPU, e.g. for a GeForce GTX 1050 the computability is 6.1, so `CUDA_ARCH := -gencode arch=compute_61,code=sm_61` .

7. Extract files in zip "multi.zip". Once again into Docker, copy .cpp files from multi folder to /home/caffe/src/caffe/layers/ and cppy .hpp files to /home/caffe/include/caffe/layers/ .

8. After you edit and copied the files, run the following commands:
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
cd /home/polsec_ai
```

### Observation
1. If you need to start the container, you should run:
```
./scripts/start_container.sh your_container_name_or_id
```


## License Plate Detector

### Prepare Dataset
1. Organize your images and annotations inside `datasets/dataset_name`. Images are to be stored in a dir called `jpg`, annotations are to be stored in a dir called `xml`, e.g. `datasets/lpd/jpg` and `datasets/lpd/xml`.
2. Run the following:
```
./scripts/configure_dataset.sh lpd xml/ jpg/ input_width input_height test_set_ratio
```
3. `lpd` is the dataset name we used for the license plate detector. `input_width`, `input_height` and `test_set_ratio` are, respectively, input width size, input height size and the ratio to divide training and testing sets, e.g. `300 300 0.01`.
4. This script will create the `lmdb` files that store the datasets used during training. It also creates two `txt` files containing the list of images used to train and test the model, e.g. `datasets/lpd/trainval_lpd.txt` and `datasets/lpd/test_lpd.txt`.

### Training Process 
1. Edit the files `protos/lpd_train.prototxt` and `protos/lpd_test.prototxt`:
    1.  `lpd_train.prototxt`: change `source` value to the `lmdb` files directory path created before, e.g. `"/home/polsec_ai/train_val_data/lpd/lmdb/trainval_lpd_lmdb/"`.
    2.  `lpd_train.prototxt`: change `label_map_file` to `"labelmap.prototxt"`.
    3. Repeat steps `1` and `2` for `lpd_test.prototxt`, changing `trainval_lpd_lmdb` to `test_lpd_lmdb`.
2. Edit the `solvers/solver_lpd.prototxt` file, change `snapshot_prefix` to the directory where weights are going to be stored, e.g. `snapshot_prefix: "snapshot_lpd/lpd"`. Also change the `net` and `test_net` parameters to the name of the prototxts for training and testing, if they're not `protos/lpd_train.prototxt` and `protos/lpd_test.prototxt`.
3. Run the following script to start training model:
```
./scripts/train_lpd.sh protos/lpd_train.prototxt snapshot_lpd solvers/solver_lpd.prototxt
```
4. If you stop the training process at some point, uncomment the line containing the parameter `--snapshot`, e.g. `--snapshot="snapshot_lpd/lpd_iter_10000.solverstate"`, where `iter_10000` refers to the model weights saved after 10000 iteration.

### Validation Process
1. Validation should be done out of the container you used to train the model.
2. Connect the Neural Compute Stick in your laptop.
3. Run the following:
```
./scripts/convert_validate.sh snapshot_lpd/ protos/deploy.prototxt lpd_results.txt shaves_number
```
4. `shaves_number` sets the numbers of shvaes/cores to be used when running the model on the NCS.
5. This script will validate all saved models and save the following information per model in `lpd_results.txt`:
    1. Iteration
    2. Mean Accuracy
    3. Mean Precision
    4. Mean Recall
6. This'll directly validate the model on the NCS. It can only be executed in a local machine.


## License Plate Classifier

### Prepare Dataset
1. Make sure you have all cropped license plate images (only plates) with their names in the pattern `AAA999_*.[jpg, png]`. These images must be stored in `datasets/dataset_name/jpg`, e.g. `datasets/lpr/jpg`. Also create a dir to save the labels, e.g. `datasets/lpr/labels`. In this case, we used `lpr` as the name of our dataset for the license plate classifier.
2. Run the following:
```
python3 scripts/convert_lpr_labels.py -d lpr -i jpg/ -o labels/
```
3. This script will save 8 files:
    1. `datasets/lpr/labels/labels[0-6].npy`: contains one-hot format labels for each plate in the dataset.
    2. `datasets/lpr/lpr_imgs.txt`: contains the paths to all images used to train/test the classifier, aligned with the label files above.
4. Enter the docker container you've configured.
5. Run the following:
```
./scripts/start_container.sh container_name_or_id
```
```
cd home/polsec_ai
```
```
python3 scripts/multilabel_lmdb.py --dataset lpr --images lpr_imgs.txt --labels labels/ --test test_set_ratio --val validation_set_ratio --width input_width --height input_height --shuffle [True, False]
```
6. This script will create the lmdb files to train/test the classifier, along with three `txt` files containing the images for train/test/validation, e.g. `datasets/lpr/lpr_train.txt`, `datasets/lpr/lpr_test.txt` and `datasets/lpr/lpr_val.txt`.

### Training Process
1. Edit file `protos/lpr_train.prototxt`
    1. For `Data` layers containing `phase: TRAIN`: 
        1. In layer with `top: data`, change `source` to `"/home/polsec_ai/train_val_data/lpr/lmdb/lpr_images"`.
        2. In layer with `top: label`, change `source` to `"/home/polsec_ai/train_val_data/lpr/lmdb/lpr_labels"`.
        3. In layer with `top: label_onehot`, change `source` to `"/home/polsec_ai/train_val_data/lpr/lmdb/lpr_labels_onehot"`.
    3. Repeat step `1.1` for `Data` layers containing `phase: TEST`, appending `_test` at the end of each path.
2. Edit the `solvers/solver_lpr.prototxt` file, change `snapshot_prefix` to the directory where weights are going to be stored, e.g. `snapshot_prefix: "snapshot_lpr/lpr"`. Also change the `net` parameter to the name of the prototxt for training, if it's not `protos/lpr_train.prototxt`.
3. Run the following to start training the model:
```
./scripts/train_lpr.sh protos/lpr_train.prototxt snapshot_lpr solvers/solver_lpr.prototxt
```
4. If you stop the training process at some point, uncomment the line containing the parameter `--snapshot`, e.g. `--snapshot="snapshot_lpr/lpr_iter_10000.solverstate"`, where `iter_10000` refers to the model weights saved after 10000 iteration.

### Validation Process
1. Run:
```
python3 scripts/validate_lpr.py -m lpr_deploy.prototxt -s snapshot_lpr/ -i lpr_val.txt -o lpr_results.txt
```
2. This script will validate all saved models and save the following information per model in `lpr_results.txt`:
    1. Mean Precision for each license plate digit [1-7]
    2. Mean Precision in general, i.e. percentage of license plates correctly recognized.
3. This'll only validate the models in the laptop/server.
