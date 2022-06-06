# StyleNeRF: A Style-based 3D-Aware Generator for High-resolution Image Synthesis </sub>

This project is a submission for Machine Learning for 3D data course. We clearly state that our work is more closer to rewriting the code for small region with minor modification instead of implementation. All other parts except the part mentioned in the report are directly borrowed from the official code, remaining the whole pipeline. Note that the code is borrowed from https://github.com/facebookresearch/StyleNeRF. 

## Requirements
* Python 3.7
* PyTorch 1.7.1
* 8 NVIDIA GeForce RTX 2080 Ti GPUs with CUDA version 11.4

You can also use `requirements.txt` for python libraries (supported by the official code).

## Data Preparation
To train or inference, you must download the FFHQ dataset which is available in https://github.com/NVlabs/ffhq-dataset or you can use Kaggle e.g. https://www.kaggle.com/datasets/xhlulu/flickrfaceshq-dataset-nvidia-resized-256px. You should modify `data=${DATASET}` to train or inference a model.

## Train
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_train.py outdir=outputs data=${DATASET} spec=paper256 model=stylenerf_ffhq
```

## Inference
```bash
CUDA_VISIBLE_DEVICES=0 python generate.py --outdir=outputs --trunc=0.7 --seeds=${SEEDS} --network=${CHECKPOINT_PATH} --render-program="rotation_camera"
```

## Metrics
```bash
CUDA_VISIBLE_DEVICES=0 python calc_metrics.py --metrics=fid50k_full --data=${DATASET} --mirror=1 --network=${CHECKPOINT_PATH}
```

## Checkpoints
https://drive.google.com/file/d/16S6ECLGpsRlv0XB1fBat3psN9vPnrT4w/view?usp=sharing, https://drive.google.com/file/d/1EjFfpotCLnEBwDtfqB4Fk3ncSPhr4bGQ/view?usp=sharing, https://drive.google.com/file/d/1IT3zz7_r4VqOmy9HCkqe1psUeR_NKACK/view?usp=sharing, https://drive.google.com/file/d/1nJ-gxi5XMs4Ex7_dOfWhcUYzF9ITwwkZ/view?usp=sharing

## License
Copyright &copy; Facebook, Inc. All Rights Reserved.

The majority of StyleNeRF is licensed under [CC-BY-NC](https://creativecommons.org/licenses/by-nc/4.0/), however, portions of this project are available under a separate license terms: all codes used or modified from [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) is under the [Nvidia Source Code License](https://nvlabs.github.io/stylegan2-ada-pytorch/license.html).