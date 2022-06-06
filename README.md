# StyleNeRF: A Style-based 3D-Aware Generator for High-resolution Image Synthesis </sub>

This project is a submission for Machine Learning for 3D data course. We mention the part we rewritten the code in the report and all others are directly borrowed from the official code. Note that the code is borrowed from https://github.com/facebookresearch/StyleNeRF. 

## Requirements
* Python 3.7
* PyTorch 1.7.1
* 8 NVIDIA GeForce RTX 2080 Ti GPUs with CUDA version 11.4

You can also use `requirements.txt` for python libraries (supported by the official code).

## Data Preparation
To train or inference, you must download the FFHQ dataset. We provide the dataset as a zip file. You should modify `data=${DATASET}` to train or inference a model.

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

## License
Copyright &copy; Facebook, Inc. All Rights Reserved.

The majority of StyleNeRF is licensed under [CC-BY-NC](https://creativecommons.org/licenses/by-nc/4.0/), however, portions of this project are available under a separate license terms: all codes used or modified from [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) is under the [Nvidia Source Code License](https://nvlabs.github.io/stylegan2-ada-pytorch/license.html).