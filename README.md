# CodeSLAM
PyTorch implementation of the conditional variational autoencoder (CVAE) in [CodeSLAM](https://arxiv.org/abs/1804.00874) for depth estimation. Implementation is based on https://github.com/lufficc/SSD.


## Requirements
### PyTorch
Choose your relevent PyTorch version here https://pytorch.org/get-started/locally/, by choosing correct system, pip/conda, GPU/CPU only. E.g for Linux using pip with no GPU, this would be

```
pip3 install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```
### Additional requirements
Download the additional requirements by
```
pip3 install -r requirements.txt
```

## Datasets
### SceneNet RGB-D
Download the SceneNet RGB-D training split(s) from https://robotvault.bitbucket.io/scenenet-rgbd.html, and place it in the datasets/ folder.

The folder structure of SceneNet RGB-D isn't compatible with how PyTorch loads data, so run ```prepare_scenenet.sh``` inside the datasets/SceneNetRGBD folder. You must give the name of the correct folder in the shell script, e.g
```
./prepare_scenenet.sh train_0
```
for configuring training split "train_0".

Example of a valid folder structure:
```
SCENENET_ROOT
|__ train_0
    |__ train
        |_ depth
        |_ instance
        |_ photo
|__ train_1
    |__ train
        |_ depth
        |_ instance
        |_ photo
|__ ...
```

## Configuration
Configuration files are located in configs/, where you can set parameters, location of trained model, demo images etc.

## Train
After downloading and placing the datasets correctly, do e.g
```
python3 train.py configs/scenenet_scratch.yaml
```
to train on the SceneNet RGB-D dataset.

Use Tensorboard to view logging metrics, by
```
tensorboard --logdir OUTPUT_DIR
```
in a new shell, where OUTPUT_DIR is the location of the outputs specified in the config file.

## Testing
After having trained a model, do e.g
```
python3 demo.py configs/scenenet_scratch.yaml
```
to test on a set of demo images located in demo/

## Adding new datasets, models, etc.
Check out https://github.com/lufficc/SSD/blob/master/DEVELOP_GUIDE.md for a guide on how to use custom datasets etc.