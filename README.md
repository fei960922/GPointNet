# Generative PointNet: Deep Energy-Based Learning on Unordered Point Sets for 3D Generation, Reconstruction and Classification

This repository contains the official pytorch implementation for the paper "Generative PointNet: Deep Energy-Based Learning on Unordered Point Sets for 3D Generation, Reconstruction and Classification"

Please access the project page for more details, datasets and pretrained checkpoints downloads: [Project Page](http://www.stat.ucla.edu/~jxie/GPointNet).

The tensorflow implementation (tf1.14) can be found at: ~~[Will Announce Soon]()~~.

## Step to run 

### Download and Install 

- Clone this repo. 
- Download pre-processed dataset for ModelNet form the project page. Store it to data folder. (The urls are subject to change, please check the project page.)
- (Optional) If you want to use pretrained model, download checkpoints and store it to checkpoint folder. 
- (Optional) Create an environment through conda by the provided environment.yml
    - You can also manually install the package:
        - Python 3.6-3.8, pytorch==1.8.0, pytorch-lightning==1.2.1, etc.
    - Typically, pytorch==1.8.0 require cuda >= 10.2. If you only have cuda 10.0 or 10.1. You may install pytorch==1.4.0
- (Optional) In order to calculate the quantitive result, compile extra pytorch operator. You can skip this.
    - see FAQ if error occured.

```{bash}
    # Clone package
    git clone git@github.com:fei960922/GPointNet.git
    cd GPointNet

    # Download dataset and checkpoint
    wget http://www.stat.ucla.edu/~jxie/GPointNet/data/modelnet_2k.zip 
    unzip -q modelnet_2k.zip 
    mkdir checkpoint
    wget http://www.stat.ucla.edu/~jxie/GPointNet/checkpoint/syn_cvpr_chair.ckpt -O checkpoint/syn_cvpr_chair.ckpt

    # Establish the environment and compile metrics.
    conda env create -f environment.yml 
    conda activate gpointnet_gpu
    cd metrics/pytorch_structural_losses
    make
```

### Point Cloud Synthesis: Train from stratch 

Please make sure you download the datasets. 

```
python src/model_point_torch.py
```

By default, it run chair synthesis with default setting on a single GPU. It takes about 8 hours to train on Nvidia RTX2080 Ti.

Please check `src/model_point_torch.py` for argument details. If you have not compiled the metrics, please add `-do_evaluation 0` to skip the evaluation.

### Synthesis results from pretrained checkpoint 

```
python tools/test_torch.py -category chair -checkpoint_path {path}.ckpt -synthesis
```

Add `-reconstruction`, `-intepolation` to perform reconstruction and intepolation. Add `-evaluate` to output quantitive result.

```
python tools/test_torch.py -category chair -checkpoint_path {path}.ckpt -synthesis -evaluate -reconstruction -intepolation
```

### Do classification 

To run classification, please download and compile [Libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/). 

```
python tools/classification_torch.py -checkpoint_path output/checkpoint_default_big.ckpt
```

See `tools/run_examples.sh` for more examples.

## FAQ

### Common issue related to evaluation metric compile

- `cd metrics/pytorch_structural_losses`
- `make`, if failed:
    - Change `c++11` to `c++14` in Makefile:Line 69-70
    - Change nvcc path in Makefile:Line 9 to match current cuda version. 
        - If install pytorch with conda, nvcc is not installed by default. 
        - Install cuda: `conda install -c conda-forge nvcc_linux-64=11.1` (11.1 is the cuda version)
        - If so, the nvcc is in `~/miniconda3/envs/{ENV}/bin/nvcc`.
    - If error on ninja, change `setup.py:Line 23` to `cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)}`
    - If compiled successfully but no file found, change Makefile:Line74-75 accordingly. `mv` mean move files.

## Reference 

    @inproceedings{GPointNet,
        title={Generative PointNet: Deep Energy-Based Learning on Unordered Point Sets for 3D Generation, Reconstruction and Classification},
        author={Xie, Jianwen and Xu, Yifei and Zheng, Zilong and Gao, Ruiqi and Wang, Wenguan and Zhu Song-Chun and Wu, Ying Nian},
        booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2021}
    }
