# Parameter-Efficient Long-Tailed Recognition

Jiang-Xin Shi, Tong Wei, Zhi Zhou, Xin-Yan Han, Jie-Jing Shao, Yu-Feng Li

This is the source code for the paper: [Parameter-Efficient Long-Tailed Recognition](https://arxiv.org/abs/2309.10019)

<div align="center">
  <img src="./assets/framework.png" width="95%"/>
</div><br/>

## Requirements

* Python 3.8
* PyTorch 2.0
* Torchvision 0.15
* Tensorboard

- Other dependencies are listed in [requirements.txt](requirements.txt).

To install requirements, run:

```sh
conda create -n pel python=3.8 -y
conda activate pel
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install tensorboard
pip install -r requirements.txt
```

We encourage installing the latest dependencies. If there are any incompatibilities, please change the dependencies with the specified version [requirements-with-version.txt](requirements-with-version.txt).

## Hardware

All experiments can be reproduced using a single GPU with 20GB of memory.

- To further reduce the GPU memory cost, gradient accumulation is recommended. Please refer to [Usage](#usage) for detailed instructions.

## Quick Start on the CIFAR-100-LT dataset

```bash
# run PEL on CIFAR-100-LT (with imbalanced ratio=100)
python main.py -d cifar100_ir100 -m clip_vit_b16_peft
```

By running the above command, you can automatically download the CIFAR-100 dataset and run the method (PEL).

## Running on Large-scale Long-tailed Datasets

### Prepare the Dataset

Download the dataset [Places](http://places2.csail.mit.edu/download.html), [ImageNet](http://image-net.org/index), and [iNaturalist 2018](https://github.com/visipedia/inat_comp/tree/master/2018).

Put files in the following locations and change the path in the data configure files in [configs/data](configs/data):

- Places

```
Path/To/Dataset
├─ train
│  ├─ airfield
|  |  ├─ 00000001.jpg
|  |  └─ ......
│  └─ ......
└─ val
   ├─ airfield
   |  ├─ Places365_val_00000435.jpg
   |  └─ ......
   └─ ......
```

- ImageNet

```
Path/To/Dataset
├─ train
│  ├─ n01440764
|  |  ├─ n01440764_18.JPEG
|  |  └─ ......
│  └─ ......
└─ val
   ├─ n01440764
   |  ├─ ILSVRC2012_val_00000293.JPEG
   |  └─ ......
   └─ ......
```

- iNaturalist 2018

```
Path/To/Dataset
└─ train_val2018
   ├─ Actinopterygii
   |  ├─ 2229
   |  |  ├─ 2c5596da5091695e44b5604c2a53c477.jpg
   |  |  └─ ......
   |  └─ ......
   └─ ......
```

### Reproduction

To reproduce the main result in the paper, please run

```bash
# run PEL on ImageNet-LT
python main.py -d imagenet_lt -m clip_vit_b16_peft

# run PEL on Places-LT
python main.py -d places_lt -m clip_vit_b16_peft

# run PEL on iNaturalist 2018
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20
```

For other experiments, please refer to [scripts](scripts) for reproduction commands.

### Usage

To train and test the proposed method on more settings, run

```bash
python main.py -d [data] -m [model] [options]
```

The `[data]` can be the name of a .yaml file in [configs/data](configs/data), including `imagenet_lt`, `places_lt`, `inat2018`, `cifar100_ir100`, `cifar100_ir50` and `cifar100_ir10`.

The `[model]` can be the name of a .yaml file in [configs/model](configs/model), including `clip_vit_b16_peft`, `clip_vit_b16_ft`, and `zsclip_vit_b16`.

The `[options]` can allow the additional configure options included in [utils/config.py](utils/config.py). Following are some examples.

- To specify the root path of datasets, add `root Path/To/Datasets`.

- To change the output directory, add an option like `output_dir NewExpDir`. Then the results will be saved in `output/NewExpDir`.

- To assign a single GPU (for example, GPU 0), add an option like `gpu 0`.

- To apply gradient accumulation, add `micro_batch_size XX`. This can further reduce GPU memory costs. Note that `XX` should be a divisor of `batch_size`.

- To test an existing model, add `test_only True`. This option will test the model trained by your configure file. To test another model, add an additional option like `model_dir output/AnotherExpDir`.

- To test an existing model on the training set, add `test_train True`.

## Acknowledgment

We thank the authors for the following repositories for code reference:
[[OLTR]](https://github.com/zhmiao/OpenLongTailRecognition-OLTR), [[Classifier-Balancing]](https://github.com/facebookresearch/classifier-balancing), [[Dassl]](https://github.com/KaiyangZhou/Dassl.pytorch), [[CoOp]](https://github.com/KaiyangZhou/CoOp).

## Citation

If you find this repo useful for your work, please cite as:

```bibtex
@article{shi2023parameter,
  title={Parameter-Efficient Long-Tailed Recognition},
  author={Shi, Jiang-Xin and Wei, Tong and Zhou, Zhi and Han, Xin-Yan and Shao, Jie-Jing and Li, Yu-Feng},
  journal={arXiv preprint arXiv:2309.10019},
  year={2023}
}
```
