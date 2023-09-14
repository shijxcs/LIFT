# Parameter-Efficient Long-Tailed Recognition

This is the source code for the paper: Parameter-Efficient Long-Tailed Recognition

### Requirements

* Python 3.8
* PyTorch 2.0
* Torchvision 0.15
* Tensorboard

- Other dependencies are listed in `requirements.txt`.

To install requirements, run:

```sh
conda create -n pel python=3.8 -y
conda activate pel
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install tensorboard
pip install -r requirements.txt
```

We encourage installing the latest dependencies, but if there are any incompatibilities, please change to the dependencies with the specified version `requirements-with-version.txt`.

### Hardware

All experiments can be reproduced using a single GPU with 20GB of memory.

### Quick Start on the CIFAR-100 dataset

```bash
python main.py -d cifar100_ir100 -m clip_vit_b16_peft
```

By running the above command, you can automatically download the CIFAR-100 dataset and run the method (PEL).

### Running on large-scale long-tailed datasets

#### Prepare the Dataset

Download the dataset [Places](http://places2.csail.mit.edu/download.html), [ImageNet](http://image-net.org/index), and [iNaturalist 2018](https://github.com/visipedia/inat_comp/tree/master/2018).

Put files in the following locations and change the path in the data configure files in `./configs/data`:

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

#### Usage

To train and test the proposed method, run

```bash
python main.py -d [data] -m [model] [options]
```

The `[data]` can be the name of a .yaml file in `configs/data`, including `imagenet_lt`, `places_lt`, `inat2018`, and `cifar100_ir100`.

The `[model]` can be the name of a .yaml file in `configs/model`, including `clip_vit_b16_peft`, `clip_vit_b16_ft`, and `zsclip_vit_b16`.

Taking the CIFAR-100-LT dataset as an example, you can 

The `[options]` can allow the additional configure options that are included in `utils/config.py`. Following are some examples.

- To specify the root path of datasets, add `root Path/To/Datasets`.

- To change the output directory, add an option like `output_dir NewExpDir`. Then the results will be saved in `output/NewExpDir`.

- To test an existing model, add `test_only True`. This option will test the model trained by your configure file. To test another model, add an additional option like `model_dir output/AnotherExpDir`.

- To assign a single GPU (for example, GPU 0), add an option like `gpu 0`.

### Acknowledgment

We thank the authors for the following repositories for code reference:
[[OLTR]](https://github.com/zhmiao/OpenLongTailRecognition-OLTR), [[Classifier-Balancing]](https://github.com/facebookresearch/classifier-balancing), [[Dassl]](https://github.com/KaiyangZhou/Dassl.pytorch), [[CoOp]](https://github.com/KaiyangZhou/CoOp).

