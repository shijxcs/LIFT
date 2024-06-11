# LIFT on ImageNet-LT
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True test_only True tte True

# LIFT on Places-LT
python main.py -d places_lt -m clip_vit_b16 adaptformer True
python main.py -d places_lt -m clip_vit_b16 adaptformer True test_only True tte True

# LIFT on iNaturalist 2018
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 test_only True tte True

# LIFT on  CIFAR-100-LT
python main.py -d cifar100_ir100 -m clip_vit_b16 adaptformer True
python main.py -d cifar100_ir50 -m clip_vit_b16 adaptformer True
python main.py -d cifar100_ir10 -m clip_vit_b16 adaptformer True
python main.py -d cifar100_ir100 -m clip_vit_b16 adaptformer True test_only True tte True
python main.py -d cifar100_ir50 -m clip_vit_b16 adaptformer True test_only True tte True
python main.py -d cifar100_ir10 -m clip_vit_b16 adaptformer True test_only True tte True

# LIFT with pre-trained ViT from ImageNet-21K
python main.py -d cifar100_ir100 -m in21k_vit_b16 adaptformer True
python main.py -d cifar100_ir50 -m in21k_vit_b16 adaptformer True
python main.py -d cifar100_ir10 -m in21k_vit_b16 adaptformer True
