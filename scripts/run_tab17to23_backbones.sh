# LIFT on ImageNet-LT with ViT-L/14
python main.py -d imagenet_lt -m clip_vit_l14 adaptformer True
python main.py -d imagenet_lt -m clip_vit_l14 adaptformer True test_only True tte True

# LIFT on Places-LT with ViT-L/14
python main.py -d places_lt -m clip_vit_l14 adaptformer True
python main.py -d places_lt -m clip_vit_l14 adaptformer True test_only True tte True

# LIFT on iNaturalist 2018 with ViT-L/14
python main.py -d inat2018 -m clip_vit_l14 adaptformer True num_epochs 20
python main.py -d inat2018 -m clip_vit_l14 adaptformer True num_epochs 20 test_only True tte True

# LIFT on iNaturalist 2018 with ViT-L/14@336px
python main.py -d inat2018 -m clip_vit_l14@336px adaptformer True num_epochs 20
python main.py -d inat2018 -m clip_vit_l14@336px adaptformer True num_epochs 20 test_only True tte True

# LIFT on ImageNet-LT with ResNet-50
python main.py -d imagenet_lt -m clip_rn50 ssf_attn True
python main.py -d imagenet_lt -m clip_rn50 bias_tuning True
python main.py -d imagenet_lt -m clip_rn50 bias_tuning True ssf_attn True

# LIFT on Places-LT with ResNet-50
python main.py -d places_lt -m clip_rn50 ssf_attn True
python main.py -d places_lt -m clip_rn50 bias_tuning True
python main.py -d places_lt -m clip_rn50 bias_tuning True ssf_attn True

# LIFT on ImageNet-LT with pre-trained ViT from ImageNet-21K
python main.py -d imagenet_lt -m in21k_vit_b16 adaptformer True tte True

# LIFT on Places-LT with pre-trained ViT from ImageNet-21K
python main.py -d places_lt -m in21k_vit_b16 adaptformer True tte True

# LIFT on iNaturalist 2018 pre-trained ViT from ImageNet-21K
python main.py -d inat2018 -m in21k_vit_b16 adaptformer True num_epochs 20 tte True
