# LIFT with different learnable parameters by changing the bottleneck dimension

python main.py -d imagenet_lt -m clip_vit_b16 tte True 
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True adapter_dim 1 tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True adapter_dim 2 tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True adapter_dim 4 tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True adapter_dim 8 tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True adapter_dim 16 tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True adapter_dim 32 tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True adapter_dim 64 tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True adapter_dim 128 tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True adapter_dim 256 tte True

python main.py -d places_lt -m clip_vit_b16 tte True
python main.py -d places_lt -m clip_vit_b16 adaptformer True adapter_dim 1 tte True
python main.py -d places_lt -m clip_vit_b16 adaptformer True adapter_dim 2 tte True
python main.py -d places_lt -m clip_vit_b16 adaptformer True adapter_dim 4 tte True
python main.py -d places_lt -m clip_vit_b16 adaptformer True adapter_dim 8 tte True
python main.py -d places_lt -m clip_vit_b16 adaptformer True adapter_dim 16 tte True
python main.py -d places_lt -m clip_vit_b16 adaptformer True adapter_dim 32 tte True
python main.py -d places_lt -m clip_vit_b16 adaptformer True adapter_dim 64 tte True
python main.py -d places_lt -m clip_vit_b16 adaptformer True adapter_dim 128 tte True
python main.py -d places_lt -m clip_vit_b16 adaptformer True adapter_dim 256 tte True

python main.py -d inat2018 -m clip_vit_b16 num_epochs 20 tte True
python main.py -d inat2018 -m clip_vit_b16 num_epochs 20 adaptformer True adapter_dim 64 tte True
python main.py -d inat2018 -m clip_vit_b16 num_epochs 20 adaptformer True adapter_dim 128 tte True
python main.py -d inat2018 -m clip_vit_b16 num_epochs 20 adaptformer True adapter_dim 192 tte True
python main.py -d inat2018 -m clip_vit_b16 num_epochs 20 adaptformer True adapter_dim 256 tte True
python main.py -d inat2018 -m clip_vit_b16 num_epochs 20 adaptformer True adapter_dim 320 tte True
python main.py -d inat2018 -m clip_vit_b16 num_epochs 20 adaptformer True adapter_dim 384 tte True
python main.py -d inat2018 -m clip_vit_b16 num_epochs 20 adaptformer True adapter_dim 448 tte True
python main.py -d inat2018 -m clip_vit_b16 num_epochs 20 adaptformer True adapter_dim 512 tte True
