# LIFT with different initialization prompts

python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True init_head None tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True prompt classname tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True prompt default tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True prompt ensemble tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True prompt descriptor tte True