# LIFT with different classifier initialization methods

python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True init_head None tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True init_head class_mean tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True init_head linear_probe tte True

python main.py -d places_lt -m clip_vit_b16 adaptformer True init_head None tte True
python main.py -d places_lt -m clip_vit_b16 adaptformer True init_head class_mean tte True
python main.py -d places_lt -m clip_vit_b16 adaptformer True init_head linear_probe tte True
