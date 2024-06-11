# Ablation study

python main.py -d imagenet_lt -m clip_vit_b16 loss_type CE init_head None
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True loss_type CE init_head None
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True loss_type LA init_head None
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True loss_type LA init_head text_feat
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True loss_type LA init_head text_feat tte True

python main.py -d places_lt -m clip_vit_b16 loss_type CE init_head None
python main.py -d places_lt -m clip_vit_b16 adaptformer True loss_type CE init_head None
python main.py -d places_lt -m clip_vit_b16 adaptformer True loss_type LA init_head None
python main.py -d places_lt -m clip_vit_b16 adaptformer True loss_type LA init_head text_feat
python main.py -d places_lt -m clip_vit_b16 adaptformer True loss_type LA init_head text_feat tte True
