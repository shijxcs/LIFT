# LIFT with different losses

python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True loss_type CE tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True loss_type Focal tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True loss_type LDAM tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True loss_type CB tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True loss_type GRW tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True loss_type LADE tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True loss_type LA tte True

python main.py -d places_lt -m clip_vit_b16 adaptformer True loss_type CE tte True
python main.py -d places_lt -m clip_vit_b16 adaptformer True loss_type Focal tte True
python main.py -d places_lt -m clip_vit_b16 adaptformer True loss_type LDAM tte True
python main.py -d places_lt -m clip_vit_b16 adaptformer True loss_type CB tte True
python main.py -d places_lt -m clip_vit_b16 adaptformer True loss_type GRW tte True
python main.py -d places_lt -m clip_vit_b16 adaptformer True loss_type LADE tte True
python main.py -d places_lt -m clip_vit_b16 adaptformer True loss_type LA tte True

python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 loss_type CE tte True
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 loss_type Focal tte True
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 loss_type LDAM tte True
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 loss_type CB tte True
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 loss_type GRW tte True
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 loss_type LADE tte True
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 loss_type LA tte True
