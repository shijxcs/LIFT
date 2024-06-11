# Performance of CLIP on ImageNet-LT and Places-LT

python main.py -d imagenet_lt -m zsclip_vit_b16
python main.py -d imagenet_lt -m clip_vit_b16 lr 0.02  # best lr for classifier fine-tuning (searched from fig. 10)

python main.py -d places_lt -m zsclip_vit_b16
python main.py -d places_lt -m clip_vit_b16 lr 0.02  # best lr for classifier fine-tuning (searched from fig. 10)

python main.py -d inat2018_k -m zsclip_vit_b16
python main.py -d inat2018_p -m zsclip_vit_b16
python main.py -d inat2018_c -m zsclip_vit_b16
python main.py -d inat2018_o -m zsclip_vit_b16
python main.py -d inat2018_f -m zsclip_vit_b16
python main.py -d inat2018_g -m zsclip_vit_b16
python main.py -d inat2018_s -m zsclip_vit_b16
