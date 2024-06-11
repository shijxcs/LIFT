# LIFT with different PEFT methods

python main.py -d imagenet_lt -m zsclip_vit_b16 tte True
python main.py -d imagenet_lt -m clip_vit_b16 tte True
python main.py -d imagenet_lt -m clip_vit_b16 full_tuning True lr 0.0005 tte True  # best lr for full fine-tuning (searched from fig. 10)
python main.py -d imagenet_lt -m clip_vit_b16 mask True mask_ratio 0.01 tte True  # best proportion for arbitrary fine-tuning (searched from fig. 4)
python main.py -d imagenet_lt -m clip_vit_b16 bias_tuning True
python main.py -d imagenet_lt -m clip_vit_b16 vpt_shallow True
python main.py -d imagenet_lt -m clip_vit_b16 vpt_deep True
python main.py -d imagenet_lt -m clip_vit_b16 adapter True
python main.py -d imagenet_lt -m clip_vit_b16 lora True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True

python main.py -d places_lt -m zsclip_vit_b16 tte True
python main.py -d places_lt -m clip_vit_b16 tte True
python main.py -d places_lt -m clip_vit_b16 full_tuning True lr 0.001 tte True  # best lr for full fine-tuning (searched from fig. 10)
python main.py -d places_lt -m clip_vit_b16 mask True mask_ratio 0.005 tte True  # best proportion for arbitrary fine-tuning (searched from fig. 4)
python main.py -d places_lt -m clip_vit_b16 bias_tuning True
python main.py -d places_lt -m clip_vit_b16 vpt_shallow True
python main.py -d places_lt -m clip_vit_b16 vpt_deep True
python main.py -d places_lt -m clip_vit_b16 adapter True
python main.py -d places_lt -m clip_vit_b16 lora True
python main.py -d places_lt -m clip_vit_b16 adaptformer True