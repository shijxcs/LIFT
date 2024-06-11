# LIFT with different training epochs

# without TTE
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True num_epochs 5
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True num_epochs 10
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True num_epochs 20
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True num_epochs 30

# with TTE (after running the above commands and getting the checkpoints)
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True num_epochs 5 test_only True tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True num_epochs 10 test_only True tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True num_epochs 20 test_only True tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True num_epochs 30 test_only True tte True

# without TTE
python main.py -d places_lt -m clip_vit_b16 adaptformer True num_epochs 5
python main.py -d places_lt -m clip_vit_b16 adaptformer True num_epochs 10
python main.py -d places_lt -m clip_vit_b16 adaptformer True num_epochs 20
python main.py -d places_lt -m clip_vit_b16 adaptformer True num_epochs 30

# with TTE (after running the above commands and getting the checkpoints)
python main.py -d places_lt -m clip_vit_b16 adaptformer True num_epochs 5 test_only True tte True
python main.py -d places_lt -m clip_vit_b16 adaptformer True num_epochs 10 test_only True tte True
python main.py -d places_lt -m clip_vit_b16 adaptformer True num_epochs 20 test_only True tte True
python main.py -d places_lt -m clip_vit_b16 adaptformer True num_epochs 30 test_only True tte True

# without TTE
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 5
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 10
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 30

# with TTE (after running the above commands and getting the checkpoints)
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 5 test_only True tte True
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 10 test_only True tte True
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 test_only True tte True
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 30 test_only True tte True
