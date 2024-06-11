# LIFT with different expand sizes (after running the main results and getting the checkpoints)

python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True test_only True tte True expand 4
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True test_only True tte True expand 8
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True test_only True tte True expand 12
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True test_only True tte True expand 16
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True test_only True tte True expand 20
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True test_only True tte True expand 24
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True test_only True tte True expand 28
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True test_only True tte True expand 32
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True test_only True tte True expand 36
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True test_only True tte True expand 40
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True test_only True tte True expand 44
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True test_only True tte True expand 48

python main.py -d places_lt -m clip_vit_b16 adaptformer True test_only True tte True expand 4
python main.py -d places_lt -m clip_vit_b16 adaptformer True test_only True tte True expand 8
python main.py -d places_lt -m clip_vit_b16 adaptformer True test_only True tte True expand 12
python main.py -d places_lt -m clip_vit_b16 adaptformer True test_only True tte True expand 16
python main.py -d places_lt -m clip_vit_b16 adaptformer True test_only True tte True expand 20
python main.py -d places_lt -m clip_vit_b16 adaptformer True test_only True tte True expand 24
python main.py -d places_lt -m clip_vit_b16 adaptformer True test_only True tte True expand 28
python main.py -d places_lt -m clip_vit_b16 adaptformer True test_only True tte True expand 32
python main.py -d places_lt -m clip_vit_b16 adaptformer True test_only True tte True expand 36
python main.py -d places_lt -m clip_vit_b16 adaptformer True test_only True tte True expand 40
python main.py -d places_lt -m clip_vit_b16 adaptformer True test_only True tte True expand 44
python main.py -d places_lt -m clip_vit_b16 adaptformer True test_only True tte True expand 48

python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 test_only True tte True expand 4
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 test_only True tte True expand 8
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 test_only True tte True expand 12
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 test_only True tte True expand 16
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 test_only True tte True expand 20
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 test_only True tte True expand 24
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 test_only True tte True expand 28
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 test_only True tte True expand 32
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 test_only True tte True expand 36
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 test_only True tte True expand 40
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 test_only True tte True expand 44
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 test_only True tte True expand 48
