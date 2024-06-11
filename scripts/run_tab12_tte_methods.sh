# LIFT with different tte methods (after running the main results and getting the checkpoints)

python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True test_only True tte True tte_mode fivecrop
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True test_only True tte True tte_mode tencrop
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True test_only True tte True tte_mode randaug randaug_times 5
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True test_only True tte True tte_mode randaug randaug_times 10
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True test_only True tte True tte_mode randaug randaug_times 15
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True test_only True tte True tte_mode randaug randaug_times 20