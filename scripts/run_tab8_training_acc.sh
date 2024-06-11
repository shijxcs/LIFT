# Training accuracy of different fine-tuning methods (after running the partial fine-tuning and the main results and getting the checkpoints)

python main.py -d imagenet_lt -m clip_vit_b16 full_tuning True partial 0 lr 0.02 test_train True  # best lr for classifier fine-tuning (searched from fig. 10)
python main.py -d imagenet_lt -m clip_vit_b16 full_tuning True partial 0 lr 0.01 test_train True  # lr equal to LIFT
python main.py -d imagenet_lt -m clip_vit_b16 full_tuning True partial 12 lr 0.0005 test_train True  # best lr for full fine-tuning (searched from fig. 10)
python main.py -d imagenet_lt -m clip_vit_b16 full_tuning True partial 12 lr 0.01 test_train True  # lr equal to LIFT
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True test_train True  # LIFT

python main.py -d places_lt -m clip_vit_b16 full_tuning True partial 0 lr 0.02 test_train True  # best lr for classifier fine-tuning (searched from fig. 10)
python main.py -d places_lt -m clip_vit_b16 full_tuning True partial 0 lr 0.01 test_train True  # lr equal to LIFT
python main.py -d places_lt -m clip_vit_b16 full_tuning True partial 12 lr 0.001 test_train True  # best lr for full fine-tuning (searched from fig. 10)
python main.py -d places_lt -m clip_vit_b16 full_tuning True partial 12 lr 0.01 test_train True  # lr equal to LIFT
python main.py -d places_lt -m clip_vit_b16 adaptformer True test_train True  # LIFT

python main.py -d inat2018 -m clip_vit_b16 full_tuning True num_epochs 20 partial 0 lr 0.01 test_train True  # best lr for classifier fine-tuning (searched from fig. 10)
python main.py -d inat2018 -m clip_vit_b16 full_tuning True num_epochs 20 partial 0 lr 0.01 test_train True  # lr equal to LIFT
python main.py -d inat2018 -m clip_vit_b16 full_tuning True num_epochs 20 partial 12 lr 0.005 test_train True  # best lr for full fine-tuning (searched from fig. 10)
python main.py -d inat2018 -m clip_vit_b16 full_tuning True num_epochs 20 partial 12 lr 0.01 test_train True  # lr equal to LIFT
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 test_train True  # LIFT
