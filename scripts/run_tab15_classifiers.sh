# LIFT with different classifiers

python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True classifier LinearClassifier tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True classifier L2NormedClassifier tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True classifier CosineClassifier scale 15 tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True classifier CosineClassifier scale 20 tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True classifier CosineClassifier scale 25 tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True classifier CosineClassifier scale 30 tte True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True classifier CosineClassifier scale 35 tte True

python main.py -d places_lt -m clip_vit_b16 adaptformer True classifier LinearClassifier tte True
python main.py -d places_lt -m clip_vit_b16 adaptformer True classifier L2NormedClassifier tte True
python main.py -d places_lt -m clip_vit_b16 adaptformer True classifier CosineClassifier scale 15 tte True
python main.py -d places_lt -m clip_vit_b16 adaptformer True classifier CosineClassifier scale 20 tte True
python main.py -d places_lt -m clip_vit_b16 adaptformer True classifier CosineClassifier scale 25 tte True
python main.py -d places_lt -m clip_vit_b16 adaptformer True classifier CosineClassifier scale 30 tte True
python main.py -d places_lt -m clip_vit_b16 adaptformer True classifier CosineClassifier scale 35 tte True

python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 classifier LinearClassifier tte True
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 classifier L2NormedClassifier tte True
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 classifier CosineClassifier scale 15 tte True
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 classifier CosineClassifier scale 20 tte True
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 classifier CosineClassifier scale 25 tte True
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 classifier CosineClassifier scale 30 tte True
python main.py -d inat2018 -m clip_vit_b16 adaptformer True num_epochs 20 classifier CosineClassifier scale 35 tte True
