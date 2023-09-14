import os
import json
from .lt_data import LT_Dataset


class iNaturalist2018(LT_Dataset):
    category_method = "name"
    categories_json = "./datasets/iNaturalist18/categories.json"
    train_txt = "./datasets/iNaturalist18/iNaturalist18_train.txt"
    test_txt = "./datasets/iNaturalist18/iNaturalist18_val.txt"

    def __init__(self, root, train=True, transform=None):
        super().__init__(root, train, transform)

        id2cname, cname2lab = self.read_category_info()

        self.names = []
        self.labels = []
        with open(self.txt) as f:
            for line in f:
                _name = id2cname[int(line.split()[1])]
                self.names.append(_name)
                self.labels.append(cname2lab[_name])

        self.classnames = self.get_classnames()
        self.cls_num_list = self.get_cls_num_list()
        self.num_classes = len(self.cls_num_list)

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        name = self.names[index]
        return image, label, name

    @classmethod
    def read_category_info(self):
        with open(self.categories_json, "rb") as file:
            category_info = json.load(file)
        
        id2cname = {}
        for id, info in enumerate(category_info):
            cname = info[self.category_method]
            id2cname[id] = cname

        cnames_unique = sorted(set(id2cname.values()))
        cname2lab = {c: i for i, c in enumerate(cnames_unique)}
        return id2cname, cname2lab

    def get_classnames(self):
        container = set()
        for label, name in zip(self.labels, self.names):
            container.add((label, name))
        mapping = {label: classname for label, classname in container}
        classnames = [mapping[label] for label in sorted(mapping.keys())]
        return classnames


class iNaturalist2018_Kingdom(iNaturalist2018):
    category_method = "kingdom"

class iNaturalist2018_Phylum(iNaturalist2018):
    category_method = "phylum"

class iNaturalist2018_Class(iNaturalist2018):
    category_method = "class"

class iNaturalist2018_Order(iNaturalist2018):
    category_method = "order"

class iNaturalist2018_Family(iNaturalist2018):
    category_method = "family"

class iNaturalist2018_Genus(iNaturalist2018):
    category_method = "genus"

class iNaturalist2018_Species(iNaturalist2018):
    category_method = "name"
