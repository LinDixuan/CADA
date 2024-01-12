import os
import json

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption
import random

def split_query(data_list):
    ids = []
    query = []
    gallery = []
    for info in data_list:
        if info[0] in ids:
            gallery.append(info)
        else:
            query.append(info)
            ids.append(info[0])
    return query, gallery

def prepare_dataset(dataset):
    all_ann_root = '/home/share/jingke/dataset/CUHK-PEDES/caption_all.json'
    part_ann_root = '/home/dixuan/program/distill_transreid'
    datasets = {'c1':'cuhk01.json', 'c3':'cuhk03.json',
                'm':'market.json', 's':'sysu.json', 'v':'viper.json'}

    if dataset == 'all':
        with open(all_ann_root) as f:
            annotation = json.load(f)
        ids = []
        for info in annotation:
            ids.append(info['id'])
        num_all = len(ids)

        pids = [i for i in range(1,num_all+1)]
        num_train = 11003
        num_valid = 1000
        num_test  = 1000
        train_pids = random.sample(pids, num_train)
        train_pids.sort()
        rest_pids = list(set(pids)-set(train_pids))
        val_pids = random.sample(rest_pids, num_valid)
        val_pids.sort()
        test_pids = list(set(rest_pids)-set(val_pids))
        test_pids.sort()

        train_label = {id: label for label, id in enumerate(train_pids)}
        val_label = {id: label for label, id in enumerate(val_pids)}
        test_label = {id: label for label, id in enumerate(test_pids)}

        train_list = []
        val_list = []
        test_list = []

        for info in annotation:
            file_path = info['file_path']
            id = info['id']
            cap1 = info['captions'][0]
            cap2 = info['captions'][1]
            if info['id'] in train_pids:
                label = train_label[id]
                train_list.append((file_path,id, cap1, cap2, label))
            elif info['id'] in val_pids:
                label = val_label[id]
                val_list.append((file_path,id, cap1, cap2, label))
            else:
                label = test_label[id]
                test_list.append((file_path,id, cap1, cap2, label))

        return train_list, val_list, test_list



    else:
        train_set = []
        test_set = []
        for data in datasets:
            dir = os.path.join(part_ann_root,datasets[data])
            with open(dir) as f:
                annotation = json.load(f)
            train_set += annotation['train']
            test_set += annotation['test']

        ids_train = []
        ids_test = []
        for info in train_set:
            ids_train.append(info['id'])
        for info in test_set:
            ids_test.append(info['id'])

        train_label = {id: label for label, id in enumerate(set(ids_train))}
        test_label = {id: label for label, id in enumerate(set(ids_test))}

        train_list = []
        val_list = []
        test_list = []
        for info in train_set:
            id = info['id']
            file_path = info['filepath']
            cap1 = info['captions'][0]
            cap2 = info['captions'][1]
            label = train_label[id]
            train_list.append((file_path,id,cap1,cap2,label))

        for info in test_set:
            id = info['id']
            file_path = info['filepath']
            cap1 = info['captions'][0]
            cap2 = info['captions'][1]
            label = test_label[id]
            test_list.append((file_path, id, cap1, cap2, label))

        return train_list, val_list, test_list





class pede_dataset(Dataset):
    def __init__(self, transform, ann_set, max_words=50, prompt='', mode='train'):
        super(pede_dataset, self).__init__()
        assert mode in ['train','valid','test']
        self.image_root = '/home/share/jingke/dataset/CUHK-PEDES/imgs'

        self.train_list, self.val_list, self.test_list = prepare_dataset(ann_set)
        self.transform = transform
        self.max_words = max_words
        self.prompt = prompt
        self.mode = mode
        if mode == 'train':
            self.data_list = self.train_list
        elif mode == 'valid':
            self.data_list = self.val_list
        else:
            self.data_list = self.test_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        info = self.data_list[index]
        image_path = os.path.join(self.image_root,info[0])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        caption1 = self.prompt + pre_caption(info[2], self.max_words)
        caption2 = self.prompt + pre_caption(info[3], self.max_words)

        return image, caption1, caption2, info[-1]

