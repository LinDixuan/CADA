import os
import json

from torch.utils.data import Dataset

from PIL import Image

from data.utils import pre_caption


def split_CUHK_PEDE():
    root_dir = '/workspace/MHA/datasets/CUHK-PEDES'
    raw_dir = 'reid_raw.json'
    all_dir = 'caption_all.json'

    with open(os.path.join(root_dir,raw_dir),'r') as f:
        cap_list = json.load(f)

    train_list = []
    val_list = []
    test_list = []
    #counting = {i:[0,0] for i in range(1, 13003 + 1)}

    for info in cap_list:
        if info['split'] == 'train':
            info1 = info.copy()
            info2 = info.copy()
            info1['captions'] = info['captions'][0]
            info2['captions'] = info['captions'][1]
            train_list.append(info1)
            train_list.append(info2)
        elif info['split'] == 'test':
            test_list.append(info)
        else:
            val_list.append(info)

        #counting[info['id']][0] += 1
        #counting[info['id']][1] += int(len(info['captions']))


    """stat_img = {i:0 for i in range(30)}
    stat_cap = {i:0 for i in range(60)}
    for i in range(1, 13003 + 1):

        stat_img[counting[i][0]] += 1
        stat_cap[counting[i][1]] += 1

    sum_img = 0
    sum_cap = 0
    for i in range(30):
        sum_img += stat_img[i] * i
    for i in range(60):
        sum_cap += stat_cap[i] * i

    print(sum_img)
    print(sum_cap)"""

    return train_list, val_list, test_list


def split_ICFG_PEDE():
    root_dir = '/workspace/MHA/datasets/ICFG-PEDES'
    raw_dir = 'split.json'

    with open(os.path.join(root_dir, raw_dir), 'r') as f:
        cap_list = json.load(f)

    train_list = cap_list['train'].copy()
    val_list = cap_list['test'].copy()
    test_list = cap_list['test'].copy()

    return train_list, val_list, test_list

class mix_pede_train(Dataset):
    def __init__(self, transform, image_root, max_words=72, prompt=''):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''
        cuhk_path = '/workspace/MHA/datasets/CUHK-PEDES/imgs'
        icfg_path = '/workspace/MHA/datasets/ICFG-PEDES/'
        train_list, _, _ = split_CUHK_PEDE()
        self.annotation = train_list
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

        self.img_ids = {}
        n = 0
        for i,ann in enumerate(self.annotation):
            img_id = ann['id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
            self.annotation[i]['file_path'] = os.path.join(cuhk_path,self.annotation[i]['file_path'])

        train_list, _, _ = split_ICFG_PEDE()
        for i, ann in enumerate(train_list):
            train_list[i]['id'] = train_list[i]['id'] + 20000
            img_id = train_list[i]['id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
            train_list[i]['file_path'] = os.path.join(icfg_path, train_list[i]['file_path'])
            train_list[i]['captions'] = ann['captions'][0]
        self.annotation = self.annotation + train_list


    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image = Image.open(ann['file_path'])
        image = self.transform(image)
        captions = ann['captions']
        captions = self.prompt+pre_caption(captions, self.max_words)
        return image, captions, self.img_ids[ann['id']]

class cuhk_pede_train(Dataset):
    def __init__(self, transform, image_root, max_words=72, prompt=''):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''
        train_list, _, _ = split_CUHK_PEDE()
        self.annotation = train_list
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann['id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = self.image_root
        image = Image.open(os.path.join(image_path,ann['file_path']))
        image = self.transform(image)
        captions = ann['captions']
        captions = self.prompt+pre_caption(captions, self.max_words)
        return image, captions, self.img_ids[ann['id']]


class cuhk_pede_caption_eval(Dataset):
    def __init__(self, transform, image_root, split):
        _, val_list, test_list = split_CUHK_PEDE()

        assert split in ['val','test']

        if split == 'val':
            self.annotation = val_list
        elif split == 'test':
            self.annotation = test_list
        self.transform = transform
        self.image_root = image_root

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.image_root,ann['file_path'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        img_id = ann['id']

        return image, img_id


class cuhk_pede_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, split, max_words=72):
        train_list, val_list, test_list = split_CUHK_PEDE()
        assert split in ['val', 'test']

        if split == 'val':
            self.annotation = val_list
        elif split == 'test':
            self.annotation = test_list
        self.transform = transform
        self.image_root = image_root

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        self.txt2pid  = []
        self.img2pid  = []

        person = {}
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['file_path'])

            caps = ann['captions']

            self.text.append(pre_caption(caps[0],max_words))
            self.text.append(pre_caption(caps[1],max_words))
            pid = ann['id']
            self.img2pid.append(pid)
            self.txt2pid.append(pid)
            self.txt2pid.append(pid)
            if pid not in person.keys():
                person[pid] = {'image':[img_id],'text':[txt_id,txt_id+1]}
            else:
                person[pid]['image'].append(img_id)
                person[pid]['text'].append(txt_id)
                person[pid]['text'].append(txt_id+1)
            txt_id = txt_id + 2

        for pid in person.keys():
            for img_id in person[pid]['image']:
                self.img2txt[img_id] = person[pid]['text']
                assert self.img2pid[img_id] == pid
            for txt_id in person[pid]['text']:
                self.txt2img[txt_id] = person[pid]['image']
                assert self.txt2pid[txt_id] == pid

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.annotation[index]['file_path'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index


# eval for train_data
class cuhk_pede_trainset_eval(Dataset):
    def __init__(self, transform, image_root, max_words=50):
        train_list, val_list, test_list = split_CUHK_PEDE()
        self.annotation = train_list
        self.transform = transform
        self.image_root = image_root

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        self.txt2pid  = {}
        self.img2pid  = {}

        person = {}
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['file_path'])

            self.text.append(pre_caption(ann['captions'],max_words))
            pid = ann['id']
            self.img2pid[img_id] = pid
            self.txt2pid[txt_id] = pid
            if pid not in person.keys():
                person[pid] = {'image':[img_id],'text':[txt_id]}
            else:
                person[pid]['image'].append(img_id)
                person[pid]['text'].append(txt_id)
            txt_id = txt_id + 1

        for pid in person.keys():
            for img_id in person[pid]['image']:
                self.img2txt[img_id] = person[pid]['text']
            for txt_id in person[pid]['text']:
                self.txt2img[txt_id] = person[pid]['image']

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.annotation[index]['file_path'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index


if __name__ == '__main__':
    pass