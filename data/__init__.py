import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from data.coco_karpathy_dataset import coco_karpathy_train, coco_karpathy_caption_eval, coco_karpathy_retrieval_eval
from data.nocaps_dataset import nocaps_eval
from data.flickr30k_dataset import flickr30k_train, flickr30k_retrieval_eval
from data.vqa_dataset import vqa_dataset
from data.nlvr_dataset import nlvr_dataset
from data.pretrain_dataset import pretrain_dataset
from data.person_search_dataset import cuhk_pede_train, cuhk_pede_caption_eval, cuhk_pede_retrieval_eval,cuhk_pede_trainset_eval,mix_pede_train
from data.icfg_dataset import icfg_pede_train, icfg_pede_retrieval_eval
from data.rstp_dataset import rstp_pede_train, rstp_pede_retrieval_eval
from transform.randaugment import RandomAugment

def create_dataset(dataset, config, min_scale=0.5):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    if type(config['image_size']) == int:
        image_size = (config['image_size'], config['image_size'])
    elif type(config['image_size']) == list or type(config['image_size']) == tuple:
        image_size = (config['image_size'][0], config['image_size'][1])

    transform_train = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Pad(10),
        transforms.RandomCrop(image_size),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(scale=(0.02, 0.4), value=(0.48145466, 0.4578275, 0.40821073)),
    ])
    """transform_train = transforms.Compose([
        transforms.RandomResizedCrop(config['image_size'], scale=(min_scale, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 5, isPIL=True, augs=['Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])"""
    transform_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset=='pretrain':
        dataset = pretrain_dataset(config['train_file'], config['laion_path'], transform_train)              
        return dataset  
    
    elif dataset=='caption_coco':   
        train_dataset = coco_karpathy_train(transform_train, config['image_root'], config['ann_root'], prompt=config['prompt'])
        val_dataset = coco_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = coco_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')   
        return train_dataset, val_dataset, test_dataset
    
    elif dataset=='nocaps':   
        val_dataset = nocaps_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = nocaps_eval(transform_test, config['image_root'], config['ann_root'], 'test')   
        return val_dataset, test_dataset   
    
    elif dataset=='retrieval_coco':          
        train_dataset = coco_karpathy_train(transform_train, config['image_root'], config['ann_root'])
        val_dataset = coco_karpathy_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'val') 
        test_dataset = coco_karpathy_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'test')          
        return train_dataset, val_dataset, test_dataset    
    
    elif dataset=='retrieval_flickr':          
        train_dataset = flickr30k_train(transform_train, config['image_root'], config['ann_root'])
        val_dataset = flickr30k_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'val') 
        test_dataset = flickr30k_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'test')          
        return train_dataset, val_dataset, test_dataset     
    
    elif dataset=='vqa': 
        train_dataset = vqa_dataset(transform_train, config['ann_root'], config['vqa_root'], config['vg_root'], 
                                    train_files = config['train_files'], split='train') 
        test_dataset = vqa_dataset(transform_test, config['ann_root'], config['vqa_root'], config['vg_root'], split='test')
        return train_dataset, test_dataset
    
    elif dataset=='nlvr': 
        train_dataset = nlvr_dataset(transform_train, config['image_root'], config['ann_root'],'train')
        val_dataset = nlvr_dataset(transform_test, config['image_root'], config['ann_root'],'val')
        test_dataset = nlvr_dataset(transform_test, config['image_root'], config['ann_root'],'test')     
        return train_dataset, val_dataset, test_dataset

    elif dataset=='caption_cuhk':
        train_dataset = cuhk_pede_train(transform_train, config['image_root'])
        val_dataset = cuhk_pede_caption_eval(transform_test, config['image_root'], 'val')
        test_dataset = cuhk_pede_caption_eval(transform_test, config['image_root'], 'test')
        return train_dataset, val_dataset, test_dataset

    elif dataset=='retrieval_cuhk':
        train_dataset = cuhk_pede_train(transform_train, config['image_root'])
        val_dataset = cuhk_pede_retrieval_eval(transform_test, config['image_root'], 'val')
        test_dataset = cuhk_pede_retrieval_eval(transform_test, config['image_root'], 'test')
        return train_dataset, val_dataset, test_dataset

    elif dataset=='cuhk_trainset_eval':
        train_dataset = cuhk_pede_train(transform_train, config['image_root'])
        test_dataset = cuhk_pede_retrieval_eval(transform_test, config['image_root'], 'test')
        val_dataset = cuhk_pede_trainset_eval(transform_test,config['image_root'])
        return train_dataset,val_dataset,test_dataset

    elif dataset=='retrieval_mix':
        train_dataset = mix_pede_train(transform_train, config['image_root'])
        val_dataset = cuhk_pede_retrieval_eval(transform_test, config['image_root'], 'val')
        test_dataset = cuhk_pede_retrieval_eval(transform_test, config['image_root'], 'test')
        return train_dataset, val_dataset, test_dataset

    elif dataset=='retrieval_icfg':
        train_dataset = icfg_pede_train(transform_train, config['image_root'])
        val_dataset = icfg_pede_retrieval_eval(transform_test, config['image_root'], 'val')
        test_dataset = icfg_pede_retrieval_eval(transform_test, config['image_root'], 'test')
        return train_dataset, val_dataset, test_dataset

    elif dataset=='retrieval_rstp':
        train_dataset = rstp_pede_train(transform_train, config['image_root'])
        val_dataset = rstp_pede_retrieval_eval(transform_test, config['image_root'], 'val')
        test_dataset = rstp_pede_retrieval_eval(transform_test, config['image_root'], 'test')
        return train_dataset, val_dataset, test_dataset
    
    
def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    

