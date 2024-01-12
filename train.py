import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.cada import build_model
import utils
from utils import cosine_lr_schedule, cos_with_warmup_lr_scheduler
from data import create_dataset, create_sampler, create_loader

def train(model, data_loader, optimizer, epoch, device, config):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_cap', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)#
    print_freq = 100

    for i,(image, caption, img_pid) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device,non_blocking=True)
        img_pid = img_pid.to(device,non_blocking=True)

        loss_itc,loss_itm, loss_cap= model(image, caption, idx=img_pid)
        loss = loss_itc + loss_itm + loss_cap

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_cap=loss_cap.item())
        metric_logger.update(loss_itc=loss_itc.item())
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.8f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluation(model, data_loader, device, config, mode,itm=False):
    assert mode in ['i2t', 't2i', 'both']
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []

    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=73, return_tensors="pt").to(device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds,dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    text_ids[:, 0] = model.tokenizer.enc_token_id

    image_feats = []
    image_embeds = []
    for image, img_id in data_loader:
        image = image.to(device)
        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat[:,0,:])
        image_embed = F.normalize(image_embed,dim=-1)

        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = image_embeds @ text_embeds.t()  # img_len * text_len
    #score_matrix_i2t = torch.full((len(data_loader.dataset.image), len(texts)), -100.0).to(device)
    score_matrix_i2t = sims_matrix.clone()
    num_tasks = utils.get_world_size()
    rank = utils.get_rank()

    if mode in ['i2t', 'both']:
        step = sims_matrix.size(0) // num_tasks + 1  # step for each machine
        start = rank * step
        end = min(sims_matrix.size(0), start + step)

        for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 100, header)):
            topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

            encoder_output = image_feats[start + i].repeat(config['k_test'], 1, 1).to(device)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
            output = model.text_encoder(text_ids[topk_idx],
                                        attention_mask=text_atts[topk_idx],
                                        encoder_hidden_states=encoder_output,
                                        encoder_attention_mask=encoder_att,
                                        return_dict=True,
                                        )
            score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)

    if mode in ['t2i','both']:
        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)

        for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 500, header)):
            if itm==True:
                topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
                encoder_output = image_feats[topk_idx.cpu()].to(device)
                bs = encoder_output.shape[0]
                encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
                output = model.text_encoder(text_ids[start + i].repeat(config['k_test'], 1),
                                            attention_mask=text_atts[start + i].repeat(config['k_test'], 1),
                                            encoder_hidden_states=encoder_output,
                                            encoder_attention_mask=encoder_att,
                                            return_dict=True,
                                            )

                score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]

                # score = model.itm_head(output.last_hidden_state[:bs])[:, 1]
                score_matrix_t2i[start + i, topk_idx] =  score

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)
    score_matrix_t2i = score_matrix_t2i + sims_matrix

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt, img2pid, txt2pid,mode='both'):  # txt2img: corresponding id

    assert mode in ['i2t','t2i','both']
    img2pid = np.asarray(img2pid)
    txt2pid = np.asarray(txt2pid)
    if mode in ['i2t','both']:
        # Images->Text
        ranks = np.zeros(scores_i2t.shape[0])  # img_num
        for index, score in enumerate(scores_i2t):
            inds = np.argsort(score)[::-1]  # from highest to lowest
            # Score
            rank = 1e20
            for i in img2txt[index]:
                tmp = np.where(inds == i)[0][0]  # position
                if tmp < rank:
                    rank = tmp  # for each image find the highest rank from all corresponding text
            ranks[index] = rank

        # Compute metrics
        tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    if mode in ['t2i','both']:
        # Text->Images
        ranks = np.zeros(scores_t2i.shape[0])

        for index, score in enumerate(scores_t2i):
            inds = np.argsort(score)[::-1]
            rank = 1e20
            for i in txt2img[index]:
                tmp = np.where(inds == i)[0][0]  # position
                if tmp < rank:
                    rank = tmp  # for each text find the highest rank from all corresponding image
            ranks[index] = rank

        #=====rank=====
        indices = np.argsort(-scores_t2i,axis=1)
        pred_labels = img2pid[indices]
        matches = np.equal(txt2pid.reshape(-1,1),pred_labels)

        num_rel = matches.sum(1)  # q
        tmp_cmc = matches.cumsum(1)  # q * k


        tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
        tmp_cmc = np.stack(tmp_cmc, 1) * matches
        AP = tmp_cmc.sum(1) / num_rel  # q
        mAP = AP.mean() * 100
        #==============

        # Compute metrics
        ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    if mode == 'i2r':
        ir1 = 0
        ir5 = 0
        ir10 = 0

    if mode == 't2i':
        tr1 = 0
        tr5 = 0
        tr10 = 0

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = ir_mean+mAP

    eval_result = {
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                    'mAP':mAP,
                   'r_mean': r_mean}
    return eval_result

def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset(config['dataset'], config)
    num_classes = len(train_dataset.img_ids)
    print("train images:{}\t train ids:{}".format(len(train_dataset), num_classes))

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                          batch_size=[config['batch_size_train']] + [
                                                              config['batch_size_test']] * 2,
                                                          num_workers=[4, 4, 4],
                                                          is_trains=[True, False, False],
                                                          collate_fns=[None, None, None])

    #### Model ####
    print("Creating model")
    if args.load_head:
        mode = 'train'
    else:
        mode = 'eval'
    model = build_model(pretrained=config['pretrained'],mode=mode,num_classes=num_classes, image_size=config['image_size'], vit=config['vit'],
                             vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])

    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module


    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = config['init_lr']
        weight_decay = config['weight_decay']

        if 'mlm_head' in key:
            lr = config['init_lr'] * 5

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.AdamW(params=params, lr=config['init_lr'], weight_decay=config['weight_decay'])

    best = 0
    best_epoch = 0
    epoch_eval = args.epoch_eval

    print("Start training")
    start_time = time.time()

    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

            train_stats = train(model, train_loader, optimizer, epoch, device, config)


        score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, device, config, mode='t2i',itm=True)
        score_val_i2ta, score_val_itc = evaluation(model_without_ddp, val_loader, device, config, mode='t2i',itm=False)

        if epoch == config['max_epoch'] - 1 or args.evaluate:

            score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, device, config, mode='t2i',itm=True)
            score_test_itc_i2t, score_test_itc_t2i = evaluation(model_without_ddp, test_loader, device, config, mode='t2i',itm=False)

            test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img,
                                   test_loader.dataset.img2txt, test_loader.dataset.img2pid,
                                   test_loader.dataset.txt2pid, mode='t2i')
            test_itc_result = itm_eval(score_test_itc_i2t, score_test_itc_t2i, test_loader.dataset.txt2img,
                                       test_loader.dataset.img2txt, test_loader.dataset.img2pid,
                                       test_loader.dataset.txt2pid, mode='t2i')
            print(test_result)
            print(test_itc_result)


        if utils.is_main_process():

            val_result = itm_eval(score_val_i2t, score_val_t2i,val_loader.dataset.txt2img,val_loader.dataset.img2txt,
                                  val_loader.dataset.img2pid,val_loader.dataset.txt2pid,mode='t2i')
            val_itc_result = itm_eval(score_val_i2ta, score_val_itc,val_loader.dataset.txt2img,val_loader.dataset.img2txt,
                                      val_loader.dataset.img2pid,val_loader.dataset.txt2pid,mode='t2i')
            print("val",val_result)
            print("itc",val_itc_result)



            if test_result['r_mean'] > best:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                if not args.evaluate:
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                best = test_result['r_mean']
                best_epoch = epoch


            if args.evaluate:

                test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img,
                                       test_loader.dataset.img2txt, test_loader.dataset.img2pid,
                                       test_loader.dataset.txt2pid, mode='t2i')
                test_itc_result = itm_eval(score_test_itc_i2t, score_test_itc_t2i, test_loader.dataset.txt2img,
                                                       test_loader.dataset.img2txt, test_loader.dataset.img2pid,
                                                       test_loader.dataset.txt2pid, mode='t2i')
                print(test_result)
                print(test_itc_result)
                log_stats = {
                             **{f'test_{k}': v for k, v in test_result.items()},
                             **{f'test_itc_{k}': v for k, v in test_itc_result.items()},
                            }
                with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            else:
                if epoch == config['max_epoch'] - 1:
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                **{f'val_{k}': v for k, v in val_result.items()},
                                **{f'test_{k}': v for k, v in test_result.items()},
                                 **{f'test_itc_{k}': v for k, v in test_itc_result.items()},
                                'epoch': epoch,
                                'best_epoch': best_epoch,
                                }
                else:
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                 **{f'val_{k}': v for k, v in val_result.items()},
                                 'epoch': epoch,
                                 'best_epoch': best_epoch,
                                 }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        if args.evaluate:
            break

        dist.barrier()
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/retrieval_cuhk.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_Person')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--max_epoch', default=0, type=int)
    parser.add_argument('--batch_size_train', default=0, type=int)
    parser.add_argument('--batch_size_test', default=0, type=int)
    parser.add_argument('--init_lr', default=0.0001, type=float)
    parser.add_argument('--epoch_eval',default=1,type=int)
    parser.add_argument('--load_head',action='store_true')
    parser.add_argument('--k_test',default=32,type=int)
    parser.add_argument('--pretrained',default='./checkpoint/model_base_retrieval_coco.pth')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    if args.max_epoch > 0:
        config['max_epoch'] = args.max_epoch
    if args.batch_size_train > 0:
        config['batch_size_train'] = args.batch_size_train
    if args.batch_size_test > 0:
        config['batch_size_test'] = args.batch_size_test

    config['init_lr'] = args.init_lr
    config['queue_size'] = args.queue_size
    if args.evaluate:
        config['pretrained'] = args.pretrained
    config['k_test'] = args.k_test

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)