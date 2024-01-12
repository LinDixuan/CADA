import warnings
warnings.filterwarnings("ignore")

from models.vit import VisionTransformer, interpolate_pos_embed
from models.med import BertConfig, BertModel, BertLMHeadModel, BertFuseModel
from models.blip import create_vit, init_tokenizer

from transformers import BertTokenizer
import torch
from torch import nn
import torch.nn.functional as F
import os


class BLIP_CLIP(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=224,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 num_classes=11003
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.num_classes = num_classes
        self.logit_scale = torch.ones([]) * (1 / 0.02)
        self.temp = torch.ones([]) * (1 / 0.07)
        self.embed_dim = 256
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)
        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, self.embed_dim)
        self.text_proj = nn.Linear(text_width, self.embed_dim)
        self.text_decoder = BertFuseModel(config=med_config)
        self.itm_head = nn.Linear(text_width, 2)

    def forward(self, image, caption, idx):

        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=60,
                              return_tensors="pt").to(image.device)
        text_output = self.text_decoder(text.input_ids, attention_mask=text.attention_mask,
                                        return_dict=True, mode='text',task='encoder')
        text_embeds = text_output.last_hidden_state
        image_embeds = self.visual_encoder(image)

        text_feats = self.text_proj(text_embeds[:,0,:])
        image_feats = self.vision_proj(image_embeds[:,0,:])

        itc_loss = self.compute_itc_loss(image_feats,text_feats,idx,intra=False,world=True)
        itm_loss = self.compute_itm_loss(image_embeds,text,image_feats,text_feats,idx)
        #cap_loss = self.compute_mlmlm_loss(text,image_embeds,task='mlm')

        return itc_loss, itm_loss

    def compute_sdm_loss(self, image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
        """
        Similarity Distribution Matching
        """
        batch_size = image_fetures.shape[0]
        pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
        pid_dist = pid - pid.t()
        labels = (pid_dist == 0).float()

        if image_id != None:
            # print("Mix PID and ImageID to create soft label.")
            image_id = image_id.reshape((-1, 1))
            image_id_dist = image_id - image_id.t()
            image_id_mask = (image_id_dist == 0).float()
            labels = (labels - image_id_mask) * factor + image_id_mask
            # labels = (labels + image_id_mask) / 2

        image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
        text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

        t2i_cosine_theta = text_norm @ image_norm.t()
        i2t_cosine_theta = t2i_cosine_theta.t()

        text_proj_image = logit_scale * t2i_cosine_theta
        image_proj_text = logit_scale * i2t_cosine_theta

        # normalize the true matching distribution
        labels_distribute = labels / labels.sum(dim=1)

        i2t_pred = F.softmax(image_proj_text, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
        t2i_pred = F.softmax(text_proj_image, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

        loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

        return loss

    def compute_itc_loss(self,image_feats, text_feats, labels, intra=False, world=False):
        if world:
            image_feats = all_gather_with_grad(image_feats)
            text_feats = all_gather_with_grad(text_feats)
            labels = concat_all_gather(labels)
        bs = image_feats.shape[0]

        """labels = torch.arange(start=0, end=bs, dtype=torch.int64)
        labels = labels.to(image_feats.device)"""

        labels = labels.view(-1,bs)
        labels = torch.eq(labels, labels.t()).float()

        # normalized features
        image_norm = image_feats / image_feats.norm(dim=-1, keepdim=True)
        text_norm = text_feats / text_feats.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_image = self.logit_scale * image_norm @ text_norm.t()
        logits_per_text = logits_per_image.t()

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2

        if intra:
            intra_image = logit_scale * image_norm @ image_norm.t()
            intra_text = logit_scale * text_norm @ text_norm.t()
            loss_ii = F.cross_entropy(intra_image,labels)
            loss_tt = F.cross_entropy(intra_text,labels)
            loss = (loss + loss_ii + loss_tt) / 2

        return loss


    def compute_itm_loss(self,image_embeds, text, image_feat, text_feat, idx):

        idx = idx.view(-1,1)
        idxs = concat_all_gather(idx)
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id  # change [CLS] to special token
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        # forward the positve image-text pair
        bs = image_embeds.shape[0]
        output_pos = self.text_decoder(encoder_input_ids,  # fusion of image and text from one pair
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       task='itm'
                                       )
        # compute sample similarity
        with torch.no_grad():
            mask = torch.eq(idx, idxs.t())

            image_feat_world = concat_all_gather(image_feat)
            text_feat_world = concat_all_gather(text_feat)

            sim_i2t = image_feat @ text_feat_world.t() * self.temp
            sim_t2i = text_feat @ image_feat_world.t() * self.temp

            weights_i2t = F.softmax(sim_i2t, dim=1) #+ 1e-4
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i = F.softmax(sim_t2i, dim=1) #+ 1e-4# for selecting hard negative
            weights_t2i.masked_fill_(mask, 0)  # exclude positive sample

        image_embeds_world = all_gather_with_grad(image_embeds)

        # select a negative image (from all ranks) for each text
        image_embeds_neg = []
        for b in range(bs):
            #neg_idx = torch.multinomial(weights_t2i[b], 1).item()  # 取样 取相似度最高的那一个
            neg_idx = torch.argmax(weights_t2i[b])
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text (from all ranks) for each image
        input_ids_world = concat_all_gather(encoder_input_ids)
        att_mask_world = concat_all_gather(text.attention_mask)

        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            #neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            neg_idx = torch.argmax(weights_i2t[b])
            text_ids_neg.append(input_ids_world[neg_idx])
            text_atts_neg.append(att_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)  # text_input       text_hard_neg
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)  # ||                ||
                                                                            # img_hard_neg    img_imput
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        output_neg = self.text_decoder(text_ids_all,
                                       attention_mask=text_atts_all,
                                       encoder_hidden_states=image_embeds_all,
                                       encoder_attention_mask=image_atts_all,
                                       return_dict=True,
                                       task='itm'
                                       )

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]],
                                  dim=0)  # multi-feat
        vl_output = self.itm_head(vl_embeddings)  # (bs*3) * 2

        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(image_embeds.device)  # (bs * 3)
        loss = F.cross_entropy(vl_output, itm_labels)
        return loss

    def compute_mlmlm_loss(self,text,image_embeds,task='mlm'):
        assert task in ['lm','mlm','mlmlm']
        loss = 0.
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        if task in ['lm','mlmlm']:
            lm_input_ids = text.input_ids.clone()
            lm_input_ids[:, 0] = self.tokenizer.bos_token_id
            lm_labels = lm_input_ids.masked_fill(lm_input_ids == self.tokenizer.pad_token_id, -100)

            decoder_output_lm = self.text_decoder(lm_input_ids,
                                                  attention_mask=text.attention_mask,
                                                  encoder_hidden_states=image_embeds,
                                                  encoder_attention_mask=image_atts,
                                                  labels=lm_labels,
                                                  return_dict=True,
                                                  output_hidden_states=True,
                                                  task='lm'
                                                  )
            loss += decoder_output_lm.loss

        if task in ['mlm', 'mlmlm']:
            mlm_input_ids = text.input_ids.clone()
            mlm_labels = mlm_input_ids.clone()
            mlm_probability = 0.3

            probability_matrix = torch.full(mlm_labels.shape, mlm_probability)
            mlm_input_ids, mlm_labels = self.mask(mlm_input_ids, self.text_decoder.config.vocab_size, image_embeds.device,
                                              targets=mlm_labels, probability_matrix=probability_matrix)
            decoder_output_mlm = self.text_decoder(mlm_input_ids,
                                                   attention_mask=text.attention_mask,
                                                   encoder_hidden_states=image_embeds,
                                                   encoder_attention_mask=image_atts,
                                                   labels=mlm_labels,
                                                   return_dict=True,
                                                   output_hidden_states=True,
                                                   task='mlm'
                                                   )
            loss += decoder_output_mlm.loss

        return loss

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        filter = ['a', 'an', 'the', 'is', 'are', 'and', 'with', 'over', 'on', 'in', 'have', 'has', 'wearing',
                  'while', 'this', 's', 'his', 'her', 'of', 'also', 'holding']
        filter_ids = self.tokenizer.convert_tokens_to_ids(filter)

        for fid in filter_ids:
            masked_indices[input_ids == fid] = False
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids



def blip_clip(pretrained='',**kwargs):
    model = BLIP_CLIP(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        print('missing keys:',msg.missing_keys)
    return model

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output      # rank_num * tensor_size

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)


def load_checkpoint(model, url_or_filename,mode='train'):
    if os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')

    state_dict = checkpoint['model']

    if 'ptr_queue' in state_dict.keys():
        del state_dict['ptr_queue']
    if 'image_queue' in state_dict.keys():
        del state_dict['image_queue']
    if 'text_queue' in state_dict.keys():
        del state_dict['text_queue']
    if 'idx_queue' in state_dict.keys():
        del state_dict['idx_queue']
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],
                                                                   model.visual_encoder)
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)

    for param_de, param_en in zip(model.text_decoder.bert.parameters(),model.text_encoder.parameters()):
        param_de.data.copy_(param_en.data)
        param_de.requires_grad = True
        param_en.requires_grad = False

    print('load checkpoint from %s' % url_or_filename)
    return model, msg