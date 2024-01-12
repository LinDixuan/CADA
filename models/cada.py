import warnings
warnings.filterwarnings("ignore")

from models.vit import VisionTransformer, interpolate_pos_embed
from models.med import BertConfig, BertModel, BertLMHeadModel, BertMLMLMHeadModel, BertJigSawModel
from models.blip import create_vit, init_tokenizer
from models.clip_models import mlm_model,GELU
from collections import OrderedDict
from transformers import BertTokenizer
import torch
from torch import nn
import torch.nn.functional as F
import os
import nltk
import math

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
        self.task = ['itc','itm','seg','attr']

        self.num_classes = num_classes
        self.logit_scale = torch.ones([]) * (1 / 0.02)
        self.temp = torch.ones([]) * (1 / 0.07)
        self.embed_dim = 256
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer,jigsaw=True)
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.config=med_config
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)
        if 'itm' not in self.task:
            for k, v in self.text_encoder.named_parameters():
                if 'cross' in k:
                    v.requires_grad = False
        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, self.embed_dim)
        self.text_proj = nn.Linear(text_width, self.embed_dim)
        if 'itm' in self.task:
            self.itm_head = nn.Linear(text_width, 2)
        if 'seg' in self.task:
            self.itm_local_head = nn.Linear(text_width, 2)
        if  'lm' in self.task or 'attr' in self.task:
            self.text_decoder = BertMLMLMHeadModel(config=med_config)
        elif 'mlm' in self.task:
            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(vision_width, vision_width)),
                             ('gelu', GELU()),
                             ('ln', nn.LayerNorm(vision_width)),
                             ('fc', nn.Linear(vision_width, med_config.vocab_size))]))
            scale = vision_width ** -0.5
            proj_std = scale * (12 ** -0.5)
            fc_std = (2 * text_width) ** -0.5
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def forward(self, image, caption, idx):

        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=73,
                              return_tensors="pt").to(image.device)
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                        return_dict=True, mode='text')
        text_embeds = text_output.last_hidden_state
        image_embeds = self.visual_encoder(image)

        text_feats = self.text_proj(text_embeds[:,0,:])

        image_feats = self.vision_proj(image_embeds[:,0,:])

        tokens = [self.tokenizer.tokenize(cap) for cap in caption]

        if 'itc' in self.task:
            itc_loss = self.compute_itc_loss(image_feats,text_feats,idx,intra=False,world=True)
        if 'itm' in self.task:
            itm_loss = self.compute_itm_loss(image_embeds,text,image_feats,text_feats,idx,tokens)

        if 'mlm' in self.task or 'lm' in self.task or 'attr' in self.task:
            cap_loss = self.compute_mlmlm_loss(text,tokens,image_embeds,'attr')

        return itc_loss, itm_loss,cap_loss


    def compute_itc_loss(self,image_feats, text_feats, labels, intra=False, world=False):
        if world:
            image_feats = all_gather_with_grad(image_feats)
            text_feats = all_gather_with_grad(text_feats)
            labels = concat_all_gather(labels)
        bs = image_feats.shape[0]

        labels = labels.view(-1,bs)
        labels = torch.eq(labels, labels.t()).float()

        # normalized features
        image_norm = image_feats / image_feats.norm(dim=-1, keepdim=True)
        text_norm = text_feats / text_feats.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_image = self.logit_scale * image_norm @ text_norm.t() #bs*5 x bs

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


    def compute_itm_loss(self,image_embeds, text, image_feat, text_feat, idx,tokens):

        idx = idx.view(-1,1)
        idxs = concat_all_gather(idx)

        encoder_input_ids = text.input_ids.clone()

        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id  # change [CLS] to special token
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        # forward the positve image-text pair
        bs = image_embeds.shape[0]
        output_pos = self.text_encoder(encoder_input_ids,  # fusion of image and text from one pair
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
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

            neg_idx = torch.argmax(weights_t2i[b])
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text (from all ranks) for each image
        input_ids_world = concat_all_gather(encoder_input_ids)
        att_mask_world = concat_all_gather(text.attention_mask)

        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.argmax(weights_i2t[b])
            text_ids_neg.append(input_ids_world[neg_idx])
            text_atts_neg.append(att_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)  # ||                ||
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        output_neg = self.text_encoder(text_ids_all,
                                       attention_mask=text_atts_all,
                                       encoder_hidden_states=image_embeds_all,
                                       encoder_attention_mask=image_atts_all,
                                       return_dict=True,
                                       )

        if 'seg' in self.task:
            last_hid = output_pos.last_hidden_state[:,1:,:]
            pos_segments = [last_hid[:, :36], last_hid[:,36:72]]
            pos_segments = torch.cat(pos_segments,dim=0)
            pos_segments = torch.mean(pos_segments,dim=1)

            last_hid = output_neg.last_hidden_state[:,1:,:]
            neg_segments = [last_hid[:, :36], last_hid[:,36:72]]
            neg_segments = torch.cat(neg_segments, dim=0)
            neg_segments = torch.mean(neg_segments,dim=1)

            vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]],
                                      dim=0)  # multi-feat
            vl_output = self.itm_head(vl_embeddings)  # (bs*3) * 2

            itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                                   dim=0).to(image_embeds.device)  # (bs * 3)

            lc_embeddings = torch.cat([pos_segments,neg_segments],dim=0)
            lc_labels = torch.cat([torch.ones(2*bs, dtype=torch.long), torch.zeros(4 * bs, dtype=torch.long)],
                                   dim=0).to(image_embeds.device)
            lc_output = self.itm_local_head(lc_embeddings)
            loss_lc = F.cross_entropy(lc_output,lc_labels)
            loss = F.cross_entropy(vl_output, itm_labels) + loss_lc #+ loss_gl

        else:
            vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]],
                                      dim=0)  # multi-feat
            vl_output = self.itm_head(vl_embeddings)  # (bs*3) * 2

            itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                                   dim=0).to(image_embeds.device)  # (bs * 3)
            loss = F.cross_entropy(vl_output, itm_labels)
        return loss


    def compute_irr_loss(self,text,image_embeds,tokens):
        input_ids = text.input_ids.clone()
        input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = input_ids.clone()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        mlm_probability = 0.15

        probability_matrix = torch.full(labels.shape, mlm_probability)
        """mlm_input_ids, mlm_labels = self.mask_irr(input_ids, vocab_size=self.config.vocab_size,device=image_embeds.device,
                                      targets=labels, probability_matrix=probability_matrix)"""
        """mlm_input_ids, mlm_labels = self.attr_mask(input_ids, tokens,
                                                  targets=labels, probability_matrix=probability_matrix)"""
        mlm_input_ids, mlm_labels = self.mask(input_ids, self.config.vocab_size,image_embeds.device,
                                                   targets=labels, probability_matrix=probability_matrix)
        mlm_embeds = self.text_encoder(mlm_input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True).last_hidden_state
        dim = mlm_embeds.shape[-1]
        score = self.mlm_head(mlm_embeds.reshape(-1,dim))
        loss = F.cross_entropy(score,mlm_labels.reshape(-1))
        return loss


    def compute_mlmlm_loss(self,text,token,image_embeds,task='mlm'):
        assert task in ['lm','mlm','mlmlm','ngram','attr']
        loss = 0.
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        if task == 'ngram':
            input_ids = text.input_ids.clone()
            input_ids[:, 0] = self.tokenizer.bos_token_id
            labels = input_ids.clone()
            mlm_probability = 0.2

            probability_matrix = torch.full(labels.shape, mlm_probability)
            input_ids, labels = self.N_gram_mask(input_ids, targets=labels, probability_matrix=probability_matrix)
            decoder_output_mlm = self.text_decoder(input_ids,
                                                   attention_mask=text.attention_mask,
                                                   encoder_hidden_states=image_embeds,
                                                   encoder_attention_mask=image_atts,
                                                   labels=labels,
                                                   return_dict=True,
                                                   output_hidden_states=True,
                                                   task='mlm'
                                                   )
            loss += decoder_output_mlm.loss


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
            mlm_probability = 0.4

            probability_matrix = torch.full(mlm_labels.shape, mlm_probability)
            mlm_input_ids, mlm_labels = self.attr_mask(mlm_input_ids, token,device=image_embeds.device,
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

        if task == 'attr':
            mlm_input_ids = text.input_ids.clone()
            mlm_labels = mlm_input_ids.clone()
            mlm_probability = 0.4

            probability_matrix = torch.full(mlm_labels.shape, mlm_probability)
            mlm_input_ids, mlm_labels = self.attr_mask(mlm_input_ids, token,
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

    def mask_irr(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = 0  # We only compute loss on masked tokens

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


    def attr_mask(self, input_ids, tokens, targets=None, masked_indices=None, probability_matrix=None):
        phrase = False
        if phrase == True:
            mask_pos = torch.zeros(probability_matrix.shape)
            grammar = "ATTR: {(<JJ><CC>?)*<NN>?<NNS>?}"
            cp = nltk.RegexpParser(grammar)

            for i, tok in enumerate(tokens):
                tree = cp.parse(nltk.pos_tag(tok))
                attr_tag = 0
                in_attr = False
                for j, pos in enumerate(tree.pos()):
                    if pos[1] == 'ATTR':
                        if in_attr:
                            pass
                        else:
                            attr_tag = attr_tag + 1
                            in_attr = True
                        if j < mask_pos.shape[1] - 1:
                            mask_pos[i, j + 1] = attr_tag
                    else:
                        in_attr = False
                probability = torch.cat([torch.zeros(1), torch.ones(attr_tag) * 0.8])
                masked_attr = torch.bernoulli(probability)
                for j in range(mask_pos.shape[1]):
                    mask_pos[i, j] = masked_attr[int(mask_pos[i, j])]
            if masked_indices is None:
                masked_indices = mask_pos.bool()
        else:
            for i,tok in enumerate(tokens):
                for j,pos in enumerate(nltk.pos_tag(tok),start=1):
                    if pos[1] not in ['JJ','NN','NNS','NNP','NNPS']:
                            probability_matrix[i,j] = 0
            if masked_indices is None:
                masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        input_ids[masked_indices] = self.tokenizer.mask_token_id
        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

def build_model(pretrained='',mode='train',**kwargs):
    model = CADA(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained,mode)
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


    print('load checkpoint from %s' % url_or_filename)
    return model, msg


