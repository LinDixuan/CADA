# Cross-Modal Adaptive Dual Association for Text-to-Image Person Retrieval

![main](model.png)



## Abstract 

Text-to-image person re-identification (ReID) aims to retrieve images of a person based on a given textual description. The key challenge is to learn the relations between detailed information from visual and textual modalities. Existing work focuses on learning a latent space to narrow the modality gap and further build local correspondences between two modalities. 
However, these methods assume that image-to-text and text-to-image associations are modality-agnostic, resulting in suboptimal associations. 
In this work, we demonstrate the discrepancy between image-to-text association and text-to-image association and proposecross-modal adaptive dual association (CADA) to build fine bidirectional image-text detailed associations. 
 Our approach features a decoder-based adaptive dual association module that enables full interaction between visual and
textual modalities, enabling bidirectional and adaptive cross-modal correspondence associations. 
Specifically, this paper proposes a bidirectional association mechanism: Association of text Tokens to image Patches (ATP) and Association of image Regions to text Attributes (ARA).
We adaptively model the ATP based on the fact that aggregating cross-modal features based on mistaken associations will lead to feature distortion.
For modeling the ARA, since attributes are typically the first distinguishing cues of a person, we explore attribute-level associations by predicting the masked text phrase using the related image region.
Finally, we learn the dual associations between texts and images, and the experimental results demonstrate the superiority of our dual formulation.



### Getting started

Clone this repo

```
git clone https://github.com/LinDixuan/CADA.git
```

Install dependencies

```
pip install -r requirements.txt
```

Any components in NLTK can be found [here](https://github.com/nltk/nltk_data/tree/gh-pages).



### Prepare pretrained model

Download pretrained baseline from [model_base](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth ) from [BLIP](https://github.com/salesforce/BLIP) and put **model_base.pth** at **./checkpoint**



### Prepare datasets

Download datasets: [CUHK-PEDES](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description)   [ICFG-PEDES](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description)  [RSTPReid](https://github.com/NjtechCVLab/RSTPReid-Dataset) and put at **./datasets**

The files shall be organized as:

```
|---configs
|---data
|---models
|---datasets
|	---CUHK-PEDES
|	---ICFG-PEDES
|	---RSTPReid
|---checkpoint
|	---model_base.pth
|
|
```



### Train

```
bash train.bash
```



If you have problem in downloading pretrained BERT models here:

```
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
```

you may manually download it from [hugging face](https://huggingface.co/docs/transformers/model_doc/bert#bert) and use this instead:

```
tokenizer = BertTokenizer.from_pretrained(bert_localpath)
```



### Evaluation

```
bash eval.bash
```



### Acknowledgments

Sincerely appreciate the contributions of [BLIP](https://github.com/salesforce/BLIP), [NLTK](https://github.com/nltk/nltk_data/tree/gh-pages), and [IRRA](https://github.com/anosorae/IRRA), for we used codes and pretrained models from these previous works.