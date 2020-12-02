

## Introduction

Code for our AAAI 2021 paper *Confidence-aware Non-repetitive Multimodal Transformers for TextCaps*.



## Installation

Our implementation is based on Pythia framework (now called [*mmf*](https://github.com/facebookresearch/mmf)), and built upon [M4C-Captioner](https://github.com/ronghanghu/pythia/tree/project/m4c_captioner_pre_release/projects/M4C_Captioner). Please refer to [Pythia's document](https://mmf.sh/docs/) for more details on installation requirements.

```shell
# install pythia based on requirements.txt
python setup.py build develop  

# install pycocoevalcap
# use the repo below instead of https://github.com/tylin/coco-caption
# note: you also need to have java on your machine
pip install git+https://github.com/ronghanghu/coco-caption.git@python23
```

Note that installing `pycocoevalcap` is required for evaluating off-line. To install `pycocoevalcap`,  java is needed.



## Data Preparation

The following is open-source data of TextCaps dataset from [M4C-Captioner's Github repository](https://github.com/ronghanghu/pythia/tree/project/m4c_captioner_pre_release/projects/M4C_Captioner). Please download them from the links below and and extract them under `data` directory.

*  [object Faster R-CNN Features of TextCaps](https://dl.fbaipublicfiles.com/pythia/features/open_images.tar.gz)
*   [OCR Faster R-CNN Features of TextCaps](https://dl.fbaipublicfiles.com/pythia/m4c/data/m4c_textvqa_ocr_en_frcn_features.tar.gz)
*  [detectron weights of TextCaps](http://dl.fbaipublicfiles.com/pythia/data/detectron_weights.tar.gz)

Our `imdb` files include new OCR tokens and recognition confidence extracted with pretrained OCR systems ( [CRAFT](https://github.com/clovaai/CRAFT-pytorch), [ABCNet](https://github.com/Yuliang-Liu/bezier_curve_text_spotting) and [four-stage STR](https://github.com/Yuliang-Liu/bezier_curve_text_spotting)). The three imdb files should be downloaded from the links below and put under `data/imdb/`.

* [imdb_train.npy](https://drive.google.com/file/d/1EzF2WB81BTs2Bgt6kFdq2PTRlQl8EQ-y/view?usp=sharing)
* [imdb_val_filtered_by_image_id.npy](https://drive.google.com/file/d/1FuqUGIsOqCkCqEGKIQAkc_08aMpjVJls/view?usp=sharing)
* [imdb_test_filtered_by_image_id.npy](https://drive.google.com/file/d/1lu3aW0oTh6CO0_L64W9PE5UNW4_H7Cj2/view?usp=sharing)


Finally, your `data` directory structure should look like this:

```shell
data
|-detectron							#download from links above
|---...
|-m4c_textvqa_ocr_en_frcn_features	#download from links above
|---...
|-open_images						#download from links above
|---...
|-vocab_textcap_threshold_10.txt 	#already provided
|-imdb								#download from links above
|---imdb_train.npy					
|---imdb_val_filtered_by_image_id.npy	
|---imdb_test_filtered_by_image_id.npy		
```



## Training

We provide an example script for training on TextCaps dataset for 12000 iterations and evaluating every 500 iterations.

```shell
./train.sh
```

This may take approximately 13 hours, depending on GPU devices. Please refer to our paper for implementation details.

First-time training will download `fasttext` model . You may also download it manually and put it under `pythia/.vector_cache/`.

During training, log file can be found under `save/cnmt/m4c_textcaps_cnmt/logs/`. You may also run training in background using `nohup` and check log file for training status.



## Evaluation

Assume that checkpoint of the trained model is saved at `save/cnmt/m4c_textcaps_cnmt/best.ckpt` (otherwise modify the `resume_file` parameter in the shell script).

### Validation set

Run the following script to generate prediction json fils for TextCaps validation set:

```shell
./eval_val.sh
```

The prediction file will be saved under `save/eval/m4c_textcaps_cnmt/reports/`.

**Evaluate the prediction file**

*  evaluate with `./textcaps_eval.py`. (**pycocoeval** is required)  For example:

```
python textcaps_eval.py --set val --pred_file YOUR_VAL_PREDICTION_FILE
```

*  Or, you can submit the json file to the TextCaps EvalAI server for result.

### Test set

Run the following script to generate prediction json file for TextCaps test set:

```shell
./eval_test.sh
```

The prediction file will be saved under `save/eval/m4c_textcaps_cnmt/reports/`.

For test set evaluation, please submit the json file to the TextCaps EvalAI server.





## Citation

```
@InProceedings{Wang_2021_AAAI,
  author = {Wang, Zhaokai and Bao, Renda and Wu, Qi and Liu, Si},
  title = {Confidence-aware Non-repetitive Multimodal Transformers for TextCaps},
  booktitle = {AAAI Conference on Artificial Intelligence (AAAI)},
  year = {2021}
}
```

