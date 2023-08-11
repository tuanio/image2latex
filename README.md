# Image to Latex

## Introduction

This respository implement the Seq2Seq Image to Latex architecture from paper “Image to Latex.” of Genthial, Guillaume. (2017).

## Architecture

This structure is based on Seq2Seq architecture, it use one Convolutional Encoder and one RNN Decoder.

- Convolution (only)
- Convolution with Row Encoder (BiLSTM)
- Convolution with Batch Norm
- ResNet 18 with Row Encoder (BiLSTM)
- ResNet 18 (only)


<div>
    <image src="https://deforani.sirv.com/Images/Github/Image2Latex/image2latex.png" />
</div>

## Dataset
### im2latex100k
- https://www.kaggle.com/datasets/shahrukhkhan/im2latex100k
- https://www.kaggle.com/datasets/tuannguyenvananh/im2latex-sorted-by-size
### im2latex170k
- https://www.kaggle.com/datasets/rvente/im2latex170k
- https://www.kaggle.com/datasets/tuannguyenvananh/im2latex-170k-meta-data

## How to use?

### Login wandb
- `wandb login <key>`

```python main.py --batch-size 2 --data-path C:\Users\nvatu\OneDrive\Desktop\dataset5\dataset5 --img-path C:\Users\nvatu\OneDrive\Desktop\dataset5\dataset5\formula_images --dataset 170k --val --decode-type beamsearch```

## Example
- <a href="https://www.kaggle.com/code/tuannguyenvananh/image2latex-resnetbilstm-lstm">ResNet Row Encoder</a>
