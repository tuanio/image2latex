#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import torch
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset
from torch import nn, Tensor
import matplotlib.pyplot as plt
from image2latex import Image2Latex, Text, TriStageLRScheduler
from pathlib import Path
import torchvision
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence
import glob
from tqdm.notebook import tqdm
import time
from jiwer import wer as cal_wer
from nltk.metrics import edit_distance
from typing import Tuple
import pytorch_lightning as pl
import argparse

parser = argparse.ArgumentParser(description="training image2latex")
parser.add_argument("-bs", type=int)
parser.add_argument("--root-data-path", description="Root data path")
parser.add_argument(
    "--train",
    action=argparse.BooleanOptionalAction,
    description="call this for training mode",
)
parser.add_argument(
    "--val",
    action=argparse.BooleanOptionalAction,
    description="call this for validating mode",
)
parser.add_argument(
    "--test",
    action=argparse.BooleanOptionalAction,
    description="call this for testing mode",
)

args = parser.parse_args()
root_data_path = args.root_data_path

data_path = Path(root_data_path)
img_path = Path(f"{root_data_path}/formula_images_processed/formula_images_processed")

bs = args.bs
lr = 1e-3
epochs = 15
max_length = 150
log_idx = 300
workers = 24

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

text = Text()
n_class = len(text.tokens)


class LatexDataset(Dataset):
    def __init__(self, data_type: str):
        super().__init__()
        assert data_type in ["train", "test", "validate"], "Not found data type"
        csv_path = data_path / f"im2latex_{data_type}.csv"
        df = pd.read_csv(csv_path)
        df["image"] = df.image.map(lambda x: img_path / x)
        self.walker = df.to_dict("records")

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        item = self.walker[idx]

        formula = item["formula"]
        image = torchvision.io.read_image(str(item["image"]))

        return image, formula


train_set = LatexDataset("train")
val_set = LatexDataset("validate")
test_set = LatexDataset("test")

steps_per_epoch = round(len(train_set) / bs)

warmup_epochs = 2
constant_epochs = 8
decay_epochs = 5

assert warmup_epochs + constant_epochs + decay_epochs == epochs, "Not equal"


def collate_fn(batch):
    formulas = [text.text2int(i[1]) for i in batch]
    formulas = pad_sequence(formulas, batch_first=True)
    sos = torch.zeros(bs, 1) + text.map_tokens["<s>"]
    eos = torch.zeros(bs, 1) + text.map_tokens["<e>"]
    print(eos.size(), sos.size(), formulas.size())
    formulas = torch.cat((sos, formulas, eos), dim=-1).to(dtype=torch.long)
    image = [i[0] for i in batch]
    max_width, max_height = 0, 0
    for img in image:
        c, h, w = img.size()
        max_width = max(max_width, w)
        max_height = max(max_height, h)
    pad = torchvision.transforms.Resize(size=(max_height, max_width))
    image = torch.stack(list(map(lambda x: pad(x), image))).to(dtype=torch.float)
    return image, formulas


class DataModule(pl.LightningDataModule):
    def __init__(self, train_set, val_set, test_set):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            shuffle=True,
            batch_size=bs,
            num_workers=workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            shuffle=False,
            batch_size=bs,
            num_workers=workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            shuffle=False,
            batch_size=bs,
            num_workers=workers,
            collate_fn=collate_fn,
        )


class Image2LatexModel(pl.LightningModule):
    def __init__(self, lr=lr, **kwargs):
        super().__init__()
        self.model = Image2Latex(**kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = TriStageLRScheduler(
            optimizer,
            init_lr=1e-4,
            peak_lr=1e-3,
            final_lr=1e-5,
            init_lr_scale=0.01,
            final_lr_scale=0.01,
            warmup_steps=steps_per_epoch * warmup_epochs,
            hold_steps=steps_per_epoch * constant_epochs,
            decay_steps=steps_per_epoch * decay_epochs,
            total_steps=steps_per_epoch * bs,
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",  # or 'epoch'
            "frequency": 1,
        }
        return [optimizer], [scheduler]
        # return optimizer

    def forward(self, images, formulas):
        return self.model(images, formulas)

    def training_step(self, batch, batch_idx):
        images, formulas = batch

        formulas_in = formulas[:, :-1]
        formulas_out = formulas[:, 1:]

        outputs = self.model(images, formulas_in)

        bs, t, _ = outputs.size()
        _o = outputs.reshape(bs * t, -1)
        _t = formulas_out.reshape(-1)
        loss = self.criterion(_o, _t)

        self.log("train loss", loss)
        self.log("lr", self.lr)

        return loss

    def validation_step(self, batch, batch_idx):
        images, formulas = batch

        formulas_in = formulas[:, :-1]
        formulas_out = formulas[:, 1:]

        outputs = self.model(images, formulas_in)

        bs, t, _ = outputs.size()
        _o = outputs.reshape(bs * t, -1)
        _t = formulas_out.reshape(-1)
        loss = self.criterion(_o, _t)
        perplexity = torch.exp(loss)

        predicts = [
            text.tokenize(self.model.decode(i.unsqueeze(0), max_length)) for i in images
        ]
        truths = [text.tokenize(text.int2text(i)) for i in formulas]

        edit_dist = torch.mean(
            torch.Tensor(
                [
                    edit_distance(pre, tru) / max(len(pre), len(tru))
                    for pre, tru in zip(predicts, truths)
                ]
            )
        )

        self.log("val loss", loss)
        self.log("val perplexity", perplexity)
        self.log("val edit distance", edit_dist)

        return loss

    def test_step(self, batch, batch_idx):
        images, formulas = batch

        formulas_in = formulas[:, :-1]
        formulas_out = formulas[:, 1:]

        outputs = self.model(images, formulas_in)

        bs, t, _ = outputs.size()
        _o = outputs.reshape(bs * t, -1)
        _t = formulas_out.reshape(-1)
        loss = self.criterion(_o, _t)
        perplexity = torch.exp(loss)

        predicts = [
            text.tokenize(self.model.decode(i.unsqueeze(0), max_length)) for i in images
        ]
        truths = [text.tokenize(text.int2text(i)) for i in formulas]

        edit_dist = torch.mean(
            torch.Tensor(
                [
                    edit_distance(pre, tru) / max(len(pre), len(tru))
                    for pre, tru in zip(predicts, truths)
                ]
            )
        )

        self.log("test loss", loss)
        self.log("test perplexity", perplexity)
        self.log("test edit distance", edit_dist)

        return loss


dm = DataModule(train_set, val_set, test_set)


emb_dim = 80
dec_dim = 512
enc_dim = 512
attn_dim = 512

model = Image2LatexModel(
    lr=lr,
    n_class=n_class,
    emb_dim=emb_dim,
    enc_dim=enc_dim,
    dec_dim=dec_dim,
    attn_dim=attn_dim,
    text=text,
)


tb_logger = pl.loggers.tensorboard.TensorBoardLogger(
    "tb_logs", name="image2latex_model"
)
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

trainer = pl.Trainer(
    logger=tb_logger,
    callbacks=[lr_monitor],
    max_epochs=epochs,
    accelerator="gpu",
    accumulate_grad_batches=32,
)

if args.train:
    print("=" * 10 + "[Train]" + "=" * 10)
    trainer.fit(datamodule=dm, model=model)

if args.test:
    print("=" * 10 + "[Test]" + "=" * 10)
    trainer.test(datamodule=dm, model=model)
