import random
import torch
from torch import nn, Tensor
from .decoder import Decoder
from .encoder import *
from .text import Text


class Image2Latex(nn.Module):
    def __init__(
        self,
        n_class: int,
        enc_dim: int = 512,
        enc_type: str = "conv_row_encoder",
        emb_dim: int = 80,
        dec_dim: int = 512,
        attn_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        decode_type: str = "greedy",
        text: Text = None,
        beam_width: int = 5,
        sos_id: int = 1,
        eos_id: int = 2,
    ):
        assert enc_type in [
            "conv_row_encoder",
            "conv_encoder",
            "conv_bn_encoder",
            "resnet_encoder",
            "resnet_row_encoder",
        ], "Not found encoder"
        super().__init__()
        self.n_class = n_class
        if enc_type == "conv_row_encoder":
            self.encoder = ConvWithRowEncoder(enc_dim=enc_dim)
        elif enc_type == "conv_encoder":
            self.encoder = ConvEncoder(enc_dim=enc_dim)
        elif enc_type == "conv_bn_encoder":
            self.encoder = ConvBNEncoder(enc_dim=enc_dim)
        elif enc_type == "resnet_encoder":
            self.encoder = ResNetEncoder(enc_dim=enc_dim)
        elif enc_type == "resnet_row_encoder":
            self.encoder = ResNetWithRowEncoder(enc_dim=enc_dim)
        enc_dim = self.encoder.enc_dim
        self.num_layers = num_layers
        self.decoder = Decoder(
            n_class=n_class,
            emb_dim=emb_dim,
            dec_dim=dec_dim,
            enc_dim=enc_dim,
            attn_dim=attn_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            sos_id=sos_id,
            eos_id=eos_id,
        )
        self.init_h = nn.Linear(enc_dim, dec_dim)
        self.init_c = nn.Linear(enc_dim, dec_dim)
        assert decode_type in ["greedy", "beamsearch"]
        self.decode_type = decode_type
        self.text = text
        self.beam_width = beam_width

    def init_decoder_hidden_state(self, V: Tensor):
        """
            return (h, c)
        """
        encoder_mean = V.mean(dim=1)
        h = torch.tanh(self.init_h(encoder_mean))
        c = torch.tanh(self.init_c(encoder_mean))
        return h, c

    def forward(self, x: Tensor, y: Tensor, y_len: Tensor):
        encoder_out = self.encoder(x)

        hidden_state = self.init_decoder_hidden_state(encoder_out)

        predictions = []
        for t in range(y_len.max().item()):
            dec_input = y[:, t].unsqueeze(1)
            out, hidden_state = self.decoder(dec_input, encoder_out, hidden_state)
            predictions.append(out.squeeze(1))

        predictions = torch.stack(predictions, dim=1)
        return predictions

    def decode(self, x: Tensor, max_length: int = 150):
        predict = []
        if self.decode_type == "greedy":
            predict = self.decode_greedy(x, max_length)
        elif self.decode_type == "beamsearch":
            predict = self.decode_beam_search(x, max_length)
        return self.text.int2text(predict)

    def decode_greedy(self, x: Tensor, max_length: int = 150):
        encoder_out = self.encoder(x)
        bs = encoder_out.size(0)

        hidden_state = self.init_decoder_hidden_state(encoder_out)

        y = torch.LongTensor([self.decoder.sos_id]).view(bs, -1)

        hidden_state = self.init_decoder_hidden_state(encoder_out)

        predictions = []
        for t in range(max_length):
            out, hidden_state = self.decoder(y, encoder_out, hidden_state)

            k = out.argmax().item()

            predictions.append(k)

            y = torch.LongTensor([k]).view(bs, -1)
        return predictions

    def decode_beam_search(self, x: Tensor, max_length: int = 150):
        """
            default: batch size equal to 1
        """
        encoder_out = self.encoder(x)
        bs = encoder_out.size(0)  # 1

        hidden_state = self.init_decoder_hidden_state(encoder_out)

        list_candidate = [
            ([self.decoder.sos_id], hidden_state, 0)
        ]  # (input, hidden_state, log_prob)
        for t in range(max_length):
            new_candidates = []
            for inp, state, log_prob in list_candidate:
                y = torch.LongTensor([inp[-1]]).view(bs, -1).to(device=x.device)
                out, hidden_state = self.decoder(y, encoder_out, state)

                topk = out.topk(self.beam_width)
                new_log_prob = topk.values.view(-1).tolist()
                new_idx = topk.indices.view(-1).tolist()
                for val, idx in zip(new_log_prob, new_idx):
                    new_inp = inp + [idx]
                    new_candidates.append((new_inp, hidden_state, log_prob + val))

            new_candidates = sorted(new_candidates, key=lambda x: x[2], reverse=True)
            list_candidate = new_candidates[: self.beam_width]

        return list_candidate[0][0]
