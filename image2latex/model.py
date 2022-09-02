import random
import torch
from torch import nn, Tensor
from . import Encoder, Decoder
from .text import Text


class Image2Latex(nn.Module):
    def __init__(
        self,
        n_class: int,
        enc_dim: int = 512,
        emb_dim: int = 512,
        dec_dim: int = 512,
        attn_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        sos_id: int = 1,
        eos_id: int = 2,
        decode_type: str = "greedy",
        text: Text = None,
    ):
        super().__init__()
        self.n_class = n_class
        self.encoder = Encoder(enc_dim=enc_dim)
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
        assert decode_type in ["greedy", "beam"]
        self.decode_type = decode_type
        self.text = text

    def init_decoder_hidden_state(self, V: Tensor):
        """
            return (h, c)
        """
        encoder_mean = V.mean(dim=[2, 3])
        h = torch.tanh(self.init_h(encoder_mean))
        c = torch.tanh(self.init_c(encoder_mean))
        return h, c

    def forward(self, x: Tensor, y: Tensor, teacher_forcing_ratio: float = 0.5):
        V = self.encoder(x)
        h, c = self.init_decoder_hidden_state(V)

        bs, target_len = y.size()
        zeros = torch.zeros(target_len, bs, self.n_class)
        outputs = x.new_tensor(zeros)

        o = None
        _input = y[:, 0]  # get first element of all batch
        for t in range(1, target_len):
            _input = _input.unsqueeze(1)

            output, (h, c), o = self.decoder(_input, V, h, c, o)

            outputs[t] = output

            teacher_force = random.random() < teacher_forcing_ratio

            top_1 = output.argmax(-1)

            _input = y[:, t] if teacher_force else top_1

        return outputs.permute(1, 0, 2)

    def decode(self, x: Tensor, max_length: int):
        if self.decode_type == "greedy":
            predict = self.decode_greedy(x, max_length)
            predict = self.text.int2text(predict)
            return predict
        # beam

    def decode_greedy(self, x: Tensor, max_length: int):
        V = self.encoder(x)
        h, c = self.init_decoder_hidden_state(V)

        bs = V.size(0)
        assert bs == 1, "Batch size must be 1"
        predict = []

        o = None
        _input = x.new_tensor(torch.zeros(bs) + self.decoder.sos_id).to(
            dtype=torch.long
        )
        for t in range(1, max_length):
            _input = _input.unsqueeze(1)

            output, (h, c), o = self.decoder(_input, V, h, c, o)

            top_1 = output.argmax(-1)  # greedy decode

            if top_1.item() == self.decoder.eos_id:
                break

            predict.append(top_1)

            _input = top_1

        return torch.LongTensor(predict)
