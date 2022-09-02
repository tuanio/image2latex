import torch
from torch import nn, Tensor
from .attention import Attention


class Decoder(nn.Module):
    def __init__(
        self,
        n_class: int,
        emb_dim: int = 512,
        dec_dim: int = 512,
        enc_dim: int = 512,
        attn_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        sos_id: int = 1,
        eos_id: int = 2,
    ):
        super().__init__()
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(n_class, emb_dim)
        self.attn = Attention(enc_dim, dec_dim, attn_dim)
        self.rnn = nn.LSTM(
            emb_dim + attn_dim,
            dec_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.concat = nn.Linear(dec_dim + attn_dim, dec_dim)
        self.out = nn.Linear(dec_dim, n_class)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, y: Tensor, V: Tensor, h: Tensor, c: Tensor, o: Tensor = None):
        """
            input:
                y: (bs, target_len)
                h: (bs, dec_dim)
                V: (bs, enc_dim, w, h)
                o: ()
        """

        bs, enc_dim, *_ = V.size()
        V = V.view(bs, -1, enc_dim)

        if not isinstance(o, Tensor):  # when not have o yet, we initialize it
            context = self.attn(h, V)
            concat_input = torch.cat((h, context), dim=-1)
            o = torch.tanh(self.concat(concat_input))

        embedded = self.embedding(y)
        rnn_input = torch.stack(
            [torch.cat((i, o), -1) for i in embedded.permute(1, 0, 2)]
        ).permute(1, 0, 2)
        out, (h, c) = self.rnn(rnn_input, (h.unsqueeze(0), c.unsqueeze(0)))
        h = h.squeeze(0)
        c = c.squeeze(0)
        context = self.attn(h, V)

        concat_input = torch.cat((h, context), dim=-1)
        o = torch.tanh(self.concat(concat_input))
        out = self.out(o)
        return self.softmax(out), (h, c), o
