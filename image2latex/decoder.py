import torch
from torch import nn, Tensor
from .attention import Attention


class Decoder(nn.Module):
    def __init__(
        self,
        n_class: int,
        emb_dim: int = 80,
        enc_dim: int = 512,
        dec_dim: int = 512,
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
        self.attention = Attention(enc_dim, dec_dim, attn_dim)
        self.concat = nn.Linear(emb_dim + enc_dim, dec_dim)
        self.rnn = nn.LSTM(
            dec_dim,
            dec_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        # self.layernorm = nn.LayerNorm((dec_dim))
        self.out = nn.Linear(dec_dim, n_class)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.apply(self.init_weights)

    def init_weights(self, layer):
        if isinstance(layer, nn.Embedding):
            nn.init.orthogonal_(layer.weight)
        elif isinstance(layer, nn.LSTM):
            for name, param in self.rnn.named_parameters():
                if name.startswith("weight"):
                    nn.init.orthogonal_(param)

    def forward(self, y, encoder_out=None, hidden_state=None):
        """
            input:
                y: (bs, target_len)
                h: (bs, dec_dim)
                V: (bs, enc_dim, w, h)
        """
        h, c = hidden_state
        embed = self.embedding(y)
        attn_context = self.attention(h, encoder_out)

        rnn_input = torch.cat([embed[:, -1], attn_context], dim=1)
        rnn_input = self.concat(rnn_input)

        rnn_input = rnn_input.unsqueeze(1)
        hidden_state = h.unsqueeze(0), c.unsqueeze(0)
        out, hidden_state = self.rnn(rnn_input, hidden_state)
        # out = self.layernorm(out)
        out = self.logsoftmax(self.out(out))
        h, c = hidden_state
        return out, (h.squeeze(0), c.squeeze(0))
