"""ConvLSTM variant with tanh gating, decoder time features, and a confidence head."""
import torch
import torch.nn as nn


class STAGTanh(nn.Module):
    """Spatial gate with tanh, allowing both amplification and suppression."""

    def __init__(self, input_dim, hidden_dim, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 1,
            kernel_size=kernel_size, padding=kernel_size // 2,
        )

    def forward(self, x, h_prev):
        g = torch.tanh(self.conv(torch.cat([x, h_prev], dim=1)))
        return x * (1 + g)


class _Cell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        self.sa = STAGTanh(input_dim, hidden_dim, kernel_size)
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim,
            kernel_size, padding=kernel_size // 2, bias=True,
        )

    def forward(self, x, cur_state):
        h_cur, c_cur = cur_state
        x = self.sa(x, h_cur)
        i, f, o, g = torch.split(
            self.conv(torch.cat([x, h_cur], dim=1)), self.hidden_dim, dim=1
        )
        i, f, o, g = (
            torch.sigmoid(i), torch.sigmoid(f),
            torch.sigmoid(o), torch.tanh(g),
        )
        c = f * c_cur + i * g
        return o * torch.tanh(c), c

    @property
    def hidden_dim(self):
        return self.conv.out_channels // 4


class STAGCTanh(nn.Module):
    """Encoder-decoder ConvLSTM without temporal attention.

    Input:  (B, T, C, H, W)
    Output: (pred, conf), each (B, L, 1, H, W).
    """

    def __init__(self, input_dim=3, hidden_dim=64, kernel_size=3,
                 num_layers=2, output_len=12):
        super().__init__()
        self.output_len = output_len
        self.hidden_dim = hidden_dim
        self.encoder_cells = nn.ModuleList([
            _Cell(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size)
            for i in range(num_layers)
        ])
        # Decoder input: previous speed + two time features.
        self.decoder_cells = nn.ModuleList([
            _Cell(3 if i == 0 else hidden_dim, hidden_dim, kernel_size)
            for i in range(num_layers)
        ])
        self.final_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.conf_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def _future_time(self, x, k):
        """Extrapolate hour/weekday for forecast step k from the last input frame."""
        B = x.shape[0]
        h = x[:, -1, 1].amax(dim=(1, 2))
        w = x[:, -1, 2].amax(dim=(1, 2))
        hh = (h * 288 + k) % 288 / 288
        day_shift = ((h * 288 + k) // 288).long()
        ww = ((w * 7 + day_shift) % 7) / 7.0
        hh = hh.view(B, 1, 1, 1).expand(B, 1, x.shape[3], x.shape[4])
        ww = ww.view(B, 1, 1, 1).expand(B, 1, x.shape[3], x.shape[4])
        return torch.cat([hh, ww], dim=1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        hiddens = [
            torch.zeros(B, self.hidden_dim, H, W, device=x.device)
            for _ in self.encoder_cells
        ]
        cells = [
            torch.zeros(B, self.hidden_dim, H, W, device=x.device)
            for _ in self.encoder_cells
        ]

        for t in range(T):
            cur = x[:, t]
            for i, cell in enumerate(self.encoder_cells):
                hiddens[i], cells[i] = cell(cur, (hiddens[i], cells[i]))
                cur = hiddens[i]

        preds, confs = [], []
        cur = x[:, -1, 0:1]
        for k in range(1, self.output_len + 1):
            d = torch.cat([cur, self._future_time(x, k)], dim=1)
            for i, cell in enumerate(self.decoder_cells):
                hiddens[i], cells[i] = cell(d, (hiddens[i], cells[i]))
                d = hiddens[i]
            out = self.final_conv(hiddens[-1])
            confs.append(self.conf_head(hiddens[-1]).unsqueeze(1))
            preds.append(out.unsqueeze(1))
            cur = out
        return torch.cat(preds, dim=1), torch.cat(confs, dim=1)
