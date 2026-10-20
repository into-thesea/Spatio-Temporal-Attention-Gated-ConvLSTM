import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """STAG: spatial gating before the ConvLSTM cell."""

    def __init__(self, input_dim, hidden_dim, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 1,
            kernel_size=kernel_size, padding=kernel_size // 2,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h_prev):
        # x: (B, C_in, H, W), h_prev: (B, C_hidden, H, W)
        combined = torch.cat([x, h_prev], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * (1 + attention)


class TemporalAttention(nn.Module):
    """Query-key attention over encoder hidden states during decoding."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, query_hidden, history_hiddens):
        # query_hidden: (B, C, H, W), history_hiddens: (B, T, C, H, W)
        B, T, C, H, W = history_hiddens.size()
        q = self.query(torch.mean(query_hidden, dim=[2, 3])).unsqueeze(1)
        k = self.key(torch.mean(history_hiddens, dim=[3, 4]))
        scores = torch.matmul(q, k.transpose(1, 2)) / self.scale
        weights = F.softmax(scores, dim=-1)
        history_flat = history_hiddens.view(B, T, -1)
        return torch.matmul(weights, history_flat).view(B, C, H, W)


class ACConvLSTMCell(nn.Module):
    """ConvLSTM cell with STAG applied to the input."""

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.padding = kernel_size // 2
        self.sa = SpatialAttention(input_dim, hidden_dim, kernel_size)
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias,
        )

    def forward(self, x, cur_state):
        h_cur, c_cur = cur_state
        x_weighted = self.sa(x, h_cur)
        combined = torch.cat([x_weighted, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i, f, o, g = (
            torch.sigmoid(cc_i), torch.sigmoid(cc_f),
            torch.sigmoid(cc_o), torch.tanh(cc_g),
        )
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ACConvLSTM(nn.Module):
    """ConvLSTM encoder-decoder with STAG and temporal attention."""

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, output_len):
        super().__init__()
        self.output_len = output_len
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.encoder_cells = nn.ModuleList([
            ACConvLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size)
            for i in range(num_layers)
        ])
        self.decoder_cells = nn.ModuleList([
            ACConvLSTMCell(1 if i == 0 else hidden_dim, hidden_dim, kernel_size)
            for i in range(num_layers)
        ])
        self.ta = TemporalAttention(hidden_dim)
        self.fusion_conv = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(p=0.2)
        self.final_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        B, T, C, H, W = x.size()
        hiddens = [
            torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
            for _ in range(self.num_layers)
        ]
        cells = [
            torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
            for _ in range(self.num_layers)
        ]
        history_h = []

        for t in range(T):
            cur_in = x[:, t]
            for i in range(self.num_layers):
                hiddens[i], cells[i] = self.encoder_cells[i](cur_in, (hiddens[i], cells[i]))
                cur_in = hiddens[i]
            history_h.append(hiddens[-1].unsqueeze(1))
        history_h = torch.cat(history_h, dim=1)

        outputs = []
        cur_dec_in = x[:, -1, 0:1]
        for _ in range(self.output_len):
            context = self.ta(hiddens[-1], history_h)
            for i in range(self.num_layers):
                hiddens[i], cells[i] = self.decoder_cells[i](cur_dec_in, (hiddens[i], cells[i]))
                cur_dec_in = hiddens[i]
            fused_h = self.fusion_conv(torch.cat([hiddens[-1], context], dim=1))
            fused_h = self.dropout(fused_h)
            hiddens[-1] = fused_h
            out_frame = self.final_conv(fused_h)
            outputs.append(out_frame.unsqueeze(1))
            cur_dec_in = out_frame
        return torch.cat(outputs, dim=1)
