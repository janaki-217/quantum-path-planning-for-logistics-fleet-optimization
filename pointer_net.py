# pointer_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, n_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, bidirectional=False)

    def forward(self, x):
        # x: (B, n, 2)
        out, (h, c) = self.lstm(x)  # out: (B, n, hidden)
        return out  # sequence outputs

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden, mask):
        # encoder_outputs: (B, n, H)
        # decoder_hidden: (B, H)
        # mask: (B, n) bool, True if already visited
        # compute scores
        # score = v^T * tanh(W1 * encoder + W2 * dec)
        enc_trans = self.W1(encoder_outputs)                    # (B, n, H)
        dec_trans = self.W2(decoder_hidden).unsqueeze(1)        # (B,1,H)
        scores = self.v(torch.tanh(enc_trans + dec_trans)).squeeze(-1)  # (B,n)
        scores = scores.masked_fill(mask, float('-inf'))
        return scores  # un-normalized logits

class Decoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(hidden_dim, hidden_dim)
        self.attn = Attention(hidden_dim)
        self.hidden_dim = hidden_dim
        self.project_input = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, encoder_outputs, decode_steps, deterministic=False):
        # encoder_outputs: (B, n, H)
        B, n, H = encoder_outputs.size()
        device = encoder_outputs.device

        # initial decoder input: mean of encoder outputs
        dec_input = encoder_outputs.mean(dim=1)  # (B, H)
        hx = torch.zeros(B, H, device=device)
        cx = torch.zeros(B, H, device=device)

        mask = torch.zeros(B, n, dtype=torch.bool, device=device)
        tours = []
        log_probs = []

        for _ in range(decode_steps):
            hx, cx = self.lstm_cell(dec_input, (hx, cx))  # (B,H)
            logits = self.attn(encoder_outputs, hx, mask)  # (B,n)
            probs = torch.softmax(logits, dim=-1)

            if deterministic:
                idx = probs.argmax(dim=-1)
            else:
                idx = torch.multinomial(probs, 1).squeeze(-1)

            selected_log_prob = torch.log(torch.gather(probs, 1, idx.unsqueeze(1)).squeeze(1) + 1e-9)
            tours.append(idx)
            log_probs.append(selected_log_prob)

            mask[torch.arange(B), idx] = True
            # next decoder input is the encoder vector for selected node (teacher forcing not used here)
            dec_input = torch.gather(encoder_outputs, 1, idx.view(B,1,1).expand(-1, -1, H)).squeeze(1)

        tours = torch.stack(tours, dim=1)        # (B, n)
        log_probs = torch.stack(log_probs, dim=1)  # (B, n)
        return tours, log_probs

class PointerNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128):
        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.decoder = Decoder(hidden_dim)

    def forward(self, x, deterministic=False):
        # x: (B,n,2)
        enc = self.encoder(x)
        tours, log_probs = self.decoder(enc, decode_steps=enc.size(1), deterministic=deterministic)
        return tours, log_probs
