#!/usr/bin/env python3
"""Neural Storyteller – Gradio App for Hugging Face Spaces (Attention model)."""

import os, json, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# ── Device ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load config ──
with open("config.json", "r") as f:
    cfg = json.load(f)

EMBED_SIZE   = cfg["embed_size"]
HIDDEN_SIZE  = cfg["hidden_size"]
NUM_REGIONS  = cfg["num_regions"]
VOCAB_SIZE   = cfg["vocab_size"]
MAX_LEN      = cfg["max_len"]
DROPOUT      = cfg["dropout"]
BEAM_WIDTH   = cfg["beam_width"]
LENGTH_PEN   = cfg.get("length_penalty", 0.7)
REP_PEN      = cfg.get("repetition_penalty", 1.2)

# ── Vocabulary class (required for unpickling) ──
class Vocabulary:
    PAD, START, END, UNK = '<pad>', '<start>', '<end>', '<unk>'

    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.word2idx = {}
        self.idx2word = {}
        self._idx = 0

    def __len__(self):
        return len(self.word2idx)

# ── Load vocabulary ──
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)


# ══════════════ Model Definitions (must match training) ══════════════

class Encoder(nn.Module):
    def __init__(self, feature_dim=2048, hidden_size=HIDDEN_SIZE,
                 num_regions=NUM_REGIONS, dropout=DROPOUT):
        super().__init__()
        self.num_regions = num_regions
        self.hidden_size = hidden_size
        self.project = nn.Linear(feature_dim, hidden_size * num_regions)
        self.bn      = nn.BatchNorm1d(hidden_size * num_regions)
        self.dropout = nn.Dropout(dropout)
        self.init_h  = nn.Linear(feature_dim, hidden_size)
        self.init_c  = nn.Linear(feature_dim, hidden_size)

    def forward(self, features):
        proj = self.dropout(F.relu(self.bn(self.project(features))))
        regions = proj.view(-1, self.num_regions, self.hidden_size)
        h0 = torch.tanh(self.init_h(features))
        c0 = torch.tanh(self.init_c(features))
        return regions, h0, c0


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W_enc = nn.Linear(hidden_size, hidden_size)
        self.W_dec = nn.Linear(hidden_size, hidden_size)
        self.V     = nn.Linear(hidden_size, 1)

    def forward(self, encoder_out, decoder_hidden):
        energy  = self.V(torch.tanh(
            self.W_enc(encoder_out) + self.W_dec(decoder_hidden).unsqueeze(1)
        ))
        weights = F.softmax(energy.squeeze(2), dim=1)
        context = (weights.unsqueeze(2) * encoder_out).sum(1)
        return context, weights


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size=EMBED_SIZE,
                 hidden_size=HIDDEN_SIZE, dropout=DROPOUT):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.attention = BahdanauAttention(hidden_size)
        self.lstm_cell = nn.LSTMCell(embed_size + hidden_size, hidden_size)
        self.fc_out    = nn.Linear(hidden_size + hidden_size, vocab_size)
        self.dropout   = nn.Dropout(dropout)

    def forward_step(self, word_idx, h, c, encoder_out):
        embed   = self.dropout(self.embed(word_idx))
        context, attn_w = self.attention(encoder_out, h)
        lstm_in = torch.cat([embed, context], dim=1)
        h, c    = self.lstm_cell(lstm_in, (h, c))
        logits  = self.fc_out(self.dropout(torch.cat([h, context], dim=1)))
        return logits, h, c, attn_w


class Seq2SeqCaptioner(nn.Module):
    def __init__(self, vocab_size, embed_size=EMBED_SIZE,
                 hidden_size=HIDDEN_SIZE, dropout=DROPOUT,
                 num_regions=NUM_REGIONS):
        super().__init__()
        self.encoder     = Encoder(2048, hidden_size, num_regions, dropout)
        self.decoder     = AttentionDecoder(vocab_size, embed_size, hidden_size, dropout)
        self.hidden_size = hidden_size

    def forward(self, features, captions, teacher_forcing_ratio=1.0):
        import random
        B = features.size(0)
        T = captions.size(1) - 1
        V = self.decoder.fc_out.out_features
        encoder_out, h, c = self.encoder(features)
        outputs = torch.zeros(B, T, V, device=features.device)
        inp = captions[:, 0]
        for t in range(T):
            logits, h, c, _ = self.decoder.forward_step(inp, h, c, encoder_out)
            outputs[:, t] = logits
            if random.random() < teacher_forcing_ratio:
                inp = captions[:, t + 1]
            else:
                inp = logits.argmax(dim=-1)
        return outputs


# ── Load trained weights ──
caption_model = Seq2SeqCaptioner(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, DROPOUT, NUM_REGIONS).to(device)
caption_model.load_state_dict(torch.load("best_model.pth", map_location=device))
caption_model.eval()

# ── ResNet50 feature extractor ──
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet = resnet.to(device)
resnet.eval()

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Beam Search with penalties ──
@torch.no_grad()
def beam_search_inference(feature, beam_width=BEAM_WIDTH,
                          length_penalty=LENGTH_PEN,
                          repetition_penalty=REP_PEN):
    feature = feature.unsqueeze(0).to(device)
    encoder_out, h0, c0 = caption_model.encoder(feature)

    start_idx = vocab.word2idx[vocab.START]
    end_idx   = vocab.word2idx[vocab.END]
    pad_idx   = vocab.word2idx[vocab.PAD]

    beams     = [(0.0, [start_idx], h0, c0)]
    completed = []

    for _ in range(MAX_LEN):
        new_beams = []
        for log_prob, seq, h, c in beams:
            inp = torch.tensor([seq[-1]], device=device)
            logits, h_new, c_new, _ = caption_model.decoder.forward_step(
                inp, h, c, encoder_out)
            logits = logits.squeeze(0)

            for prev_tok in set(seq):
                if prev_tok not in (start_idx, end_idx, pad_idx):
                    logits[prev_tok] /= repetition_penalty

            log_probs = F.log_softmax(logits, dim=-1)
            topk_lp, topk_idx = log_probs.topk(beam_width)

            for k in range(beam_width):
                token   = topk_idx[k].item()
                new_lp  = log_prob + topk_lp[k].item()
                new_seq = seq + [token]
                if token == end_idx:
                    score = new_lp / (len(new_seq) ** length_penalty)
                    completed.append((score, new_seq))
                else:
                    new_beams.append((new_lp, new_seq, h_new, c_new))

        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]
        if not beams or len(completed) >= beam_width:
            break

    if not completed and beams:
        for lp, seq, _, _ in beams:
            completed.append((lp / (len(seq) ** length_penalty), seq))

    completed.sort(key=lambda x: x[0], reverse=True)
    best_seq = completed[0][1] if completed else [start_idx]

    words = [vocab.idx2word[i] for i in best_seq
             if vocab.idx2word[i] not in (vocab.START, vocab.END, vocab.PAD)]
    return " ".join(words)


# ── Prediction function for Gradio ──
def predict(image):
    """Take a PIL image -> return generated caption string."""
    if image is None:
        return "Please upload an image."
    image = image.convert("RGB")
    img_tensor = img_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = resnet(img_tensor).view(1, -1).squeeze(0)

    caption = beam_search_inference(feature)
    return caption


# ── Gradio Interface ──
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="🧠 Neural Storyteller – Image Captioning",
    description=(
        "Upload any image and this Seq2Seq model (ResNet50 encoder + "
        "Attention LSTM decoder) trained on Flickr30k will generate "
        "a natural language caption using beam search."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
