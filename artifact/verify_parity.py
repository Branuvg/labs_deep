import json
import math
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
D_MODEL, N_HEADS, D_K = 32, 2, 16


def positional_encoding(max_len):
    pe = torch.zeros(max_len, D_MODEL)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, D_MODEL, 2).float() * (-math.log(10000.0) / D_MODEL))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def layer_norm(x, gamma, beta):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return gamma * (x - mean) / torch.sqrt(var + 1e-5) + beta


def forward(idxs, raw_weights, max_len, key_mask=None, causal=False, output='Wcls'):
    w = {name: torch.tensor(value) for name, value in raw_weights.items()}
    x = w['E'][idxs] + positional_encoding(max_len)
    q, k, v = x @ w['WQ'], x @ w['WK'], x @ w['WV']
    q = q.view(max_len, N_HEADS, D_K).transpose(0, 1)
    k = k.view(max_len, N_HEADS, D_K).transpose(0, 1)
    v = v.view(max_len, N_HEADS, D_K).transpose(0, 1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(D_K)
    if key_mask is not None:
        scores = scores.masked_fill((~key_mask).view(1, 1, max_len), float('-inf'))
    if causal:
        scores = scores.masked_fill(torch.triu(torch.ones(max_len, max_len), diagonal=1).bool(), float('-inf'))
    attention = F.softmax(scores, dim=-1)
    attended = (attention @ v).transpose(0, 1).contiguous().view(max_len, D_MODEL) @ w['WO']
    x2 = layer_norm(x + attended, w['gamma1'], w['beta1'])
    ff = F.relu(x2 @ w['W1'] + w['bias1']) @ w['W2'] + w['bias2']
    x3 = layer_norm(x2 + ff, w['gamma2'], w['beta2'])
    if output == 'Wcls':
        logits = x3[0] @ w['Wcls'] + w['bcls']
    else:
        logits = x3 @ w['Wlm'] + w['blm']
    return logits, attention


encoder_data = json.loads((ROOT / 'encoder_weights.json').read_text())
gpt_data = json.loads((ROOT / 'gpt_weights.json').read_text())
js = json.loads(subprocess.check_output(['node', str(ROOT / 'artifact/verify.mjs'), '--json'], text=True))

encoder_w2i = {word: i for i, word in enumerate(encoder_data['vocab'])}
words = 'this movie is surprisingly good'.split()
encoder_tokens = ['<CLS>'] + [word if word in encoder_w2i else '<UNK>' for word in words]
encoder_idxs = [encoder_w2i[token] for token in encoder_tokens]
encoder_mask = [True] * len(encoder_idxs)
encoder_idxs += [encoder_w2i['<PAD>']] * (16 - len(encoder_idxs))
encoder_mask += [False] * (16 - len(encoder_mask))
enc_logits, enc_attention = forward(
    torch.tensor(encoder_idxs), encoder_data['weights'], 16, torch.tensor(encoder_mask)
)

gpt_w2i = {word: i for i, word in enumerate(gpt_data['vocab'])}
gpt_tokens = ['<BOS>', 'the', 'film', '<EOS>'] + ['<PAD>'] * 10
gpt_idxs = torch.tensor([gpt_w2i.get(token, gpt_w2i['<UNK>']) for token in gpt_tokens])
gpt_logits, gpt_attention = forward(gpt_idxs, gpt_data['weights'], 14, causal=True, output='Wlm')

js_enc_logits = torch.tensor(js['encoderLogits'])
js_enc_attention = torch.tensor(js['encoderAttention'])
js_gpt_logits = torch.tensor(js['gptLogits'])
js_gpt_attention = torch.tensor(js['gptAttention'])

enc_logit_diff = (enc_logits - js_enc_logits).abs().max().item()
enc_attention_diff = (enc_attention[:, :len(encoder_tokens), :len(encoder_tokens)] - js_enc_attention).abs().max().item()
gpt_logit_diff = (gpt_logits - js_gpt_logits).abs().max().item()
gpt_attention_diff = (gpt_attention - js_gpt_attention).abs().max().item()

assert enc_logit_diff < 2e-5
assert enc_attention_diff < 2e-6
assert gpt_logit_diff < 2e-5
assert gpt_attention_diff < 2e-6
assert torch.triu(js_gpt_attention, diagonal=1).abs().max().item() == 0

print(f'OK: paridad encoder logits, diferencia máxima={enc_logit_diff:.3e}')
print(f'OK: paridad encoder atención, diferencia máxima={enc_attention_diff:.3e}')
print(f'OK: paridad GPT logits, diferencia máxima={gpt_logit_diff:.3e}')
print(f'OK: paridad GPT atención, diferencia máxima={gpt_attention_diff:.3e}')
print('OK: máscara causal estricta, atención futura máxima=0.000e+00')
