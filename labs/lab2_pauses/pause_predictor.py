"""Pause predictor — skeleton for lab 2.

Run as a script to score the predictor on the prepared data::

    python pause_predictor.py

Precision, recall and F1 are computed for `is_pause_after`, and MAE for `pause_duration`
on true positives only. The last word of every utterance is excluded.

rubert-tiny2 + BiLSTM + CRF.
POS-фичи через spaCy: текущий токен + соседи + расстояния до пунктуации.
Пост-процессинг: . ! ? (жёстко) + - (только при сомнении модели).
"""
import csv
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF
from sklearn.metrics import f1_score, precision_score, recall_score, mean_absolute_error
from tqdm import tqdm

PAUSE_PREDICTOR_DATA = 'data/RUSLAN_pause_metadata.csv'
MODEL_PATH = 'models/pause_model.pt'

RUBERT_NAME = 'cointegrated/rubert-tiny2'
RUBERT_DIM = 312

PUNCT_DIM = 8
EXTRA_DIM = 6
POS_CATS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET',
            'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN',
            'PUNCT', 'SCONJ', 'VERB', 'X']
POS_DIM = len(POS_CATS)
DIST_DIM = 2
SENT_LEN_DIM = 1
FEAT_DIM = PUNCT_DIM + EXTRA_DIM + 3 * POS_DIM + DIST_DIM + SENT_LEN_DIM

HIDDEN = 128
BATCH_SIZE, EPOCHS = 16, 4
LR_HEAD = 1e-3
LR_RUBERT = 2e-5
WEIGHT_DECAY = 1e-4
VAL_FRACTION = 0.1
DUR_LOSS_WEIGHT = 0.5
DROPOUT = 0.3
MAX_LEN = 128
SEED = 42
FORCE_RETRAIN = True

LOG_EPS = 1e-3
DUR_MIN, DUR_MAX = 0.03, 1.5

POS_WEIGHT = 3.2

FORCE_PUNCT_PAUSE = True
FORCE_PUNCT_DUR = 0.15
FORCE_DASH_DUR = 0.12      
DASH_PROB_LO = 0.3        
DASH_PROB_HI = 0.5

PUNCT_LIST = [',', '.', '—', '-', '?', '!', ';', ':']
PUNCT_INDEX = {p: i for i, p in enumerate(PUNCT_LIST)}

CONJ_WORDS = {'и', 'а', 'но', 'или', 'либо', 'то'}
PART_WORDS = {'же', 'ли', 'бы', 'не', 'ни', 'вот', 'вон', 'ведь', 'уж'}


#spaCy
_NLP = None
_POS_CACHE = {}


def _init_nlp():
    global _NLP
    if _NLP is not None:
        return _NLP
    try:
        import spacy
        _NLP = spacy.load('ru_core_news_sm',
                          disable=['parser', 'ner', 'lemmatizer'])
        print('[spaCy] ru_core_news_sm loaded (parser/ner/lemmatizer disabled)')
    except Exception as e:
        print(f'[spaCy] НЕ загружен ({e}). POS-фичи будут пустыми.')
        _NLP = False
    return _NLP


def get_pos_tags(words):
    if not words:
        return []
    key = ' '.join(words)
    if key in _POS_CACHE:
        return _POS_CACHE[key]
    nlp = _init_nlp()
    if not nlp:
        _POS_CACHE[key] = [''] * len(words)
        return _POS_CACHE[key]
    try:
        doc = nlp(key)
        if len(doc) != len(words):
            tags = [''] * len(words)
        else:
            tags = [t.pos_ for t in doc]
    except Exception:
        tags = [''] * len(words)
    _POS_CACHE[key] = tags
    return tags


# Признаки
def split_word_punct(label_raw: str) -> tuple[str, np.ndarray]:
    s = str(label_raw).strip()
    s = re.sub(r'[«»""]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()

    punct = np.zeros(PUNCT_DIM, dtype=np.float32)
    m = re.search(r'([.,!?;:—–\-]+)\s*$', s)
    if m:
        for ch in m.group(1):
            if ch in PUNCT_INDEX:
                punct[PUNCT_INDEX[ch]] = 1.0
            elif ch == '–':
                punct[PUNCT_INDEX['-']] = 1.0
        word = s[:m.start()].strip()
    else:
        word = s
    return word.lower(), punct


def first_subword_ids(tokenizer, words: list[str]) -> list[int]:
    if len(words) == 0:
        return []
    enc = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
    first = {}
    for k, wid in enumerate(enc.word_ids()):
        if wid is not None and wid not in first:
            first[wid] = enc['input_ids'][k]
    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0
    return [first.get(j, unk_id) for j in range(len(words))]


def build_features(words: list[str], puncts: list[np.ndarray]) -> np.ndarray:
    """65 фич: punct(8) + base(6) + POS×3(48) + dist(2) + sent_len(1)."""
    n = len(words)
    feats = np.zeros((n, FEAT_DIM), dtype=np.float32)

    pos_tags = get_pos_tags(words)

    dist_prev = [n] * n
    last_p = -1
    for i in range(n):
        if puncts[i].any():
            last_p = i
        dist_prev[i] = i - last_p if last_p >= 0 else n

    dist_next = [n] * n
    next_p = n
    for i in range(n - 1, -1, -1):
        if puncts[i].any():
            next_p = i
        dist_next[i] = next_p - i

    sent_len_norm = min(n, 40) / 40.0

    prev_punct_any = 0.0
    for i in range(n):
        w = words[i]
        feats[i, :PUNCT_DIM] = puncts[i]

        feats[i, 8 + 0] = i / max(n - 1, 1)
        feats[i, 8 + 1] = min(len(w), 20) / 20.0
        feats[i, 8 + 2] = 1.0 if w in CONJ_WORDS else 0.0
        feats[i, 8 + 3] = 1.0 if w in PART_WORDS else 0.0
        feats[i, 8 + 4] = 1.0 if i == n - 1 else 0.0
        feats[i, 8 + 5] = prev_punct_any

        cur_pos = pos_tags[i] if i < len(pos_tags) else ''
        if cur_pos in POS_CATS:
            feats[i, 14 + POS_CATS.index(cur_pos)] = 1.0

        prev_pos = pos_tags[i - 1] if i > 0 and i - 1 < len(pos_tags) else ''
        if prev_pos in POS_CATS:
            feats[i, 30 + POS_CATS.index(prev_pos)] = 1.0

        next_pos = pos_tags[i + 1] if i + 1 < n and i + 1 < len(pos_tags) else ''
        if next_pos in POS_CATS:
            feats[i, 46 + POS_CATS.index(next_pos)] = 1.0

        feats[i, 62] = min(dist_prev[i], 10) / 10.0
        feats[i, 63] = min(dist_next[i], 10) / 10.0
        feats[i, 64] = sent_len_norm

        prev_punct_any = float(puncts[i].any())
    return feats

class PauseDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.samples = []
        for _, g in df.groupby('id', sort=False):
            raws = g['label_raw'].tolist()
            words, puncts = [], []
            for r in raws:
                w, p = split_word_punct(r)
                words.append(w)
                puncts.append(p)
            feats = build_features(words, puncts)
            pause = g['is_pause_after'].values.astype(int)
            dur_raw = g['pause_duration'].values.astype(float)
            dur_log = np.log(dur_raw + LOG_EPS).astype(np.float32)
            self.samples.append((words, feats, pause, dur_log, dur_raw))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def make_collate(tokenizer):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def collate(batch):
        words_b, feats_b, pause_b, dlog_b, draw_b = zip(*batch)
        max_len = min(max(len(x) for x in words_b), MAX_LEN)
        B = len(words_b)

        ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        feat_t = torch.zeros(B, max_len, FEAT_DIM, dtype=torch.float32)
        mask = torch.zeros(B, max_len, dtype=torch.bool)
        tags = torch.zeros(B, max_len, dtype=torch.long)
        durations_log = torch.zeros(B, max_len, dtype=torch.float32)
        durations_raw = torch.zeros(B, max_len, dtype=torch.float32)

        for i in range(B):
            w = words_b[i][:max_len]
            f = feats_b[i][:max_len]
            p = pause_b[i][:max_len]
            dl = dlog_b[i][:max_len]
            dr = draw_b[i][:max_len]
            n = len(w)
            sub_ids = first_subword_ids(tokenizer, w)
            ids[i, :n] = torch.tensor(sub_ids, dtype=torch.long)
            feat_t[i, :n] = torch.tensor(f)
            mask[i, :n] = True
            tags[i, :n] = torch.tensor(p, dtype=torch.long)
            durations_log[i, :n] = torch.tensor(dl)
            durations_raw[i, :n] = torch.tensor(dr)

        return ids, feat_t, mask, tags, durations_log, durations_raw
    return collate


class _PauseNet(nn.Module):
    """CRF + регрессия."""
    def __init__(self, rubert):
        super().__init__()
        self.rubert = rubert
        self.lstm = nn.LSTM(RUBERT_DIM + FEAT_DIM, HIDDEN,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(HIDDEN * 2, 2)
        self.crf = CRF(2, batch_first=True)
        self.regressor = nn.Linear(HIDDEN * 2, 1)

    def forward(self, x_ids, x_feats, x_mask):
        out = self.rubert(input_ids=x_ids, attention_mask=x_mask).last_hidden_state
        inp = torch.cat([out, x_feats], dim=-1)
        lstm_out, _ = self.lstm(inp)
        lstm_out = self.dropout(lstm_out)
        emissions = self.classifier(lstm_out)
        return emissions, lstm_out


class PausePredictor:
    def __init__(self, model_path: str = MODEL_PATH):
        torch.manual_seed(SEED)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        if os.path.exists(model_path) and not FORCE_RETRAIN:
            self.load()

    def fit(self, train_df, val_df=None):
        from transformers import AutoTokenizer, AutoModel

        _init_nlp()

        self.tokenizer = AutoTokenizer.from_pretrained(RUBERT_NAME)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or '[PAD]'
        rubert = AutoModel.from_pretrained(RUBERT_NAME)
        self.model = _PauseNet(rubert).to(self.device)

        print('[fit] build train dataset...')
        ds = PauseDataset(train_df)
        collate = make_collate(self.tokenizer)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

        head_params = [p for n, p in self.model.named_parameters()
                       if not n.startswith('rubert.')]
        rubert_params = [p for n, p in self.model.named_parameters()
                         if n.startswith('rubert.')]
        opt = torch.optim.AdamW([
            {'params': head_params, 'lr': LR_HEAD},
            {'params': rubert_params, 'lr': LR_RUBERT},
        ], weight_decay=WEIGHT_DECAY)

        best_f1, best_state = -1.0, None
        for epoch in range(EPOCHS):
            self.model.train()
            total = 0.0
            for batch in loader:
                ids, feats, mask, tags, dlog, draw = batch
                ids = ids.to(self.device)
                feats = feats.to(self.device)
                mask = mask.to(self.device)
                tags = tags.to(self.device)
                dlog = dlog.to(self.device)
                draw = draw.to(self.device)

                opt.zero_grad()
                emissions, out = self.model(ids, feats, mask)
                loss = self._loss(emissions, out, tags, mask, dlog, draw)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
                total += loss.item()

            msg = f'epoch {epoch+1}/{EPOCHS}, loss {total/len(loader):.4f}'
            if val_df is not None:
                val_out = _run_predictor(self, val_df, f'val ep{epoch+1}', verbose=False)
                val_out = val_out[val_out.is_last_word == 0]
                f1 = f1_score(val_out.is_pause_after, val_out.is_pause_hat, zero_division=0)
                msg += f', val_f1 {f1:.4f}'
                if f1 > best_f1:
                    best_f1 = f1
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            print(msg)
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f'best val_f1: {best_f1:.4f}')

    def _loss(self, emissions, out, tags, mask, dlog, draw):
        """CRF-лосс с масштабированием emissions под дисбаланс классов."""
        emissions_w = emissions.clone()
        emissions_w[..., 1] = emissions_w[..., 1] * POS_WEIGHT
        crf_loss = -self.model.crf(emissions_w, tags, mask=mask, reduction='mean')

        pause_mask = (tags == 1) & mask
        if pause_mask.any():
            pred_log = self.model.regressor(out).squeeze(-1)
            dur_loss = F.l1_loss(pred_log[pause_mask], dlog[pause_mask])
        else:
            dur_loss = torch.tensor(0.0, device=emissions.device)
        return crf_loss + DUR_LOSS_WEIGHT * dur_loss

    @torch.no_grad()
    def predict(self, tokens):
        if self.model is None:
            raise RuntimeError('Model not fitted or loaded')
        n = len(tokens)
        if n == 0:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=float)

        words, puncts = [], []
        for raw in tokens:
            w, p = split_word_punct(raw)
            words.append(w)
            puncts.append(p)
        feats = build_features(words, puncts)
        n_eff = min(n, MAX_LEN)

        sub_ids = first_subword_ids(self.tokenizer, words[:n_eff])
        ids = torch.tensor([sub_ids], dtype=torch.long, device=self.device)
        feat_t = torch.tensor(feats[:n_eff][None, :, :], dtype=torch.float32, device=self.device)
        mask = torch.ones_like(ids, dtype=torch.bool, device=self.device)

        self.model.eval()
        emissions, out = self.model(ids, feat_t, mask)
        tags = self.model.crf.decode(emissions, mask=mask)[0]
        probs = F.softmax(emissions, dim=-1)[0, :, 1].cpu().numpy()
        pred_log = self.model.regressor(out).squeeze(-1)[0].cpu().numpy()

        is_pause = np.array(tags, dtype=int)[:n_eff]
        durations = np.exp(pred_log[:n_eff]) - LOG_EPS
        durations = np.where(is_pause == 1, np.clip(durations, DUR_MIN, DUR_MAX), 0.0)

        # Пост-процессинг
        if FORCE_PUNCT_PAUSE:
            for i in range(n_eff):
                s = str(tokens[i]).rstrip()
                last_char = s[-1:] if s else ''
                # .!?
                if last_char in '!?.':
                    if is_pause[i] == 0:
                        is_pause[i] = 1
                        durations[i] = FORCE_PUNCT_DUR
                # если модель сомневается
                elif last_char == '-' and DASH_PROB_LO < probs[i] < DASH_PROB_HI:
                    if is_pause[i] == 0:
                        is_pause[i] = 1
                        durations[i] = FORCE_DASH_DUR

        is_pause = (durations >= DUR_MIN).astype(int)
        durations = np.where(is_pause == 1, durations, 0.0)

        if n > n_eff:
            is_pause = np.concatenate([is_pause, np.zeros(n - n_eff, dtype=int)])
            durations = np.concatenate([durations, np.zeros(n - n_eff, dtype=float)])

        return is_pause, durations

    def predict_durations(self, tokens):
        def expand_token(t, p):
            return [t, '<SIL>'] if p else [t]

        def expand_dur(d):
            return [-1.0, float(d)] if d > 0.0 else [-1.0]

        is_pause, durations = self.predict(tokens)
        tokens_w_pauses = np.concatenate([expand_token(t, p) for t, p in zip(tokens, is_pause)])
        durations_w_pauses = np.concatenate([expand_dur(d) for d in durations]).astype(np.float32)
        return tokens_w_pauses, durations_w_pauses

    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save({'state_dict': self.model.state_dict()}, self.model_path)
        print(f'saved to {self.model_path}')

    def load(self):
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(RUBERT_NAME)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or '[PAD]'
        rubert = AutoModel.from_pretrained(RUBERT_NAME)
        self.model = _PauseNet(rubert).to(self.device)
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()


def calc_metrics(df):
    rec = recall_score(df.is_pause_after, df.is_pause_hat, zero_division=0)
    prc = precision_score(df.is_pause_after, df.is_pause_hat, zero_division=0)
    f1 = f1_score(df.is_pause_after, df.is_pause_hat, zero_division=0)
    tp = df[(df.is_pause_after == 1) & (df.is_pause_hat == 1)]
    mae = mean_absolute_error(tp.pause_duration, tp.pause_duration_hat) if len(tp) else float('nan')
    print(f'PRC: {prc:.4f}, REC: {rec:.4f}, F1: {f1:.4f}; MAE: {mae:.4f}; TP: {len(tp)}')


def _run_predictor(pp, part, name, verbose=True):
    part = part.copy()
    is_pause_hat, pause_dur_hat = [], []
    iterator = part.groupby('id', sort=False)
    if verbose:
        iterator = tqdm(iterator, desc=name)
    for sid, g in iterator:
        is_p, dur_p = pp.predict(g['label_raw'].values)
        is_pause_hat += list(is_p)
        pause_dur_hat += list(dur_p)
    part['is_pause_hat'] = is_pause_hat
    part['pause_duration_hat'] = pause_dur_hat
    return part


def test_pause_predictor():
    df = pd.read_csv(PAUSE_PREDICTOR_DATA, sep='|', quoting=csv.QUOTE_NONE)
    train_df = df[df['set'] == 'train'].copy()
    ids = train_df['id'].unique()
    rng = np.random.default_rng(SEED)
    val_ids = set(rng.choice(ids, size=int(len(ids) * VAL_FRACTION), replace=False))
    fit_df = train_df[~train_df['id'].isin(val_ids)]
    val_df = train_df[train_df['id'].isin(val_ids)]

    pp = PausePredictor()
    if pp.model is None:
        pp.fit(fit_df, val_df=val_df)
        pp.save()
    else:
        print(f'loaded model from {pp.model_path}')

    for name, part in [('train', fit_df), ('val', val_df), ('test', df[df['set'] == 'test'])]:
        out = _run_predictor(pp, part, name)
        print(f'\n=== {name} ===')
        calc_metrics(out[out.is_last_word == 0])


if __name__ == '__main__':
    test_pause_predictor()