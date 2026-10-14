"""Grapheme-to-phoneme front end — lab 3."""
from __future__ import annotations

import csv
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import tqdm

TOKENS_PHONES_DATA = 'data/tokens_phones.csv'
LEXICON_PATH = 'data/lexicon.csv'
MFA_DICT_PATH = 'data/russian_mfa_norm.dict'

SIL = '<SIL>'


class G2P:
    """Turns a sentence into a phone sequence."""

    def __init__(self,
                 lexicon: Optional[Dict[str, Counter]] = None,
                 mfa_dict: Optional[Dict[str, str]] = None) -> None:
        self.lexicon = lexicon or {}
        self.mfa_dict = mfa_dict or {}

    def _transcribe_token(self, token: str) -> List[str]:
        if token == SIL:
            return [SIL]
        if token in self.lexicon:
            best = self.lexicon[token].most_common(1)[0][0]
            return best.split()
        if token in self.mfa_dict:
            return self.mfa_dict[token].split()
        return []

    def transcribe(self, tokens: Union[List[str], np.ndarray]) -> List[List[str]]:
        return [self._transcribe_token(t) for t in tokens]


def phone_error_rate(reference: List[List[str]],
                     hypothesis: List[List[str]]) -> float:
    total_distance = 0
    total_length = 0
    for ref, hyp in zip(reference, hypothesis):
        total_length += len(ref)
        if not hyp:
            total_distance += len(ref)
            continue
        previous = list(range(len(hyp) + 1))
        for i, ref_phone in enumerate(ref, start=1):
            current = [i] + [0] * len(hyp)
            for j, hyp_phone in enumerate(hyp, start=1):
                current[j] = min(previous[j] + 1,
                                 current[j - 1] + 1,
                                 previous[j - 1] + (ref_phone != hyp_phone))
            previous = current
        total_distance += previous[-1]
    return total_distance / total_length if total_length else float('nan')


def word_accuracy(reference: List[List[str]],
                  hypothesis: List[List[str]]) -> float:
    if not reference:
        return float('nan')
    return float(np.mean([ref == hyp for ref, hyp in zip(reference, hypothesis)]))


def load_tokens(path: str = TOKENS_PHONES_DATA) -> pd.DataFrame:
    tokens = pd.read_csv(path, sep='|', quoting=csv.QUOTE_NONE)
    tokens['token'] = tokens.token.astype(str).map(
        lambda s: unicodedata.normalize('NFC', s))
    tokens['phones'] = tokens.phones.fillna('').astype(str)
    unpronounced = (tokens.phones == '') | tokens.phones.str.split().map(
        lambda p: 'spn' in p)
    print(f'dropping {unpronounced.sum()} tokens without a transcription '
          f'({unpronounced.mean():.2%})')
    return tokens[~unpronounced].reset_index(drop=True)


def build_lexicon(tokens: pd.DataFrame) -> Dict[str, Counter]:
    lexicon: Dict[str, Counter] = defaultdict(Counter)
    for token, phones in tokens[['token', 'phones']].values:
        lexicon[token][phones] += 1
    return lexicon


def load_mfa_dict(path: str = MFA_DICT_PATH) -> Dict[str, str]:
    out = {}
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) != 2:
                continue
            word, phones = parts
            if phones.strip() == 'spn':
                continue
            out[word] = phones
    return out


def calc_metrics(g2p: G2P, tokens: pd.DataFrame,
                 lexicon: Dict[str, Counter]) -> None:
    sentence_ref: List[List[str]] = []
    sentence_hyp: List[List[str]] = []
    oov_ref: List[List[str]] = []
    oov_hyp: List[List[str]] = []
    in_vocab_ref: List[List[str]] = []
    in_vocab_hyp: List[List[str]] = []

    for _, sentence in tqdm.tqdm(tokens.groupby('id', sort=False)):
        sentence = sentence.sort_values('token_id')
        words = sentence.token.values
        predicted = g2p.transcribe(words)
        if len(predicted) != len(words):
            raise ValueError(f'transcribe returned {len(predicted)} outputs '
                             f'for {len(words)} tokens')
        reference = [p.split() for p in sentence.phones.values]

        sentence_ref.append([p for word in reference for p in word if p != SIL])
        sentence_hyp.append([p for word in predicted for p in word if p != SIL])
        for token, ref, hyp in zip(words, reference, predicted):
            if token in lexicon:
                in_vocab_ref.append(ref)
                in_vocab_hyp.append(hyp)
            else:
                oov_ref.append(ref)
                oov_hyp.append(hyp)

    n_words = len(in_vocab_ref) + len(oov_ref)
    print(f'running text  PER {phone_error_rate(sentence_ref, sentence_hyp):.4f}')
    print(f'in-vocabulary PER {phone_error_rate(in_vocab_ref, in_vocab_hyp):.4f}  '
          f'({len(in_vocab_ref)} words, {len(in_vocab_ref) / n_words:.1%})')
    print(f'OOV           PER {phone_error_rate(oov_ref, oov_hyp):.4f}  '
          f'word accuracy {word_accuracy(oov_ref, oov_hyp):.4f}  '
          f'({len(oov_ref)} words, {len(oov_ref) / n_words:.1%})')


def test_g2p() -> None:
    tokens = load_tokens()
    train = tokens[tokens.set == 'train']
    test = tokens[tokens.set == 'test']
    print(f'train {len(train)} tokens / test {len(test)} tokens')

    lexicon = build_lexicon(train)
    mfa = load_mfa_dict()
    print(f'lexicon: {len(lexicon)} types, '
          f'{sum(1 for c in lexicon.values() if len(c) > 1)} with several pronunciations')
    print(f'MFA dict: {len(mfa)} words')

    g2p = G2P(lexicon, mfa)

    print('\nCalculate metrics, training fold:')
    calc_metrics(g2p, train, lexicon)

    print('\nCalculate metrics, testing fold:')
    calc_metrics(g2p, test, lexicon)


if __name__ == '__main__':
    test_g2p()