"""Подготовка метаданных для лабораторной работы №4 (акустическая модель).

Читает `data/tokens_phones_sil.csv` (токены + <SIL>) и сохраняет два файла:

- `data/metadata_phones_gt.csv`   — эталонная разметка;
- `data/metadata_phones.csv`      — предсказанная через G2P.

Формат: `id|text|phones`
  - `text` — нормализованные токены через пробел (без <SIL>);
  - `phones` — фонемы через пробел, с токеном <SIL> на месте пауз;
  - все фонемы ∈ common/phones_mfa.txt.

Запуск из директории labs/lab3_g2p:

    python prepare_metadata.py
"""
import csv
import unicodedata

import pandas as pd
import tqdm

from g2p import G2P, build_lexicon, load_mfa_dict

TOKENS_SIL_PATH = 'data/tokens_phones_sil.csv'
TOKENS_PLAIN_PATH = 'data/tokens_phones.csv'
METADATA_GT_PATH = 'data/metadata_phones_gt.csv'
METADATA_PRED_PATH = 'data/metadata_phones.csv'
PHONES_MFA_PATH = '../../common/phones_mfa.txt'

SIL = '<SIL>'


def load_target_phones(path: str) -> set[str]:
    """Читает целевой список фонем (56 штук)."""
    with open(path, encoding='utf-8') as fh:
        return {unicodedata.normalize('NFC', line.strip())
                for line in fh if line.strip()}


def load_tokens_sil(path: str) -> pd.DataFrame:
    """Читает tokens_phones_sil.csv.

    - `token` = '' (пауза) преобразуется в NaN → далее везде трактуется как <SIL>
      через `phones == '<SIL>'`.
    - `phones` = '<SIL>' — маркер паузы.
    """
    df = pd.read_csv(path, sep='|', quoting=csv.QUOTE_NONE)
    df['token'] = df.token.fillna('').astype(str)
    df['phones'] = df.phones.fillna('').astype(str)
    df['id'] = df['id'].astype(str)
    df['set'] = df['set'].astype(str)
    df['token_id'] = df['token_id'].astype(int)

    def fix_token(row):
        tok = row['token']
        if tok == '':
            return SIL   # пауза
        return unicodedata.normalize('NFC', tok)

    df['token'] = df.apply(fix_token, axis=1)
    return df


def build_gt(tokens_sil: pd.DataFrame) -> pd.DataFrame:
    """Эталонная разметка: id|text|phones.

    - text - токены через пробел, БЕЗ <SIL>;
    - phones - фонемы через пробел, С <SIL> на месте пауз.
    """
    rows = []
    for uid, sent in tqdm.tqdm(tokens_sil.groupby('id', sort=False),
                               desc='GT'):
        sent = sent.sort_values('token_id')
        words, phones_field = sent.token.tolist(), sent.phones.tolist()
        text_tokens = [w for w in words if w != SIL]
        phone_seq = []
        for w, ph in zip(words, phones_field):
            if w == SIL or ph == SIL:
                phone_seq.append(SIL)
            else:
                phone_seq.extend(ph.split())
        rows.append({
            'id': uid,
            'text': ' '.join(text_tokens),
            'phones': ' '.join(phone_seq),
        })
    return pd.DataFrame(rows)


def build_pred(tokens_sil: pd.DataFrame, g2p: G2P) -> pd.DataFrame:
    """Предсказанная разметка: прогон всех токенов через G2P, склейка по id.

    <SIL> передаётся в G2P, но G2P возвращает его как есть.
    """
    rows = []
    for uid, sent in tqdm.tqdm(tokens_sil.groupby('id', sort=False),
                               desc='Pred'):
        sent = sent.sort_values('token_id')
        words = sent.token.tolist()
        predicted = g2p.transcribe(words)
        text_tokens = [w for w in words if w != SIL]
        phone_seq = []
        for w, ph_list in zip(words, predicted):
            if w == SIL:
                phone_seq.append(SIL)
            else:
                phone_seq.extend(ph_list)
        rows.append({
            'id': uid,
            'text': ' '.join(text_tokens),
            'phones': ' '.join(phone_seq),
        })
    return pd.DataFrame(rows)


def check_all_phones(df: pd.DataFrame, target: set[str], name: str) -> None:
    """Проверка: все фонемы ∈ target (кроме <SIL>)."""
    seen = set()
    for line in df.phones:
        for p in line.split():
            seen.add(unicodedata.normalize('NFC', p))
    bad = seen - target - {SIL}
    if bad:
        print(f'{name}: фонемы вне target: {sorted(bad)}')
    else:
        print(f'{name}: все фонемы ∈ target')


def main() -> None:
    target = load_target_phones(PHONES_MFA_PATH)
    print(f'target inventory: {len(target)} phones')

    tokens_sil = load_tokens_sil(TOKENS_SIL_PATH)
    print(f'tokens_phones_sil: {len(tokens_sil)} rows, '
          f'{tokens_sil.id.nunique()} sentences')

    # Лексикон строим из tokens_phones.csv (без <SIL>) на train-части
    tokens_plain = pd.read_csv(TOKENS_PLAIN_PATH, sep='|',
                               quoting=csv.QUOTE_NONE)
    tokens_plain['token'] = tokens_plain.token.astype(str)
    tokens_plain['phones'] = tokens_plain.phones.fillna('').astype(str)
    train_plain = tokens_plain[tokens_plain.set == 'train']
    lexicon = build_lexicon(train_plain)
    mfa = load_mfa_dict()
    print(f'lexicon: {len(lexicon)} types, mfa: {len(mfa)} words')

    g2p = G2P(lexicon, mfa)

    gt = build_gt(tokens_sil)
    pred = build_pred(tokens_sil, g2p)

    gt.to_csv(METADATA_GT_PATH, sep='|', index=False,
              quoting=csv.QUOTE_NONE, encoding='utf-8')
    pred.to_csv(METADATA_PRED_PATH, sep='|', index=False,
                quoting=csv.QUOTE_NONE, encoding='utf-8')
    print(f'\nwrote {METADATA_GT_PATH} ({len(gt)} rows)')
    print(f'wrote {METADATA_PRED_PATH} ({len(pred)} rows)')

    check_all_phones(gt, target, 'metadata_phones_gt')
    check_all_phones(pred, target, 'metadata_phones')


if __name__ == '__main__':
    main()