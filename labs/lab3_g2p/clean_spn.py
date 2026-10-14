"""Remove every utterance that contains an `spn` phone from lab 3 artifacts.

Reads `data/tokens_phones.csv` and `data/tokens_phones_sil.csv`, finds all `id`s
whose phones field contains `spn`, drops every row of those ids, and rewrites the
files in place (keeping a `.bak` copy). Also rebuilds `data/lexicon.csv` from the
cleaned training rows.
"""
import csv
import os
import shutil
from collections import Counter, defaultdict

import pandas as pd

TOKENS = 'data/tokens_phones.csv'
TOKENS_SIL = 'data/tokens_phones_sil.csv'
LEXICON = 'data/lexicon.csv'


def find_bad_ids(path: str) -> set[str]:
    df = pd.read_csv(path, sep='|', quoting=csv.QUOTE_NONE)
    df['phones'] = df.phones.fillna('').astype(str)
    bad = df[df.phones.str.contains('spn', regex=False)]
    return set(bad['id']), df


def clean(path: str, bad_ids: set[str]) -> None:
    shutil.copy(path, path + '.bak')
    df = pd.read_csv(path, sep='|', quoting=csv.QUOTE_NONE)
    before = len(df)
    df = df[~df.id.isin(bad_ids)].reset_index(drop=True)
    df.to_csv(path, sep='|', quoting=csv.QUOTE_NONE, index=False, encoding='utf-8')
    print(f'{path}: {before} -> {len(df)} rows  '
          f'(dropped {before - len(df)})')


def rebuild_lexicon(path: str) -> None:
    df = pd.read_csv(path, sep='|', quoting=csv.QUOTE_NONE)
    df['phones'] = df.phones.fillna('').astype(str)
    df = df[(df.set == 'train') & (~df.phones.str.contains('spn', regex=False))]
    lexicon: dict[str, Counter] = defaultdict(Counter)
    for token, phones in df[['token', 'phones']].values:
        if not phones:
            continue
        lexicon[token][phones] += 1
    with open(LEXICON, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.writer(fh, delimiter='|', quoting=csv.QUOTE_NONE)
        writer.writerow(['word', 'phones', 'count'])
        for word in sorted(lexicon):
            for phones, count in lexicon[word].most_common():
                writer.writerow([word, phones, count])
    print(f'{LEXICON}: {len(lexicon)} word types')


def main():
    bad_ids, _ = find_bad_ids(TOKENS)
    print(f'bad utterances (contain spn): {len(bad_ids)}')
    clean(TOKENS, bad_ids)
    clean(TOKENS_SIL, bad_ids)
    rebuild_lexicon(TOKENS)


if __name__ == '__main__':
    main()