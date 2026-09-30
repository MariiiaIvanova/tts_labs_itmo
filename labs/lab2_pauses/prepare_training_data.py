"""Build pause predictor training data — lab 2.

Joins the lab 1 normalized metadata with the MFA word alignment and writes one row per
word to `data/RUSLAN_pause_metadata.csv`::

    id|label|label_raw|duration|is_last_word|is_pause_after|pause_duration|set

Utterances whose id ends in 0 or 5 go to `test`, the rest to `train`.

Run from the lab directory::

    python prepare_training_data.py
"""
import csv
import glob
import numpy as np
import os
import pandas as pd
from praatio import textgrid
import tqdm

RUSLAN_META = '../../data/metadata_RUSLAN_22200_normalized.csv'
ALIGN_DIR = '../../data/RUSLAN_align_v2/'
RESULT_PATH = 'data/RUSLAN_pause_metadata.csv'

# Параметры фильтрации шума
PAUSE_MIN_DUR = 0.03        # паузы до 30 мс
MIN_WORDS_PER_UTT = 3      # предложения короче - выкидываем
MAX_PAUSE_DENSITY = 0.4    # доля пауз > 40%


def read_text_grids(ruslan: pd.DataFrame, align_root: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Read the MFA TextGrid of every utterance in `ruslan`.

    Args:
        ruslan: Metadata with `id` and `nrm` columns.
        align_root: Directory with `{id}.TextGrid` files.

    Returns:
        Word intervals (`label`, `duration`, `id`; silence has label ``""``), phone
        intervals (same columns; silence is ``"<SIL>"``), and one space-joined phone
        string per metadata row (``""`` when the TextGrid is missing).
    """
    word_docs = []
    phn_docs = []
    phoneme_sequences = []
    for wave_id, text in tqdm.tqdm(ruslan[['id', 'nrm']].values):
        try:
            tg = textgrid.openTextgrid(os.path.join(align_root, wave_id + '.TextGrid'), True)
        except:
            phoneme_sequences.append('')
            continue
        words, phones = tg.tiers

        words = words.entries
        for i in words:
            word_docs.append({'label': i.label, 'duration': i.end - i.start, 'id': wave_id})

        phoneme_sequences.append(' '.join([p.label for p in phones.entries]))
        phones = phones.entries
        for i in phones:
            if i.label == '':
                phn_docs.append({'label': '<SIL>', 'duration': i.end - i.start, 'id': wave_id})
            else:
                phn_docs.append({'label': i.label, 'duration': i.end - i.start, 'id': wave_id})
    word_df = pd.DataFrame(word_docs)
    phn_df = pd.DataFrame(phn_docs)
    return word_df, phn_df, phoneme_sequences


def align_text_and_textgrid(tokens: pd.DataFrame, text: str) -> pd.DataFrame:
    """Attach the original text form of each aligned word as `label_raw`."""
    raw_tokens = []
    text_lower = text.lower()
    previous_word = -1
    for t, d, i in tokens[['label', 'duration', 'id']].values:
        if t == '':
            raw_tokens.append('<SIL>')
            continue
        splits = text_lower.split(t, maxsplit=1)
        if len(splits) == 1:
            print(f'Error aligning f{i}!')
            print(t, text, tokens)
            return tokens
        if previous_word == -1:
            raw_tokens.append(text[:len(splits[0] + t)].strip())
        else:
            raw_tokens[previous_word] += text[:len(splits[0])].strip()
            raw_tokens.append(text[len(splits[0]):len(splits[0] + t)])
        text_lower = splits[1]
        text = text[len(splits[0] + t):]
        previous_word = len(raw_tokens) - 1
    if len(text) and (previous_word >= 0):
        raw_tokens[previous_word] += text.strip()
    tokens['label_raw'] = raw_tokens
    return tokens


def add_pause_labels(align: pd.DataFrame) -> pd.DataFrame:
    """Mark which words are followed by a pause, and for how long.

    Пауза засчитывается только если её длительность >= PAUSE_MIN_DUR.
    Это убирает микро-задержки MFA на стыках слов.
    """
    is_last_word = []
    pause_after = []
    pause_duration = []
    last_word = -1
    for idx, (label, dur) in enumerate(align[['label', 'duration']].values):
        if label == '':
            # длительность SIL
            if last_word >= 0 and dur >= PAUSE_MIN_DUR:
                pause_after[last_word] = True
                pause_duration[last_word] = dur
            pause_after.append(False)
            pause_duration.append(0.)
            is_last_word.append(False)
        else:
            pause_after.append(False)
            pause_duration.append(0.)
            is_last_word.append(False)
            last_word = idx
    if last_word >= 0:
        is_last_word[last_word] = True
    align['is_last_word'] = is_last_word
    align['is_pause_after'] = pause_after
    align['pause_duration'] = pause_duration
    return align


# фильтрация шумных предложений
def filter_noisy_utterances(df: pd.DataFrame) -> pd.DataFrame:
    """Убирает предложения, которые портят обучение.

    Критерии:
      - меньше MIN_WORDS_PER_UTT слов (нечего учить);
      - доля пауз > MAX_PAUSE_DENSITY;

    Args:
        df: DataFrame с колонками `id`, `is_pause_after`, `is_last_word`.

    Returns:
        Отфильтрованный DataFrame.
    """
    keep_ids = []
    stats = {'too_short': 0, 'too_dense': 0, 'kept': 0}
    for sid, g in df.groupby('id', sort=False):
        # исключаем последнее слово
        g_eff = g[g.is_last_word == 0]
        n = len(g_eff)
        if n < MIN_WORDS_PER_UTT:
            stats['too_short'] += 1
            continue
        density = g_eff.is_pause_after.mean()
        if density > MAX_PAUSE_DENSITY:
            stats['too_dense'] += 1
            continue

        keep_ids.append(sid)
        stats['kept'] += 1
    print(f'\nФильтрация предложений:')
    print(f'  too_short  (<{MIN_WORDS_PER_UTT} слов): {stats["too_short"]}')
    print(f'  too_dense  (>{MAX_PAUSE_DENSITY} пауз):  {stats["too_dense"]}')
    print(f'  kept:                             {stats["kept"]}')
    return df[df['id'].isin(keep_ids)].copy()


def main() -> None:
    """Read metadata and alignments, label pauses, filter noise, split and save."""
    ruslan = pd.read_csv(f'{RUSLAN_META}', sep='|', names=['id', 'raw', 'nrm'], quoting=csv.QUOTE_NONE)
    word_df, _, _ = read_text_grids(ruslan, ALIGN_DIR)

    # Дополнительная нормализация
    word_df.label = word_df.label.str.replace('‐', '-')
    word_df.label = word_df.label.str.replace('‑', '-')
    ruslan.nrm = ruslan.nrm.str.replace('‐', '-')
    ruslan.nrm = ruslan.nrm.str.replace('‑', '-')
    ruslan.nrm = ruslan.nrm.str.replace('’', "'")
    ruslan.nrm = ruslan.nrm.str.replace('\\((.*?)\\)', '[bracketed]', regex=True)
    ruslan.nrm = ruslan.nrm.str.replace('\\<(.*?)\\>', '[bracketed]', regex=True)

    # Выравнивание TextGrid-токенов и текста
    aligns = []
    for n, i in tqdm.tqdm(ruslan[['nrm', 'id']].values):
        tokens = word_df[word_df.id == i]
        aligns.append(align_text_and_textgrid(tokens, n))

    # Метки пауз
    aligns = [add_pause_labels(a) for a in aligns]

    # Сборка DataFrame
    pause_df = pd.concat(aligns)

    pause_df = pause_df[pause_df.label_raw != '<SIL>']
    pause_df = pause_df[pause_df.label_raw.notna()]
    pause_df = pause_df.reset_index(drop=True)

    pause_df.is_last_word = pause_df.is_last_word.astype(int)
    pause_df.is_pause_after = pause_df.is_pause_after.astype(int)

    # Фильтрация шума
    print(f'\nДо фильтрации: {pause_df["id"].nunique()} предложений, {len(pause_df)} токенов')
    pause_df = filter_noisy_utterances(pause_df)
    print(f'После фильтрации: {pause_df["id"].nunique()} предложений, {len(pause_df)} токенов')

    # Разбиение train/test по id % 5 == 0
    pause_df['set'] = 'train'
    pause_df.loc[pause_df.id.str.split('_', expand=True)[0].astype(int) % 5 == 0, 'set'] = 'test'

    # Отчёт по доле пауз
    print(f'\nИтоговая статистика:')
    train = pause_df[pause_df.set == 'train']
    test = pause_df[pause_df.set == 'test']
    print(f'  train: {len(train)} токенов, доля пауз: {train.is_pause_after.mean():.4f}')
    print(f'  test:  {len(test)} токенов, доля пауз: {test.is_pause_after.mean():.4f}')

    pause_df.to_csv(f'{RESULT_PATH}', sep='|', index=False, header=True, quoting=csv.QUOTE_NONE)
    print(f'\nSaved to {RESULT_PATH}')


if __name__ == '__main__':
    main()