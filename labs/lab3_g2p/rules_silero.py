"""Простые правила графем→фонем для OOV-слов русского языка.

Посимвольное отображение + контекстные правила
(смягчение, йотирование, оглушение, озвончение, «г→в», -ться/-тся).
Ударение расставляет silero-stress,
редукция безударных гласных — по этому ударению.
Все возвращаемые фонемы принадлежат common/phones_mfa.txt.
"""
import re

# Базовое отображение: буква → фонема (согласные)
CONSONANTS = {
    'б': 'b', 'в': 'v', 'г': 'ɡ', 'д': 'd̪', 'ж': 'ʐ', 'з': 'z̪',
    'к': 'k', 'л': 'ɫ', 'м': 'm', 'н': 'n̪', 'п': 'p', 'р': 'r',
    'с': 's̪', 'т': 't̪', 'ф': 'f', 'х': 'x', 'ц': 't̪s̪', 'ч': 'tɕ',
    'ш': 'ʂ', 'щ': 'ɕ', 'й': 'j',
}

# Базовое отображение: буква → фонема (гласные)
VOWELS = {
    'а': 'a', 'о': 'o', 'у': 'u', 'ы': 'ɨ', 'э': 'ɛ', 'и': 'i',
}

# Гласные, которые смягчают предшествующую согласную (е, ё, и, ю, я)
SOFTENING_VOWELS = {'е', 'ё', 'и', 'ю', 'я'}

# Форма гласной после мягкой согласной
SOFT_VOWEL_MAP = {
    'е': 'e', 'ё': 'o', 'ю': 'u', 'я': 'a', 'и': 'i',
}

# Согласные, которые НИКОГДА не смягчаются
ALWAYS_HARD = {'ж', 'ш', 'ц'}

# Согласные, которые ВСЕГДА мягкие
ALWAYS_SOFT = {'ч', 'щ', 'й'}

# Соответствие: твёрдая согласная → мягкая (в целевом наборе — отдельные фонемы)
SOFTEN = {
    'b': 'bʲ', 'v': 'vʲ', 'd̪': 'dʲ', 'z̪': 'zʲ', 's̪': 'sʲ',
    't̪': 'tʲ', 'f': 'fʲ', 'p': 'pʲ', 'm': 'mʲ', 'n̪': 'ɲ',
    'r': 'rʲ', 'ɫ': 'ʎ', 'ɡ': 'ɟ', 'x': 'ç', 'k': 'c',
}

# Оглушение в конце слова: звонкая → глухая (включая мягкие)
FINAL_DEVOICE = {
    'b': 'p', 'v': 'f', 'ɡ': 'k', 'd̪': 't̪', 'ʐ': 'ʂ', 'z̪': 's̪',
    'bʲ': 'pʲ', 'vʲ': 'fʲ', 'dʲ': 'tʲ', 'zʲ': 'sʲ',
}

# Оглушение внутри слова: звонкая перед глухой → глухая
INNER_DEVOICE = FINAL_DEVOICE

# Множества букв-категорий для контекстных правил
VOICED = set('бвгджз')
VOICELESS = set('кпстфхцчшщ')

# Ленивая загрузка модели расстановки ударения
_accentor = None


def _get_accentor():
    """Ленивая загрузка silero-stress."""
    global _accentor
    if _accentor is None:
        from silero_stress import load_accentor
        _accentor = load_accentor()
    return _accentor


def _is_vowel(ch: str) -> bool:
    """Гласная ли буква (обычная или смягчающая)."""
    return ch in VOWELS or ch in SOFTENING_VOWELS


def _soften(phones: list[str]) -> None:
    """Смягчить последнюю согласную (заменить на мягкую пару)."""
    if not phones:
        return
    last = phones[-1]
    if last in SOFTEN:
        phones[-1] = SOFTEN[last]


def _devocalize(phones: list[str], word: str, i: int) -> None:
    """Оглушить последнюю согласную, если следующая буква — глухая."""
    if i + 1 >= len(word):
        return
    nxt = word[i + 1]
    if nxt not in VOICELESS:
        return
    if not phones:
        return
    last = phones[-1]
    if last in INNER_DEVOICE:
        phones[-1] = INNER_DEVOICE[last]


def get_stress_index(word: str, phones: list[str]) -> int | None:
    """Индекс ударной гласной в phones по расстановке silero-stress.

    Возвращает None, если модель недоступна или не справилась.
    """
    try:
        accentor = _get_accentor()
        stressed = accentor(word)
    except Exception as exc:
        print(f'silero-stress failed on "{word}": {exc}')
        return None

    # Считаем номер ударной гласной по буквам
    vowel_count = 0
    stressed_vowel_num = None
    for i, ch in enumerate(stressed):
        if ch == '+' and i + 1 < len(stressed):
            stressed_vowel_num = vowel_count
        if ch.lower() in 'аеёиоуыэюя':
            vowel_count += 1

    if stressed_vowel_num is None:
        return None

    # Сопоставляем с индексом в phones
    FULL = {'a', 'o', 'e', 'i', 'u', 'ɛ'}
    vowel_idx = 0
    for idx, p in enumerate(phones):
        if p in FULL:
            if vowel_idx == stressed_vowel_num:
                return idx
            vowel_idx += 1
    return None


def reduce_vowels(phones: list[str], stress_idx: int | None = None) -> list[str]:
    """Редукция безударных гласных.

    Если stress_idx задан — не редуцируем гласную на этой позиции.
    Иначе — эвристика: последняя гласная ударная.
    """
    FULL = {'a', 'o', 'e', 'i', 'u', 'ɛ'}
    REDUCE = {
        'a': 'ɐ', 'o': 'ɐ', 'e': 'ɪ', 'i': 'ɪ', 'u': 'ʊ', 'ɛ': 'ɪ',
    }

    if stress_idx is None:
        for idx in range(len(phones) - 1, -1, -1):
            if phones[idx] in FULL:
                stress_idx = idx
                break

    out = []
    for idx, p in enumerate(phones):
        if idx != stress_idx and p in FULL:
            out.append(REDUCE.get(p, p))
        else:
            out.append(p)
    return out


def transcribe_oov(word: str) -> list[str]:
    """Транскрибирует OOV-слово по простым правилам буква→фонема."""
    word = word.lower()

    # Правило: «г → v» в окончаниях -ого, -его
    if 'ого' in word[-4:] or 'его' in word[-4:]:
        if word.endswith('ого') or word.endswith('его'):
            pos = len(word) - 2
            word = word[:pos] + 'в' + word[pos + 1:]

    # «сегодня» — историческое исключение
    word = word.replace('сегодн', 'севодн')

    # Схлопывание удвоенных букв (кроме «нн»)
    word = re.sub(r'([^н])\1', r'\1', word)

    # Возвратные глаголы: -ться / -тся → t̪s̪ ə
    reflexive_suffix = None
    if word.endswith('ться'):
        word = word[:-4]
        reflexive_suffix = ['t̪s̪', 'ə']
    elif word.endswith('тся'):
        word = word[:-3]
        reflexive_suffix = ['t̪s̪', 'ə']

    phones: list[str] = []
    n = len(word)
    i = 0

    while i < n:
        ch = word[i]

        if ch == 'ь':
            _soften(phones)
            i += 1
            continue

        if ch == 'ъ':
            i += 1
            continue

        # Йотированные гласные в начале слова или после гласной/ь/ъ
        if ch in 'еёюя':
            prev_is_vowelish = (i == 0 or word[i - 1] in 'аеёиоуыэюяьъ')
            if prev_is_vowelish:
                phones.append('j')

        if ch in VOWELS:
            if ch == 'и':
                _soften(phones)
            phones.append(VOWELS[ch])
            i += 1
            continue

        if ch in SOFTENING_VOWELS:
            _soften(phones)
            phones.append(SOFT_VOWEL_MAP[ch])
            i += 1
            continue

        if ch in CONSONANTS:
            if ch == 'с' and i + 1 < n and word[i + 1] in VOICED:
                phones.append('z̪')
            else:
                phones.append(CONSONANTS[ch])
            _devocalize(phones, word, i)
            if i + 1 < n and word[i + 1] == 'ь':
                _soften(phones)
            i += 1
            continue

        i += 1

    # Оглушение в самом конце
    if phones and phones[-1] in FINAL_DEVOICE:
        phones[-1] = FINAL_DEVOICE[phones[-1]]

    # Редукция безударных гласных с помощью silero-stress
    stress_idx = get_stress_index(word, phones)
    phones = reduce_vowels(phones, stress_idx)

    # Приклеиваем постфикс -ся
    if reflexive_suffix is not None:
        phones.extend(reflexive_suffix)

    return phones