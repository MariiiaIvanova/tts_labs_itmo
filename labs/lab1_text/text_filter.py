"""Normalized / non-normalized classifier — skeleton for lab 1.

Run as a script to score yourself on the development set::

    python text_filter.py
"""

import csv

import pandas as pd
import re
import emoji

from sklearn.metrics import f1_score, precision_score, recall_score

DEV_SET_PATH = "data/dev_sentences.csv"


class TextFilter:
    """Decides whether an utterance is usable as a training example.

    Example:
        >>> textfilter = TextFiзтттlter()
        >>> textfilter.filter("Я вышел из дома.")
        1
        >>> textfilter.filter("Александрову Г. П.")
        0
    """

    def __init__(self):
        """Prepare the classifier's resources.

        Compiled regular expressions, abbreviation and contraction dictionaries, a
        trained model — anything that should not be rebuilt for every utterance.
        """

        # Here goes your initialization logic
        abbrev_list = [
            'АН', 'АТС', 'АХЧ', 'БТ', 'ВМК', 'ГБ', 'ГПУ', 'ДПИ', 'КВВК', 'КВН',
            'КП', 'КПП', 'КПСС', 'КПЭ', 'ЛГУ', 'МВД', 'МГУ', 'МПВО', 'МТС', 'НКВД',
            'ПТУ', 'СНО', 'СС', 'ССР', 'СХШ', 'США', 'ТГУ', 'ТПИ', 'УВД', 'ФБР',
            'ФД', 'ЦДЛ', 'ЦО', 'ЧК'
        ]
        
        # Паттерн для поиска всех аббревиатур
        abbrev_pattern = '|'.join([rf'\b{abv}\b' for abv in abbrev_list])

        standard_punct = r'.,!?;:…—\-–‑"\'«»()'

        self.patterns = {

            # Цифры
            'digits': r'\d',
            
            # Латиница
            'latin': r'[A-Za-z]',
            
            # Аббревиатуры
            'abbrev': abbrev_pattern,
            'other_abbrev': r'\b(?:тов\.|проф\.|доц\.|акад\.|д-р|канд\.)',

            # Сокращения
            'with_dots': r'\b(?:т\.\s*е|т\.\s*к|т\.\s*д|т\.\s*п|и\.\s*т\.\s*д|и\.\s*т\.\s*п|т\.\s*о)\b',
            'address': r'\b(?:ул\.?|д\.?|г-н|г-жа|г\.?|пр\.?|пер\.?|пл\.?|бульв\.?|наб\.?|ш\.?|стр\.?|рис\.?|см\.?|табл\.?)\b',
            'initials': r'\b[А-Я]\.\s*(?:[А-Я]\.\s*)?[А-Я][а-я]+\b',
            
            # Технические символы, обозначения и нестандартные знаки
            'tech_symbols': r'[*\/@+<>{}\[\]\\|~`=^&]',
            'symbols': r'[%°$€£¥©®™№±√∫∑∞≈≠≤≥]',

            'nonstandard_punct': r'[^\w\s' + standard_punct + r']',
            
            # Эмодзи
            'emoji_text': r'[:;=][-o*^]?[()\[\]{}<>DdpPрР/\\|3cCOo0]|[:;=][()]|[XO][-]?[()]|<3|3<',

            # междометия
            'short_interjection': r'\b(?:ага|агу|ай|ахаха|ах|брр|гм|гмм|е|ё|ммм|ого|ой|ох|охохо|угу|ура|ух|ха|хе|хм|хмм|хы|ых|ыы|ыых|э|эх|ю|мда|хо|хе|хи|гы|гх|фу|фи|тьфу|эй|эге|эхе|эх)\b',
            'e_interjection2': r'\b[еёэы]\b|\b[еёэы]\s*[‒–—‐-]\s*[еёэ](?:\s*[‒–—‐-]\s*[еёэ])?\b',
            'dubble_interjection':  r'\b(?:х[еёэ]х[еёэ])+\b|\b(?:хаха)+\b|\b(?:хихи)+\b|\b(?:хохо)+\b|\b[ы]{2,}\b',
            
            # Повторяющиеся и смешанные знаки препинания
            'special_punct': r'[!?]{2,}|[!?][,.?!]|[,.]?[!?]{2,}|[()\[\]{}]',
        }  
        pass

    def filter(self, text: str) -> int:
        """Classify a single utterance.

        Args:
            text: Utterance text, already passed through :class:`TextNormalizer`.

        Returns:
            ``1`` if the text is normalized and the utterance can be used for
            training; 
            ``0`` if it contains something the speaker pronounced
            differently from how it is written, and the utterance should be dropped.
        """
        if any(emoji.is_emoji(char) for char in text):
            return 0
        
        # Here goes your filterting logic
        for pattern in self.patterns.values():
            if re.search(pattern, text, re.IGNORECASE):
                return 0  # ненормализован
        return 1


if __name__ == "__main__":
    textfilter = TextFilter()

    dev_files = pd.read_csv(
        DEV_SET_PATH, sep="|", encoding="utf-8", quoting=csv.QUOTE_NONE, header=0
    )

    dev_files["predicted"] = dev_files["text"].apply(textfilter.filter)
    
    prc = precision_score(dev_files["is_normalized"], dev_files["predicted"])
    rec = recall_score(dev_files["is_normalized"], dev_files["predicted"])
    f1 = f1_score(dev_files["is_normalized"], dev_files["predicted"])
    print(f"F1 Score is {f1:.4f}, Precision is {prc:.4f}, Recall is {rec:.4f}")
