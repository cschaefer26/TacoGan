""" adapted from https://github.com/keithito/tacotron """

import re
from abc import ABC, abstractmethod
from typing import Callable

import unidecode
from phonemizer.phonemize import phonemize

from text.numbers_de import normalize_numbers_de
from text.numbers_en import normalize_numbers_en


_whitespace_re = re.compile(r'\s+')
_abbreviations_en = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'), ('mr', 'mister'), ('dr', 'doctor'), ('st', 'saint'),
    ('co', 'company'), ('jr', 'junior'), ('maj', 'major'), ('gen', 'general'),
    ('drs', 'doctors'), ('rev', 'reverend'), ('lt', 'lieutenant'), ('hon', 'honorable'),
    ('sgt', 'sergeant'), ('capt', 'captain'), ('esq', 'esquire'), ('ltd', 'limited'),
    ('col', 'colonel'), ('ft', 'fort'),
]]


def expand_abbreviations_en(text):
    for regex, replacement in _abbreviations_en:
        text = re.sub(regex, replacement, text)
    return text


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def to_phonemes(text, language):
    return phonemize(text, language=language, backend='espeak', strip=True,
                     preserve_punctuation=True, with_stress=False, njobs=1,
                     punctuation_marks=';:,.!?¡¿—…"«»“”()', language_switch='remove-flags')


class Cleaner(ABC):

    @abstractmethod
    def __call__(self, text):
        raise NotImplementedError()


class BasicCleaner(Cleaner):

    def __init__(self, language):
        self.language = language

    def __call__(self, text):
        text = collapse_whitespace(text)
        text = to_phonemes(text, self.language)
        return text


class EnglishCleaner(Cleaner):

    def __call__(self, text):
        text = unidecode.unidecode(text)
        text = normalize_numbers_en(text)
        text = collapse_whitespace(text)
        text = to_phonemes(text, 'en')
        return text


class GermanCleaner(Cleaner):

    def __call__(self, text):
        text = normalize_numbers_de(text)
        text = collapse_whitespace(text)
        text = to_phonemes(text, 'de')
        return text


def get_cleaner(language: str) -> Callable[[str], str]:
    if language == 'en':
        return EnglishCleaner()
    elif language == 'de':
        return GermanCleaner()
    else:
        return BasicCleaner(language)
