""" adapted from https://github.com/keithito/tacotron """

import re
from typing import Callable

import unidecode

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


def basic_cleaners(text):
    text = text.lower()
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    text = unidecode.unidecode(text)
    text = text.lower()
    text = normalize_numbers_en(text)
    text = collapse_whitespace(text)
    return text


def german_cleaners(text):
    text = text.lower()
    text = normalize_numbers_de(text)
    text = collapse_whitespace(text)
    return text


def get_cleaners(cleaners_str: str) -> Callable[[str], str]:
    if cleaners_str == 'english_cleaners':
        return english_cleaners
    elif cleaners_str == 'german_cleaners':
        return german_cleaners
    else:
        raise ValueError(f'cleaners not supported: {cleaners_str}')

