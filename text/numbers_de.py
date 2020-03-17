import re

from num2words import num2words

_decimal_number_re = re.compile(r'([0-9]+[\.,][0-9]+)')
_number_re = re.compile(r'[0-9]+')


def _expand_decimal(m):
    m = m.group(1).replace(',', '.')
    parts = m.split('.')
    expanded = [parts[0], ' Komma ']
    for p in parts[1:]:
        p = ' '.join(list(p))
        expanded.append(p)
    return ''.join(expanded)


def _expand_number(m):
    num = int(m.group(0))
    if 1500 < num < 2000:
        if num % 100 == 0:
            return num2words(num // 100, lang='de') + 'hundert'
        else:
            return num2words(num // 100, lang='de') + 'hundert' \
                   + num2words(num % 100, lang='de')
    else:
        return num2words(num, lang='de')


def normalize_numbers_de(text):
    text = re.sub(_decimal_number_re, _expand_decimal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text
