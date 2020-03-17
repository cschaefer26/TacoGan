import unittest

from text.numbers_de import normalize_numbers_de
from text.numbers_en import normalize_numbers_en
from text.text_cleaner import english_cleaners
from text.tokenizer import Tokenizer


class TestTokenizer(unittest.TestCase):

    def test_encode(self):
        cleaner = english_cleaners
        symbols = ' abcdefghijklmnopqrstuvwxyz.'
        tokenizer = Tokenizer(cleaner, symbols)

        encoded = tokenizer.encode('a b&')
        self.assertEqual([2, 1, 3], encoded)

        encoded = tokenizer.encode('He is 12 years old.')
        decoded = tokenizer.decode(encoded)
        self.assertEqual('he is twelve years old.', decoded)