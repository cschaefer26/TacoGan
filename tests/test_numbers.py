import unittest

from text.numbers_de import normalize_numbers_de
from text.numbers_en import normalize_numbers_en


class TestNumbers(unittest.TestCase):

    def test_normalize_en(self):
        text = normalize_numbers_en('It happened 1984.')
        self.assertEqual('It happened nineteen eighty-four.', text)

        text = normalize_numbers_en('It happened 2016.')
        self.assertEqual('It happened twenty sixteen.', text)

        text = normalize_numbers_en('This is 10.5 pound.')
        self.assertEqual('This is ten point five pound.', text)

        text = normalize_numbers_en('1050')
        self.assertEqual('ten fifty', text)

    def test_normalize_de(self):
        text = normalize_numbers_de('Es war 1984.')
        self.assertEqual('Es war neunzehnhundertvierundachtzig.', text)

        text = normalize_numbers_de('Es war 2016.')
        self.assertEqual('Es war zweitausendsechzehn.', text)

        text = normalize_numbers_de('Es war 1800.')
        self.assertEqual('Es war achtzehnhundert.', text)

        text = normalize_numbers_de('Macht 1002 Dollar.')
        self.assertEqual('Macht eintausendzwei Dollar.', text)

        text = normalize_numbers_de('Macht 12,5 Euro.')
        self.assertEqual('Macht zwölf Komma fünf Euro.', text)

        text = normalize_numbers_de('Macht 9.52 Euro.')
        self.assertEqual('Macht neun Komma fünf zwei Euro.', text)

        text = normalize_numbers_de('Das sind 1000.04 Grad.')
        self.assertEqual('Das sind eintausend Komma null vier Grad.', text)

