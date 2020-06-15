from typing import Callable, List


class Tokenizer:

    def __init__(self,
                 cleaners: Callable[[str], str],
                 symbols: str) -> None:
        self.cleaner = cleaners
        self.symbols = list(symbols)
        self.symbol_id = {s: i for i, s in enumerate(symbols, 1)}
        self.id_symbol = {i: s for i, s in enumerate(symbols, 1)}
        self.id_symbol[0] = '_'

    def encode(self, text: str) -> List[int]:
        text = self.cleaner(text)
        encoded = [self.symbol_id[s] for s in text if s in self.symbol_id]
        return encoded

    def decode(self, sequence: List[int]) -> str:
        decoded = [self.id_symbol[s] for s in sequence if s in self.id_symbol]
        return ''.join(decoded)

