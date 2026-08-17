from collections import Counter
import time
class Vocab:
    def __init__(self, text, min_freq=5):
        '''
        :param text: 2D array of tokens
        :param min_freq: minimum frequency for a token to not be removed
        '''
        # flatten text
        text = [token for line in text for token in line]

        self.original_count = sorted(Counter(text).items(), key = lambda x: x[1], reverse=True)
        self.count = {tok: freq for tok, freq in self.original_count if freq >= min_freq}
        self.token_to_idx = {tok: i for i, (tok, freq) in enumerate(self.count.items())}
        self.idx_to_token = {i: tok for i, (tok, freq) in enumerate(self.count.items())}

    def __len__(self):
        return len(self.count)

    def __getitem__(self, indices):
        if indices is not None and isinstance(indices, list):
            return [self[index] for index in indices]
        if isinstance(indices, str):
            if indices not in self.token_to_idx.keys():
                raise IndexError
            return self.token_to_idx[indices]
        if isinstance(indices, int):
            if indices not in self.idx_to_token.keys():
                raise IndexError
            return self.idx_to_token[indices]
        raise TypeError(f"Unsupported key type: {type(indices)}")
