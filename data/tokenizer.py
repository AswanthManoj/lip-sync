from collections import Counter

class CharacterTokenizer:
    def __init__(self, texts=None):
        self.char_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_char = {0: '<PAD>', 1: '<UNK>'}
        
        if texts is not None:
            self.fit(texts)
    
    def fit(self, texts):
        """Build vocabulary from texts"""
        # Count characters
        all_chars = Counter(''.join(texts))
        
        # Add characters to vocabulary
        for char, _ in all_chars.most_common():
            if char not in self.char_to_idx:
                idx = len(self.char_to_idx)
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
        
        print(f"Tokenizer vocabulary size: {len(self.char_to_idx)}")
        return self
    
    def encode(self, text):
        """Convert text to indices"""
        return [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) for char in text]
    
    def decode(self, indices):
        """Convert indices to text"""
        return ''.join(self.idx_to_char.get(idx, '<UNK>') for idx in indices)
    
    def vocab_size(self):
        return len(self.char_to_idx)
