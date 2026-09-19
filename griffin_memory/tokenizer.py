"""No downloaded vocabulary or weights. Byte tokenizer or locally trained BPE."""
import json
from pathlib import Path

PAD, BOS, EOS = 0, 1, 2
SPECIALS = ["<pad>", "<bos>", "<eos>"]


class Tokenizer:
    def __init__(self, backend=None):
        self.backend = backend

    @property
    def vocab_size(self):
        return 259 if self.backend is None else self.backend.get_vocab_size()

    def encode(self, text):
        if self.backend is None:
            return [b + 3 for b in text.encode("utf-8")]
        return self.backend.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        if self.backend is None:
            return bytes(i - 3 for i in ids if 3 <= i < 259).decode("utf-8", errors="replace")
        return self.backend.decode(ids, skip_special_tokens=True)

    def save(self, path):
        if self.backend is None:
            Path(path).write_text(json.dumps({"type": "utf8_bytes", "version": 1}))
        else:
            self.backend.save(str(path))

    @classmethod
    def load(cls, path):
        obj = json.loads(Path(path).read_text())
        if obj.get("type") == "utf8_bytes":
            return cls()
        from tokenizers import Tokenizer as BPETokenizer
        backend = BPETokenizer.from_file(str(path))
        if [backend.token_to_id(s) for s in SPECIALS] != [PAD, BOS, EOS]:
            raise ValueError("Tokenizer must use PAD=0, BOS=1, EOS=2")
        return cls(backend)

    @classmethod
    def train_bpe(cls, texts, vocab_size):
        from tokenizers import Tokenizer as BPETokenizer, models, trainers, pre_tokenizers, decoders
        if vocab_size < 259:
            raise ValueError("BPE vocabulary must have at least 259 entries")
        backend = BPETokenizer(models.BPE())
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS,
                                     initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        backend.train_from_iterator(texts, trainer=trainer)
        return cls(backend)
