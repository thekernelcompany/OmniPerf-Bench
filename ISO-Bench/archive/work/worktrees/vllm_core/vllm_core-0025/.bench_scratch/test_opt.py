import time
from vllm.transformers_utils.tokenizer import decode_tokens, encode_tokens

# Dummy tokenizer to simulate fast and slow paths
class DummyTokenizer:
    def __init__(self):
        # minimal attributes used by utils (not needed for this bench)
        pass

    def decode(self, token_ids, skip_special_tokens=None):
        # Simulate overhead: unnecessary list conversion and joins
        lst = list(token_ids)  # extra conversion
        # Simulate processing
        acc = 0
        for t in lst:
            acc ^= (t * 1315423911) & 0xFFFFFFFF
        return str(acc)

    # Fast path that avoids extra work
    def _decode(self, token_ids, skip_special_tokens=None):
        acc = 0
        for t in token_ids:
            acc ^= (t * 1315423911) & 0xFFFFFFFF
        return str(acc)

    def encode(self, text, **kwargs):
        # Simulate overhead: intermediate list and extra operations
        codes = [ord(c) for c in list(text)]
        out = []
        for v in codes:
            out.append((v * 3) % 256)
        return out

    # Fast path that avoids redundant list creation
    def _encode(self, text, **kwargs):
        out = []
        for c in text:
            out.append((ord(c) * 3) % 256)
        return out


def bench_decode(tokenizer, n_iters=2000, toks=1024):
    token_ids = list(range(toks))
    t0 = time.perf_counter()
    s = 0
    for _ in range(n_iters):
        s += int(decode_tokens(tokenizer, token_ids))
    t1 = time.perf_counter()
    return t1 - t0, s


def bench_encode(tokenizer, n_iters=2000, length=1024):
    text = "a" * length
    t0 = time.perf_counter()
    s = 0
    for _ in range(n_iters):
        s += sum(encode_tokens(tokenizer, text))
    t1 = time.perf_counter()
    return t1 - t0, s


def main():
    tok = DummyTokenizer()
    dec_t, s1 = bench_decode(tok)
    enc_t, s2 = bench_encode(tok)
    print({"decode_time": dec_t, "encode_time": enc_t, "checksum": (s1, s2)})


if __name__ == "__main__":
    main()
