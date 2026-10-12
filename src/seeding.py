import zlib

def item_seed(base_seed: int, prompt_name: str, idx: int) -> int:
    return zlib.crc32(f"{base_seed}|{prompt_name}|{idx}".encode("utf-8"))
