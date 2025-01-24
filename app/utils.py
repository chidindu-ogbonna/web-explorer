def chunkify(s: str, chunk_size: int) -> list[str]:
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]
