"""Small dependency-free helpers."""

_OPEN, _CLOSE = "<think>", "</think>"


def _partial_suffix(buf: str, tag: str) -> int:
    """Length of the longest suffix of buf that is a proper prefix of tag."""
    for n in range(min(len(buf), len(tag) - 1), 0, -1):
        if tag.startswith(buf[-n:]):
            return n
    return 0


def strip_think(pieces):
    """Filter a stream of text pieces, dropping anything inside <think>...</think>.

    Works even when a tag is split across pieces. Yields only visible text.
    """
    buf, in_think, just_closed = "", False, False
    for piece in pieces:
        buf += piece
        while True:
            if in_think:
                end = buf.find(_CLOSE)
                if end == -1:
                    buf = buf[-(len(_CLOSE) - 1):]      # keep a possible partial closing tag
                    break
                buf, in_think, just_closed = buf[end + len(_CLOSE):], False, True
            else:
                if just_closed:                          # drop blank lines right after </think>
                    buf = buf.lstrip("\n")
                    if not buf:
                        break
                    just_closed = False
                start = buf.find(_OPEN)
                if start == -1:
                    keep = _partial_suffix(buf, _OPEN)
                    out, buf = buf[:len(buf) - keep], buf[len(buf) - keep:]
                    if out:
                        yield out
                    break
                if start > 0:
                    yield buf[:start]
                buf, in_think = buf[start + len(_OPEN):], True
    if not in_think and buf:
        yield buf


def parse_search_plan(out: str):
    """Turn the model's routing output into a search query, or None if it said NO."""
    text = "".join(strip_think([out])).strip()
    if not text:
        return None
    line = text.splitlines()[0].strip().strip("\"'`“”").strip()
    if line.upper().rstrip(".") == "NO":
        return None
    for prefix in ("search query:", "query:", "search:"):
        if line.lower().startswith(prefix):
            line = line[len(prefix):].strip().strip("\"'`“”").strip()
    return line[:120] or None
