import re

from tqdm import tqdm

LTR_PARTIALS = set("▏▎▍▌▋▊▉")


class RtlTqdm(tqdm):
    """tqdm progress bar that fills from right to left."""

    @staticmethod
    def format_meter(n, total, elapsed, **kwargs):  # type: ignore
        s = tqdm.format_meter(n, total, elapsed, **kwargs)

        def reverse_bar(match):
            bar = match.group(1)[::-1]
            return "|" + "".join("▐" if ch in LTR_PARTIALS else ch for ch in bar) + "|"

        return re.sub(r"\|([^|]+)\|", reverse_bar, s, count=1)
