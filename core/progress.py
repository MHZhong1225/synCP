from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TypeVar

T = TypeVar("T")


def progress(items: Iterable[T], desc: str, total: int | None = None, leave: bool = True) -> Iterator[T]:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        yield from items
        return

    yield from tqdm(items, desc=desc, total=total, leave=leave, dynamic_ncols=True)


@contextmanager
def progress_bar(total: int, desc: str):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        yield _NullProgressBar()
        return

    with tqdm(total=total, desc=desc, dynamic_ncols=True) as bar:
        yield bar


class _NullProgressBar:
    def update(self, n: int = 1) -> None:
        return None
