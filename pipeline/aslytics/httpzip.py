"""Open a huge remote .zip and read single files out of it with HTTP range requests.

Lets us pull a few thousand clips from a 43 GB archive without downloading it.
(The `remotezip` package fails on Microsoft's server, which rejects suffix ranges.)
"""

from __future__ import annotations

import io
import zipfile

import requests

BLOCK = 4 * 1024 * 1024


class HttpFile(io.RawIOBase):
    """Read-only seekable file over HTTP, with a one-block read-ahead cache."""

    def __init__(self, url: str, session: requests.Session | None = None) -> None:
        self.url = url
        self.session = session or requests.Session()
        head = self.session.head(url, allow_redirects=True, timeout=60)
        head.raise_for_status()
        self.size = int(head.headers["Content-Length"])
        self.pos = 0
        self._cache_start = 0
        self._cache = b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, min(self.size, base + offset))
        return self.pos

    def _fetch(self, start: int, length: int) -> bytes:
        end = min(self.size, start + length) - 1
        resp = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=300)
        resp.raise_for_status()
        if resp.status_code != 206:
            raise IOError(f"Server ignored the range request (HTTP {resp.status_code})")
        return resp.content

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        cached_end = self._cache_start + len(self._cache)
        if not (self._cache_start <= self.pos and self.pos + n <= cached_end):
            self._cache_start = self.pos
            self._cache = self._fetch(self.pos, max(n, BLOCK))
        offset = self.pos - self._cache_start
        self.pos += n
        return self._cache[offset:offset + n]

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def open_remote_zip(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(HttpFile(url), buffer_size=BLOCK))
