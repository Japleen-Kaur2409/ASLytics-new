"""YouTube captions with timing. Ported from the old prototype's youtube_caption_utils.py.

Used for offline experiments. The extension reads captions in the page itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

CONTRACTIONS = {
    "it's": "it is", "you're": "you are", "i'm": "i am", "don't": "do not",
    "can't": "cannot", "they're": "they are", "i'll": "i will", "you'll": "you will",
    "we'll": "we will", "they'll": "they will", "i'd": "i would", "you'd": "you would",
    "he'd": "he would", "she'd": "she would", "we'd": "we would", "they'd": "they would",
    "it'll": "it will", "won't": "will not", "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "weren't": "were not", "hasn't": "has not", "haven't": "have not",
    "hadn't": "had not", "doesn't": "does not", "didn't": "did not",
    "wouldn't": "would not", "shouldn't": "should not", "mightn't": "might not",
    "mustn't": "must not", "let's": "let us", "who's": "who is", "what's": "what is",
    "where's": "where is", "when's": "when is", "why's": "why is", "how's": "how is",
    "couldn't": "could not", "shan't": "shall not", "she's": "she is", "he's": "he is",
    "that's": "that is", "there's": "there is", "here's": "here is", "we're": "we are",
    "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
}

_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


@dataclass
class Caption:
    start: float
    duration: float
    text: str

    @property
    def end(self) -> float:
        return self.start + self.duration


def extract_video_id(url_or_id: str) -> str | None:
    """Video ID from a watch / youtu.be / shorts / embed URL, or a bare ID."""
    if _ID.match(url_or_id):
        return url_or_id
    parsed = urlparse(url_or_id)
    host = parsed.netloc.lower().removeprefix("www.").removeprefix("m.")
    candidate = None
    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif host.endswith("youtube.com"):
        parts = [p for p in parsed.path.split("/") if p]
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [None])[0]
        elif len(parts) >= 2 and parts[0] in ("shorts", "embed", "live"):
            candidate = parts[1]
    return candidate if candidate and _ID.match(candidate) else None


def expand_contractions(text: str) -> str:
    def swap(match: re.Match) -> str:
        word = match.group(0)
        return CONTRACTIONS.get(word.lower().replace("’", "'"), word)

    return re.sub(r"[A-Za-z]+['’][A-Za-z]+", swap, text)


def clean_text(text: str) -> str:
    """Drop sound tags like [Music] and speaker chevrons, collapse whitespace."""
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text)
    text = text.replace(">>", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def fetch_captions(video_id: str, languages: tuple[str, ...] = ("en",)) -> list[Caption]:
    """Timed English captions for a video. Raises if none are available."""
    from youtube_transcript_api import YouTubeTranscriptApi

    fetched = YouTubeTranscriptApi().fetch(video_id, languages=list(languages))
    captions = []
    for snippet in fetched:
        text = clean_text(snippet.text)
        if text:
            captions.append(Caption(float(snippet.start), float(snippet.duration), text))
    return captions
