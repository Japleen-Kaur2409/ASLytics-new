"""Command line entry point: `aslytics <command>`."""

from __future__ import annotations

import argparse
from pathlib import Path


def cmd_lexicon(args: argparse.Namespace) -> None:
    from .asl_citizen import build_lexicon

    df = build_lexicon(args.data, per_sign=args.per_sign, limit=args.limit, glosses=args.gloss)
    ok = df[df["ok"]]
    print(f"{len(ok)}/{len(df)} clips passed quality, covering {ok['gloss'].nunique()} signs")


def cmd_how2sign(args: argparse.Namespace) -> None:
    from .how2sign import build_corpus

    df = build_corpus(args.root, args.data, args.split, limit=args.limit)
    ok = df[df["ok"]]
    hours = ok["n_frames"].sum() / 15 / 3600
    print(f"{len(ok)}/{len(df)} sentences passed quality ({hours:.1f} h of signing)")


def cmd_preview(args: argparse.Namespace) -> None:
    from .posefile import load_clip
    from .preview import save_preview

    for path in args.clips:
        out = Path(args.out) / (Path(path).stem + ".png")
        save_preview(load_clip(path), out, title=Path(path).stem)
        print(out)


def cmd_stitch(args: argparse.Namespace) -> None:
    from .manifest import load_manifest
    from .posefile import load_clip, save_clip
    from .preview import save_preview
    from .stitch import stitch

    lexicon = load_manifest(args.data, "lexicon")
    lexicon = lexicon[lexicon["ok"]].drop_duplicates("gloss").set_index("gloss")
    missing = [g for g in args.glosses if g not in lexicon.index]
    if missing:
        raise SystemExit(f"Not in the dictionary: {missing}")
    clips = [load_clip(Path(args.data) / lexicon.loc[g, "pose_path"]) for g in args.glosses]
    result = stitch(clips)
    out = Path(args.out)
    save_clip(result, out)
    save_preview(result, out.with_suffix(".png"), frames=12, title=" ".join(args.glosses))
    print(f"{out} ({result.duration:.1f}s)")


def cmd_export(args: argparse.Namespace) -> None:
    from .export import export_lexicon

    stats = export_lexicon(args.data, args.out)
    print(f"{stats['signs']} signs, {stats['frames']} frames, {stats['megabytes']:.1f} MB -> {args.out}")


def cmd_captions(args: argparse.Namespace) -> None:
    from .captions import extract_video_id, fetch_captions

    video_id = extract_video_id(args.url)
    if not video_id:
        raise SystemExit(f"Not a YouTube URL or video ID: {args.url}")
    for c in fetch_captions(video_id):
        print(f"{c.start:8.2f} {c.duration:6.2f}  {c.text}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="aslytics")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("lexicon", help="build the sign dictionary from ASL Citizen")
    p.add_argument("--data", default="../data")
    p.add_argument("--per-sign", type=int, default=2)
    p.add_argument("--limit", type=int, help="stop after this many videos")
    p.add_argument("--gloss", nargs="*", help="only these glosses, e.g. APPLE HUG")
    p.set_defaults(func=cmd_lexicon)

    p = sub.add_parser("how2sign", help="build the sentence corpus from local How2Sign clips")
    p.add_argument("--root", required=True, help="folder containing sentence_level/")
    p.add_argument("--split", required=True, choices=["train", "val", "test"])
    p.add_argument("--data", default="../data")
    p.add_argument("--limit", type=int)
    p.set_defaults(func=cmd_how2sign)

    p = sub.add_parser("preview", help="render pose clips to PNG strips")
    p.add_argument("clips", nargs="+")
    p.add_argument("--out", default="../data/previews")
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("stitch", help="stitch dictionary signs into one clip")
    p.add_argument("glosses", nargs="+")
    p.add_argument("--data", default="../data")
    p.add_argument("--out", default="../data/stitched.npz")
    p.set_defaults(func=cmd_stitch)

    p = sub.add_parser("export-lexicon", help="pack the dictionary for the browser extension")
    p.add_argument("--data", default="../data")
    p.add_argument("--out", default="../extension/public/lexicon")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("captions", help="print timed captions for a YouTube video")
    p.add_argument("url")
    p.set_defaults(func=cmd_captions)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
