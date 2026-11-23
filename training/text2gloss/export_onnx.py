"""Export the fine-tuned model to quantised ONNX for in-browser inference, and check it.

    python export_onnx.py --model model/ --out onnx_model/

Writes the layout transformers.js expects: tokenizer and config files at the
top level and the int8 encoder/decoder under onnx/.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from optimum.onnxruntime import ORTModelForSeq2SeqLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

from train import MAX_TARGET, PREFIX


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model")
    parser.add_argument("--out", default="onnx_model")
    parser.add_argument("--try", dest="samples", nargs="*", default=["Hello, my name is Sam.", "I don't like apples."])
    args = parser.parse_args()
    out, full = Path(args.out), Path(args.out) / "_full"

    ORTModelForSeq2SeqLM.from_pretrained(args.model, export=True).save_pretrained(full)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.save_pretrained(out)
    shutil.copy(full / "config.json", out / "config.json")
    if (full / "generation_config.json").exists():
        shutil.copy(full / "generation_config.json", out / "generation_config.json")

    config = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    for name in ("encoder_model", "decoder_model", "decoder_with_past_model"):
        if not (full / f"{name}.onnx").exists():
            continue
        ORTQuantizer.from_pretrained(full, file_name=f"{name}.onnx").quantize(save_dir=out / "onnx", quantization_config=config)
    shutil.rmtree(full)

    total = sum(p.stat().st_size for p in (out / "onnx").glob("*.onnx")) / 1e6
    print(f"Quantised model: {total:.0f} MB in {out}/onnx")

    quantised = ORTModelForSeq2SeqLM.from_pretrained(
        out, subfolder="onnx",
        encoder_file_name="encoder_model_quantized.onnx",
        decoder_file_name="decoder_model_quantized.onnx",
        decoder_with_past_file_name="decoder_with_past_model_quantized.onnx",
    )
    for sentence in args.samples:
        ids = tokenizer(PREFIX + sentence, return_tensors="pt")
        gloss = tokenizer.decode(quantised.generate(**ids, max_length=MAX_TARGET)[0], skip_special_tokens=True)
        print(f"{sentence} -> {gloss}")


if __name__ == "__main__":
    main()
