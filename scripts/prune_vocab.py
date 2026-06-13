"""Prune the student's vocabulary to the tokens its task actually uses (W7 step 3).

The conversion corpus touches only ~4.5 k of Gemma's 262 k tokens, but the vocab
indexes the two largest tensors — the input embedding (262 k x 1536) and the
Per-Layer-Embedding / PLE table (262 k x 8960). Shrinking the vocab to ``--target``
slices both, cutting ~2.6 B params with no effect on the transformer blocks.

Correctness hinges on two things:

  * **Merge closure.** A BPE token only forms if its whole merge sub-tree is also
    kept; otherwise a word re-segments differently and the model sees
    out-of-distribution tokens. So the kept set is the transitive merge closure of
    the corpus tokens, plus all 256 byte-fallback tokens (leaves) and all special
    tokens. byte_fallback=True guarantees any unseen character still encodes.

  * **Verification gate.** Before touching weights, we re-tokenize the entire
    corpus with the pruned tokenizer and require that it yields EXACTLY the
    original ids remapped through old->new (and that decode round-trips). If that
    fails, segmentation diverged and we abort — no embedding surgery happens.

Output is a standalone mlx_lm ``gemma4_text`` dir with sliced embeddings, a pruned
tokenizer, and id references (bos/eos/pad) remapped in every config.

Usage:
    uv run python scripts/prune_vocab.py \
        --model models/gemma4-e2b-textonly-iter400-bf16 \
        --target 16384 \
        --out models/gemma4-e2b-textonly-iter400-bf16-vocab16k
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path
from typing import cast

import mlx.core as mx
from tokenizers import Tokenizer
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def corpus_token_ids(model_dir: Path, corpus_glob: str, extra_texts: list[str]) -> set[int]:
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    used: set[int] = set()
    for fp in glob.glob(corpus_glob):
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = " ".join(m.get("content", "") for m in rec.get("messages", []))
            used.update(tok.encode(text))
    # Tokens the model sees at INFERENCE but not in the training records: the skill
    # text + the prompt boilerplate that build_gemma_prompt prepends. Missing these
    # makes the eval prompt re-segment into out-of-distribution tokens.
    for text in extra_texts:
        used.update(tok.encode(text))
    return used


def inference_prompt_texts() -> list[str]:
    """Representative inference prompts (skill + boilerplate) for every use case,
    so their tokens are retained. Envelope/script tokens come from data/processed."""
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from datamorph.models.gemma_script_teacher import build_gemma_prompt  # noqa: E402

    env = {"format": "csv", "schema": {"columns": ["a", "b"]}, "sample_rows": [{"a": 1}]}
    texts = []
    for instr, fmt in [
        ("Convert this CSV to nested JSON.", "json"),
        ("Flatten this JSON to CSV.", "csv"),
        ("Parse this TXT log into CSV.", "csv"),
        ("Render this CSV as a TXT report.", "txt"),
        ("Migrate this file to the new schema.", "json"),
    ]:
        texts.append(build_gemma_prompt(env, instr, fmt))
        texts.append(build_gemma_prompt(env, instr, fmt, feedback="SyntaxError: unexpected EOF"))
    return texts


def merge_closure(seeds: set[str], form_all: dict[str, list[tuple[str, str]]]) -> set[str]:
    """All tokens reachable by recursively decomposing each seed via EVERY merge
    that can produce it. A BPE token often has several decompositions (e.g.
    ``Metadata`` = ``Meta``+``data`` or ``Met``+``adata``) and the encoder picks
    one by rank — so we must retain the pieces of all of them, else the pruned
    tokenizer re-segments along a path whose merges we dropped."""
    seen: set[str] = set()
    stack = list(seeds)
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        for a, b in form_all.get(t, ()):
            if a not in seen:
                stack.append(a)
            if b not in seen:
                stack.append(b)
    return seen


def build_kept(tok_json: dict, used_ids: set[int], target: int) -> tuple[list[str], dict[int, int]]:
    """Return (kept tokens sorted by old id, old_id->new_id map)."""
    vocab: dict[str, int] = tok_json["model"]["vocab"]
    inv = {i: t for t, i in vocab.items()}
    form_all: dict[str, list[tuple[str, str]]] = {}
    for a, b in tok_json["model"]["merges"]:
        form_all.setdefault(a + b, []).append((a, b))

    corpus_toks = {inv[i] for i in used_ids if i in inv}
    closure = merge_closure(corpus_toks, form_all)
    byte_toks = {t for t in vocab if len(t) == 6 and t.startswith("<0x") and t.endswith(">")}
    specials = {t["content"] for t in tok_json["added_tokens"]}
    must = closure | byte_toks | specials
    if len(must) > target:
        raise SystemExit(
            f"must-keep ({len(must)}) exceeds target ({target}); raise --target"
        )

    # Fill remaining slots with the lowest-old-id tokens (most fundamental/frequent).
    fillers = sorted(t for t in vocab if t not in must)
    fillers.sort(key=lambda t: vocab[t])
    kept_set = set(must)
    for t in fillers:
        if len(kept_set) >= target:
            break
        kept_set.add(t)

    kept = sorted(kept_set, key=lambda t: vocab[t])
    old2new = {vocab[t]: i for i, t in enumerate(kept)}
    return kept, old2new


def build_pruned_tokenizer(tok_json: dict, kept: list[str]) -> dict:
    kept_set = set(kept)
    new_vocab = {t: i for i, t in enumerate(kept)}

    # Keep a merge only if both inputs and the merged result are retained.
    new_merges = [
        [a, b]
        for a, b in tok_json["model"]["merges"]
        if a in kept_set and b in kept_set and (a + b) in kept_set
    ]

    pruned = json.loads(json.dumps(tok_json))  # deep copy
    pruned["model"]["vocab"] = new_vocab
    pruned["model"]["merges"] = new_merges
    new_added = []
    for at in pruned["added_tokens"]:
        if at["content"] in new_vocab:
            at = dict(at)
            at["id"] = new_vocab[at["content"]]
            new_added.append(at)
    pruned["added_tokens"] = new_added
    return pruned


def verify(
    orig_tok: Tokenizer,
    pruned_tok: Tokenizer,
    old2new: dict[int, int],
    corpus_glob: str,
    extra_texts: list[str],
) -> None:
    """Hard gate: pruned ids must equal remapped original ids on every text the
    model sees — training records AND inference prompts (skill + boilerplate)."""

    def texts():
        for fp in glob.glob(corpus_glob):
            for line in open(fp, encoding="utf-8"):
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    yield " ".join(m.get("content", "") for m in rec.get("messages", []))
        yield from extra_texts

    n = mismatch = 0
    for text in texts():
        orig = orig_tok.encode(text).ids
        want = [old2new[i] for i in orig if i in old2new]
        if len(want) != len(orig):
            mismatch += 1  # an original token wasn't kept -> would re-segment
        else:
            got = pruned_tok.encode(text).ids
            if got != want or pruned_tok.decode(got) != orig_tok.decode(orig):
                mismatch += 1
        n += 1
    rate = 100 * (n - mismatch) / n if n else 0
    print(f"      verification: {n - mismatch}/{n} texts exact ({rate:.2f}%)")
    if mismatch:
        raise SystemExit(f"ABORT: {mismatch} texts diverged — segmentation not preserved")


def remap_configs(model_dir: Path, out: Path, old2new: dict[int, int], target: int) -> None:
    cfg = json.loads((model_dir / "config.json").read_text())
    cfg["vocab_size"] = target
    cfg["vocab_size_per_layer_input"] = target
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if key in cfg and isinstance(cfg[key], int) and cfg[key] in old2new:
            cfg[key] = old2new[cfg[key]]
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    gc_path = model_dir / "generation_config.json"
    if gc_path.exists():
        gc = json.loads(gc_path.read_text())
        for key in ("bos_token_id", "pad_token_id"):
            if isinstance(gc.get(key), int) and gc[key] in old2new:
                gc[key] = old2new[gc[key]]
        if isinstance(gc.get("eos_token_id"), list):
            gc["eos_token_id"] = [old2new.get(i, i) for i in gc["eos_token_id"]]
        elif isinstance(gc.get("eos_token_id"), int) and gc["eos_token_id"] in old2new:
            gc["eos_token_id"] = old2new[gc["eos_token_id"]]
        (out / "generation_config.json").write_text(json.dumps(gc, indent=2))

    tc_path = model_dir / "tokenizer_config.json"
    if tc_path.exists():
        tc = json.loads(tc_path.read_text())
        atd = tc.get("added_tokens_decoder")
        if isinstance(atd, dict):
            tc["added_tokens_decoder"] = {
                str(old2new[int(i)]): v for i, v in atd.items() if int(i) in old2new
            }
        (out / "tokenizer_config.json").write_text(json.dumps(tc, indent=2))


def slice_embeddings(model_dir: Path, old2new: dict[int, int], target: int) -> dict:
    weights = cast("dict[str, mx.array]", mx.load(str(model_dir / "model.safetensors")))
    # Row order in the new tensor follows new_id = rank of old id.
    old_ids_in_new_order = sorted(old2new, key=lambda oid: old2new[oid])
    idx = mx.array(old_ids_in_new_order)
    for key in ("model.embed_tokens.weight", "model.embed_tokens_per_layer.weight"):
        w = weights[key]
        assert w.shape[0] == 262144, f"{key} not vocab-indexed?"
        weights[key] = w[idx]
        assert weights[key].shape[0] == target
    return weights


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="text-only fused bf16 dir (step 1+2 output)")
    ap.add_argument("--corpus", default=str(PROJECT_ROOT / "data/processed/*.jsonl"))
    ap.add_argument("--target", type=int, default=16384)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model_dir = Path(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok_json = json.loads((model_dir / "tokenizer.json").read_text())

    print("[1/6] collecting corpus + inference-prompt tokens ...")
    extra_texts = inference_prompt_texts()
    used = corpus_token_ids(model_dir, args.corpus, extra_texts)
    print(f"      {len(used)} distinct tokens used")

    print(f"[2/6] building kept set (merge closure + bytes + specials, target={args.target}) ...")
    kept, old2new = build_kept(tok_json, used, args.target)
    print(f"      kept {len(kept)} tokens -> new ids 0..{len(kept) - 1}")

    print("[3/6] building pruned tokenizer ...")
    pruned_json = build_pruned_tokenizer(tok_json, kept)
    pruned_tok = Tokenizer.from_str(json.dumps(pruned_json))
    orig_tok = Tokenizer.from_str(json.dumps(tok_json))
    (out / "tokenizer.json").write_text(json.dumps(pruned_json))
    print(f"      merges {len(tok_json['model']['merges'])} -> {len(pruned_json['model']['merges'])}")

    print("[4/6] VERIFY: pruned tokenization == remapped original (corpus + prompts) ...")
    verify(orig_tok, pruned_tok, old2new, args.corpus, extra_texts)

    print("[5/6] slicing embeddings + remapping config ids ...")
    weights = slice_embeddings(model_dir, old2new, args.target)
    remap_configs(model_dir, out, old2new, args.target)
    for fn in ("chat_template.jinja",):
        if (model_dir / fn).exists():
            shutil.copy(model_dir / fn, out / fn)

    print(f"[6/6] writing model to {out} ...")
    mx.save_safetensors(str(out / "model.safetensors"), weights)
    n_params = sum(int(v.size) for v in weights.values())
    print(f"      text-only pruned params: {n_params / 1e9:.3f} B")
    print(f"\nDone -> {out}")
    print("Eval: GEMMA_TEXT_ONLY=1 GEMMA_MLX_MODEL=<out> python scripts/run_pipeline_baseline.py ...")


if __name__ == "__main__":
    main()
