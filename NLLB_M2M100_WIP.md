# WIP: NLLB-200 / M2M-100 encoder-decoder support for vLLM

**Status: DRAFT — not yet validated on hardware.** Adds in-tree support for the
`M2M100ForConditionalGeneration` architecture, which covers **NLLB-200**
(`facebook/nllb-200-{distilled-600M,1.3B,3.3B}`) and **M2M-100**
(`facebook/m2m100_{418M,1.2B}`) — currently unsupported by vLLM (only Whisper is
in-tree for enc-dec; BART/mBART/Florence2 live in the external `bart-plugin`).

## What was done
- New model file `vllm/model_executor/models/m2m_100.py`, ported from the proven
  `bart-plugin` BART implementation (Apache-2.0) — reuses its encoder/decoder
  attention, cross-attention, KV-cache wiring, and `EncDecMultiModalProcessor`
  text path unchanged.
- Registered `M2M100ForConditionalGeneration` in `_MULTIMODAL_MODELS`
  (`registry.py`), alongside Whisper.

## M2M-100 / NLLB architectural diffs vs BART (handled)
1. **Sinusoidal positional embeddings** (`M2M100SinusoidalPositionalEmbedding`)
   instead of BART's learned table. Precomputed, non-persistent buffer (matches
   HF — not in the checkpoint), positions offset by `padding_idx + 1`.
2. **Final `layer_norm`** at the end of the encoder *and* decoder stacks,
   replacing BART's post-embedding `layernorm_embedding` (which M2M lacks).
   Named `layer_norm` to match HF keys `model.{encoder,decoder}.layer_norm`.
3. `scale_embedding=True` (embed_scale = sqrt(d_model)) — already handled by the
   scaled word-embedding class.

## VALIDATION TODO (needs a vLLM build + GPU — not run here)
1. **Numeric parity vs HF**: load `facebook/nllb-200-1.3B`, generate
   `spa_Latn -> quy_Latn`, compare token-by-token / ChrF against
   `transformers` greedy + beam. (We have an exact NLLB eval harness in the
   rosettia project to diff against.)
2. **LM-head scaling (likely fix needed)**: the BART base's `*ParallelLMHead`
   divides logits by `embed_scale`. For `scale_embedding=False` (vanilla BART)
   that's a no-op, but M2M/NLLB use `scale_embedding=True`, and HF M2M100 does
   **NOT** rescale logits. Verify and probably set the LM-head `embed_scale=1.0`
   (or drop the division) for M2M.
3. **Weight loading**: confirm no missing/unexpected keys; that
   `embed_positions.weights` (sinusoidal buffer) is correctly ignored; and the
   qkv / cross-attn kv stacking mapping holds for M2M weight names.
4. **forced_bos / target-language token**: confirm the enc-dec prompt path sets
   the target `forced_bos_token_id` (e.g. `quy_Latn`) correctly.
5. **Tokenizer**: NLLB `add_special_tokens` behavior in the decoder prompt.

## Why
Enables fast vLLM serving (and LoRA hot-swap → async RL rollouts) for NLLB-200,
the strongest open low-resource MT backbone. Motivated by the Rosettia
Spanish->Chanka-Quechua project, where NLLB rollouts are currently HF-`generate`
bound.
