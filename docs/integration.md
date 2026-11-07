# CSR backend integration

CSR decoding integrates with MiniOneRec and supports selectable candidate catalogs. `evaluate.py` and `rl.py` accept `--constraint_backend reference|csr`, default reference. The Trainer shares an immutable device index and makes independent stateful processors for rollout and validation. No SFT loss switch is needed.

`csr_decode.CSRLogitsProcessor(paths, eos_token_id, device)` accepts complete output token paths with one final EOS. Paths exclude the prompt but must include all generated response suffix tokens. Match the exact tokenizer and response serialization of the model; do not assume a SID string maps to a single token. No vocabulary resize is performed. Build a fresh processor per generate call/catalog, never share its mutable counter across threads or requests. The reference-entry adapter also verifies the response header.

Only ordinary Transformers autoregressive generation is supported. vLLM with the CSR Trainer backend is rejected, and speculative decoding is not validated. For sharded models, explicitly ensure constraint tensors use the same device as generation scores. Empty catalogs are rejected by the original-entry adapter. Invalid prefixes force EOS and should be counted as incomplete outputs downstream.

Changing the catalog builds a fresh index; the request must pin one catalog throughout generation. Never filter a full-catalog result after generation as a substitute for constrained search. Do not change history or select snapshots based on held-out targets.

Configure model/data paths and training/generation parameters for the target workload. Test coverage includes six CPU interface tests and the catalog-switch example. Real-model GPU validation of the public entrypoints is not yet available; the reported performance measurements use a separate benchmark implementation.
