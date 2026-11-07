# CSR backend integration

The public contribution is dynamic candidate catalogs and CSR decoding. Official source remains as the host framework. `evaluate.py` and `rl.py` accept `--constraint_backend reference|csr`, default reference. The Trainer shares an immutable device index and makes independent stateful processors for rollout and validation. No SFT loss switch is needed.

`csr_decode.CSRLogitsProcessor(paths, eos_token_id, device)` accepts complete output token paths with one final EOS. Paths exclude the prompt but must include all generated response suffix tokens. Match the exact tokenizer and response serialization of the model; do not assume a SID string maps to a single token. No vocabulary resize is performed. Build a fresh processor per generate call/catalog, never share its mutable counter across threads or requests. The reference-entry adapter also verifies the response header.

Only ordinary Transformers autoregressive generation is supported. vLLM with the CSR Trainer backend is rejected, and speculative decoding is not validated. For sharded models, explicitly ensure constraint tensors use the same device as generation scores. Empty catalogs are rejected by the original-entry adapter. Invalid prefixes force EOS and should be counted as incomplete outputs downstream.

Changing the catalog builds a fresh index; the request must pin one catalog throughout generation. Never filter a full-catalog result after generation as a substitute for constrained search. Do not change history or select snapshots based on held-out targets.

Official scripts are not a new turnkey deployment system. Supply model/data paths and preserve the caller's training and generation settings. Six CPU interface tests and the catalog-switch example passed in the current upload check. Real GPU smoke/throughput checks of these packaged entrypoints remain pending.
