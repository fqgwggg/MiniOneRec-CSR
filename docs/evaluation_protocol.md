# Dynamic catalog and decoding evaluation

## Catalog experiment

Use the unique SID universe, seeds 2026/2027/2028 and nested retained sets 90/70/50%, independent of request targets. SID collisions are groups. Histories remain unchanged. Full/subset predictions must have identical request IDs, ordering and targets; reject duplicate IDs. Compare accuracy on exactly the available-target cohort for each snapshot. Keep all original requests for legality and Top-K shortfall. Empty cohorts have undefined metrics, not zero accuracy.

The bundled catalog_tools.py can regenerate this experiment method on caller data. Its generated snapshot should not be claimed identical to historical artifacts without checking the recorded catalog hashes. The included catalog_metrics.json retains per-snapshot identities, denominators and paired metrics. Nonzero illegal outputs make the evaluator fail instead of silently removing predictions.

Historical evidence uses Industrial supplied three-part SID, 3,670 unique SID groups and 4,533 test requests. Frozen SFT/RL/SASRec models are test carriers, not claimed new model contributions. Six SASRec empty histories remain failures; SFT/RL produced Top-50. The public package does not include full predictions, models or datasets. Aggregates reproduce plots, not per-request scores.

## Historical performance evidence

generation_benchmark.json: 48 length-stratified validation requests, batch8, beam50, fixed full catalog, one warmup per backend/block and three repetitions on A100-SXM4-80GB. Token outputs match. The full validation count 4,532 is not the timing sample count. Index construction and tokenization are separate. Synchronized callback profiling is intrusive and is not the exact fraction of uninstrumented generation.

rl_update_benchmark.json and reverse: same late-training checkpoint, three warmups and ten measured full updates; repeat opposite backend order. Prompt/completion/reward/advantage/final parameter equivalence was checked. Excludes validation, save/load; do not claim full-run A/B from a short test.

## Public benchmark for a new workload

Run `examples/benchmark_generation.py --help`. Provide a model path/revision and tokenized JSON with eos_token_id, pad_token_id, paths, input_ids and attention_mask. Paths must use the exact model response protocol. Both backends execute actual model.generate; warm up, alternate order, synchronize timing, compare output tokens. Index build is reported separately; no prediction cache. The dictionary oracle in this public CLI is a new comparison implementation, not necessarily identical to the historical official callback.

CLI parsing/help and CSR CPU semantics are checked; this new packaged GPU CLI has not yet been validated on a real model. Historical measurements are preserved evidence, not a claim that v1.7 reran them. Report hardware, software, input/model/catalog identities, beam, dtype, repeats and measurement boundaries for new results.
