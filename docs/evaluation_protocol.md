# Catalog constraints and decoding evaluation

## Catalog protocol

Use unique SID groups with seeds 2026/2027/2028 and nested retention ratios of 90/70/50%, sampled independently of request targets. Preserve user histories. Full/subset predictions must have identical request IDs, ordering and targets. Compare accuracy on the same available-target cohort; retain all original requests for legality and Top-K completeness. Empty cohorts have undefined accuracy.

`catalog_tools.py` applies this protocol to caller data. Match catalog hashes when comparing experiments. `catalog_metrics.json` records snapshot identities, denominators and paired metrics. The evaluator rejects duplicate request IDs and illegal outputs.

The experiment uses Industrial three-part SIDs: 3,670 unique groups and 4,533 test requests. Fixed SFT/RL/SASRec models provide recommendation outputs. Six SASRec empty-history requests count as failures; SFT/RL return Top-50. Included aggregates support summary analysis; per-request recalculation requires prediction files.

## Performance measurements

`generation_benchmark.json`: 48 validation requests stratified by input length, batch 8, beam 50, fixed full catalog; one warmup per backend/block and three repetitions on NVIDIA A100-SXM4-80GB. Generated tokens match. The full validation set contains 4,532 requests. Index construction and tokenization are measured separately. Constraint-module timing synchronizes the GPU around each callback; complete-generation timing uses a separate measurement run. These measurements quantify backend differences at their respective scopes.

`rl_update_benchmark.json` and `rl_update_benchmark_reverse.json`: both backends start from the same trained model state, with three warmups and ten measured updates. The second experiment reverses backend order. Sampling, rewards, advantages and final policy parameters match across all checked distributed processes. Timing includes matching evidence-hash operations and excludes validation, checkpoint saving and loading. The reported reduction describes this bounded update benchmark.

The catalog-size experiment uses a single A100-SXM4-80GB, fixed SFT/RL models, and 1,835–3,670 SID groups. Each catalog uses the same 48 representative validation requests, batch 8 and beam 50; each batch has one warmup and five timed repetitions per backend. P50/P95 summarize 30 batch timings per backend/catalog. Index construction and device setup use five repetitions and report the median. All 20 model/catalog comparisons have identical generated tokens.

## Public benchmark interface

`examples/benchmark_generation.py` accepts a model path/revision and tokenized JSON containing eos_token_id, pad_token_id, paths, input_ids and attention_mask. Paths must follow the model response protocol. Both backends run model.generate, alternate measurement order, synchronize timing and compare output tokens. Index construction is measured separately; generation executes without prediction caching. This CLI uses its own dictionary reference backend; the reported server measurements use the official callback adapter.

CPU semantics, CLI argument parsing and parameter forwarding are tested. Real-model GPU validation of the public CLI is not yet available. Record hardware, software, model/input/catalog identities, beam, dtype, repeats and timing boundaries with each benchmark.
