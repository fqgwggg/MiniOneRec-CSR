# Attribution and scope

MiniOneRec host framework: https://github.com/AkaliKong/MiniOneRec , fixed commit 526306229326013a7ca46f5b2d8f55ea669aa159. Original source/assets and Apache-2.0 LICENSE retained. Optional backend integration changes evaluate.py, rl.py and minionerec_trainer.py; default reference behavior is preserved. Full datasets, embeddings, weights and caches are excluded.

Conceptual inspiration: Su et al., Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators, https://arxiv.org/abs/2602.22647 ; author implementation https://github.com/youtube/static-constraint-decoding. The CSR backend is a project-specific PyTorch adaptation, not the author code or a full algorithm/performance reproduction.

Our contribution is dynamic catalog constraint handling, CSR integration and the associated correctness/performance evaluation. Frozen existing models are experimental carriers. Public materials do not claim model-training reproduction as a project contribution. Qwen/Amazon assets are not redistributed and have separate terms.
