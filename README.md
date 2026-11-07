# MiniOneRec：动态目录与 CSR 约束解码

为 MiniOneRec 增加可切换目录的约束解码后端：在生成过程中限制合法 SID，采用 CSR 数组和批量张量操作代替逐候选前缀查询。

<strong>项目贡献聚焦两点：动态推荐目录、CSR 张量化约束解码。</strong> 保留官方目录和训练入口，提供参数开关、独立接口、可运行示例及针对这两项功能的实验结果。模型与基线用于评估约束正确性和性能收益。

核心思想受 [STATIC 论文](https://arxiv.org/abs/2602.22647)启发，本项目将其稀疏状态转移思路适配到 MiniOneRec / PyTorch。原方法与实现可参阅 [STATIC 作者仓库](https://github.com/youtube/static-constraint-decoding)。

## 1. 实现原理

MiniOneRec 通过逐 token 生成商品的语义标识符（SID）完成推荐。为保证结果对应合法商品，官方参考实现建立“前缀 → 合法下一 token”的映射，在束搜索中逐条查询候选前缀，并屏蔽非法 token。

这种方式需要逐束执行 Python 查询；当前序列位于 GPU 时，将前缀转换为列表还可能引入设备同步和数据传输，增加约束处理开销。

受 STATIC 启发，本项目采用 **CSR 张量化约束解码**：先在 CPU 上合并合法 SID 路径的公共前缀，再按节点排列出边，构建三个数组：

| 数组 | 作用 |
|---|---|
| `row_ptr` | 定位每个节点的出边区间 |
| `token_ids` | 保存出边对应的合法 token |
| `next_nodes` | 保存选择 token 后到达的节点 |

如下图所示，前缀 `[1]` 对应节点 1，其出边区间为 `[2,4)`。读取 `token_ids[2:4]` 即得到合法下一 token `{3,4}`，对应的后继节点由 `next_nodes[2:4]` 给出。

![CSR矩阵构建：合法路径、前缀树、状态转移矩阵、CSR数组与GPU分数掩码](assets/csr_matrix_construction.png)

生成时，后端在 GPU 上批量查询多个候选束的转移关系，再统一构造掩码：

$$
\widetilde{\ell}_v =
\begin{cases}
\ell_v, & v \in \mathcal{A}_c(p),\\
-\infty, & \text{其他}.
\end{cases}
$$

其中，ℓ<sub>v</sub> 为模型的 log-softmax 分数，𝒜<sub>c</sub>(p) 为目录 <i>c</i> 下前缀 <i>p</i> 的合法下一 token 集合。

该实现保持合法集合和评分规则一致，将逐束 Python 查询转换为批量张量操作，减少主机侧调度与部分同步开销，使约束计算更适合 GPU。目录索引可重复使用；每步解码仍包含前缀遍历与稠密掩码构造。性能实验分别测量约束模块、完整生成和 RL 更新的耗时，呈现优化在不同计算范围内的实际收益。

<strong>动态目录用于验证 CSR 约束解码的正确性、适用性与性能。</strong> 实验固定模型和用户历史，切换允许推荐的 SID 子集，并为各目录构建独立索引；检查不可用商品的屏蔽情况，对比两种后端的输出与耗时，同时在目标仍可用的相同请求上配对计算推荐指标，分析候选目录变化的影响。

本项目将 STATIC 的稀疏状态转移思路适配到 MiniOneRec/PyTorch 的束搜索流程，提供可切换的 CSR 约束解码接口。

## 2. 已取得的效果

### 分层耗时对照

在同一模型、输入和合法目录下，对比参考后端与 CSR 后端，并检查生成输出一致性。

![CSR约束解码的分层耗时对照](assets/results/performance.png)

| 测量范围 | 参考约束 | CSR 约束 | 耗时降低 |
|---|---:|---:|---:|
| 约束模块（独立计时） | 933.68 ms | 107.09 ms | 88.53%（约8.72×） |
| 完整生成（beam 50，聚合耗时） | 3.32046 s | 2.49001 s | 25.01% |
| 完整 RL 更新（10次均值） | 8.33675 s/update | 8.18524 s/update | 1.82% |

前两项基于48条代表性请求，分别测量约束模块耗时与完整生成耗时；模块计时在调用前后执行 GPU 同步；RL 更新取预热后10次均值，交换运行顺序复测后收益为 **1.70%～1.82%**。完整生成和 RL 更新还包含模型计算等步骤，因此分别报告三个层级的收益。

### 不同目录规模下的表现

固定 SFT、RL 模型，将合法 SID 目录按三个种子保留90%、70%、50%，与完整目录一起进行后端对照。每份目录使用相同48条代表性请求，batch=8、beam=50；每批预热后，各后端重复5次。

![不同目录规模下的CSR生成延迟与索引成本](assets/results/catalog_performance.png)

在 **1835～3670个 SID** 范围内，CSR 完整生成耗时降低 **23.7%～24.3%**，20组对照的生成 token 均一致。索引占用约 **153～295 KiB** GPU 空间；CPU 构建成本随目录增大，通过复用索引摊销。

完整测试预测中，每个子目录的4533条请求均参与约束检查，SFT/RL 未输出不可用 SID，并返回完整 Top-50。计时与一致性口径见[评估协议](docs/evaluation_protocol.md)。

## 3. 代码结构

| 模块 | 职责 |
|---|---|
| `evaluate.py`、`rl.py`、`minionerec_trainer.py` | 原入口接入、rollout/验证处理器与索引复用 |
| `constraint_backend.py`、`csr_decode.py` | 后端选择、前缀检查与独立 `CSRLogitsProcessor` 接口 |
| `src/decoding/` | CSR 索引、张量查询、分数掩码与参考后端 |
| `catalog_tools.py` | 嵌套目录快照与配对指标 |
| `examples/`、`tests/` | 接入示例、生成 A/B 与功能测试 |

基于 MiniOneRec 提交 `526306229326013a7ca46f5b2d8f55ea669aa159`，保留上游 SFT、RQ 与数据处理流程。参见[上游说明](docs/upstream_readme.md)。

## 4. 依赖与接入

依赖见 `requirements-csr.txt`（Transformers 4.57.1）；上游训练依赖见 `requirements.txt`。

`evaluate.py`、`rl.py` 通过 `--constraint_backend csr` 启用 CSR，默认 `reference`。训练与解码参数沿用现有配置；当前接入 Transformers 常规生成路径，vLLM 路径会触发兼容性错误。

独立使用：

```python
from transformers import LogitsProcessorList
from csr_decode import CSRLogitsProcessor

processor = CSRLogitsProcessor(paths, tokenizer.eos_token_id, device=model.device)
outputs = model.generate(
    **inputs, **generation_kwargs,
    logits_processor=LogitsProcessorList([processor]),
)
```

`paths` 为响应 token 路径，包含响应结束符及最终 EOS；模型与目录使用一致的 tokenizer。索引可复用，每次生成使用独立处理器。详见[接入说明](docs/integration.md)。

功能检查：

```bash
python -m unittest tests.test_csr_decode tests.test_backend_selection
```

## 5. 目录与评估接口

`catalog_tools.py create` 按固定种子生成 90%/70%/50% 嵌套 SID 快照；`evaluate` 对完整目录与子目录预测进行配对评估。

- **目录粒度**：唯一 SID；碰撞商品按 SID 组处理。每份快照重建索引，请求期间固定目录版本，保留用户历史。
- **输入约定**：SID 字符串列表；预测记录为 `{request_id, output, predict}`，两组预测的请求顺序与目标一致。
- **评估口径**：全部请求检查约束与 Top-K 完整性，目标可用的同一请求集合计算配对 HR/NDCG。

参数示例见 `examples/catalog_demo.py`，数据与计时约定见[评估协议](docs/evaluation_protocol.md)。

## 6. 验证与来源

测试覆盖 CPU 接口语义、参数传递与目录切换；公共入口和基准 CLI 尚无真实模型 GPU 验证结果。性能结果来自独立服务器基准实验，测量条件见[评估协议](docs/evaluation_protocol.md)。`results/` 保存汇总数据；逐请求指标复算需提供预测文件。

保留 MiniOneRec 上游代码与许可，CSR 约束解码借鉴 STATIC 的稀疏转移思路。代码与资源归属见[第三方说明](THIRD_PARTY_NOTICES.md)。
