# Qwen3-4B 单卡推理部署与调优

在一张 NVIDIA GPU 上用 vLLM 提供 OpenAI 兼容的对话接口，测量客户端从发出请求到收到首个 token 的时间，以及生成后续 token 的速度。实验比较三种配置，客户端依次维持 1、4、16 个在途请求。

## 实测结果

2026-11-27 在 NVIDIA GeForce RTX 5070 Laptop GPU（8,151 MiB）、WSL2 Ubuntu 24.04、驱动 595.97 上测得。使用 vLLM 0.11.2、PyTorch 2.9.0+cu128。每格对应一次 32 请求批次，输入均为 38 tokens，输出固定为 32 tokens。

**瓶颈在服务端排队。** 16 并发时，基线每轮最多调度 4 条序列，平均首 token 延迟 1,815.3 ms，其中排队占 1,762.1 ms（97%）。将上限提高到 16 后，排队降至 2.1 ms，输出吞吐从 161.2 升至 595.0 tokens/s（3.69 倍）。

### 基线 · FP8，最多 4 条序列

| 并发 | TTFT 均值 / P95 (ms) | TPOT 均值 (ms/token) | 输出吞吐 (tokens/s) | 整卡峰值 (MiB) | 引擎排队均值 (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 30.4 / 33.6 | 23.72 | 41.8 | 7481 | 0.0 |
| 4 | 48.2 / 51.3 | 24.17 | 159.9 | 7479 | 0.0 |
| 16 | 1815.3 / 2416.9 | 24.69 | 161.2 | 7479 | 1762.1 |

### 调参后 · FP8，最多 16 条序列

| 并发 | TTFT 均值 / P95 (ms) | TPOT 均值 (ms/token) | 输出吞吐 (tokens/s) | 整卡峰值 (MiB) | 引擎排队均值 (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 30.1 / 33.0 | 23.95 | 41.4 | 7494 | 0.0 |
| 4 | 47.6 / 55.6 | 23.76 | 163.1 | 7494 | 0.0 |
| 16 | 99.6 / 131.8 | 24.18 | 595.0 | 7493 | 2.1 |

### 量化后 · AWQ 4-bit，最多 16 条序列

| 并发 | TTFT 均值 / P95 (ms) | TPOT 均值 (ms/token) | 输出吞吐 (tokens/s) | 整卡峰值 (MiB) | 引擎排队均值 (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 25.1 / 30.5 | 20.17 | 49.2 | 7494 | 0.0 |
| 4 | 49.9 / 57.3 | 20.78 | 184.2 | 7498 | 0.0 |
| 16 | 123.7 / 166.7 | 19.21 | 703.4 | 7500 | 3.3 |

16 并发下，AWQ 的输出吞吐为 703.4 tokens/s，平均 TTFT 是 123.7 ms（FP8 为 99.6 ms）。与调参后的 FP8 相比，模型加载占用从 4.18 降至 2.51 GiB，vLLM 为 KV cache 分配的容量从 11,392 增至 23,520 tokens。三组固定了相同的显存预算，因此量化省下的空间主要留给了 KV cache。 [FP8 引擎记录](results/tuned-engine.txt) · [AWQ 引擎记录](results/awq-engine.txt)

运行 `python report.py results` 可从提交的原始记录重算三张表。 [逐请求数据与 curl 回包](results/)

## 运行环境与模型

- Linux 或可使用 NVIDIA GPU 的 WSL2 Ubuntu，Python 3.12、C 编译器及 Python 开发头文件、`uv`、`curl`、`ss`。运行前用 `nvidia-smi` 确认 GPU 0 可见。
- 基线和调参使用官方 [Qwen3-4B-FP8](https://huggingface.co/Qwen/Qwen3-4B-FP8)，再切换至同系列的官方 [Qwen3-4B-AWQ](https://huggingface.co/Qwen/Qwen3-4B-AWQ)。模型修订号固定在 `serve.sh`。原版 16-bit 权重文件约 8.04 GB，这里选用能完整放入笔记本显卡的两个版本。
- 三组共用 74% 的显存预算、512 token 上下文、512 token 的单轮调度预算、Triton attention 与 eager 模式。关闭前缀缓存，避免重复提示词命中缓存。基线允许每轮最多调度 4 条序列，调参组提高到 16，量化组保持 16。三组权重均驻留 GPU。FP8 到 AWQ 的差异包含权重格式和对应算子的变化。vLLM 从权重配置识别量化格式。

## 先聊一句

在仓库根目录运行以下命令。首次启动会下载公开模型。本地服务监听 `127.0.0.1:8017`，无需 API Key。

```bash
uv venv --managed-python --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
bash serve.sh baseline
```

在同一环境的第二个终端运行 `bash smoke.sh`，即可看到完整 JSON。脚本实际发送的请求如下。

```bash
curl -sS http://127.0.0.1:8017/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-lab","messages":[{"role":"user","content":"用一句中文解释什么是首 token 延迟。"}],"temperature":0,"max_tokens":80,"chat_template_kwargs":{"enable_thinking":false}}'
```

## 复现实验

先在第一个终端按 Ctrl+C 关闭手动启动的服务，再在已激活的 Python 环境中执行以下命令。

```bash
bash run.sh
python report.py .local/results
```

`run.sh` 为每个配置启动独立服务，等待就绪，执行一次 curl 对话，然后压测并释放 GPU。可用 `bash run.sh tuned` 单独复跑一组。新结果写入 `.local/results/`，运行日志在 `.local/`。离线运行时可将 `MODEL_DIR` 指向包含上述两个模型同名目录的父目录，权重应对应脚本中的修订号。

## 读数口径

30 行的 `bench.py` 负责有界并发和逐请求计时，`experiment.py` 负责 SSE 解析、预热、监控与记录。

- 每个配置、每档并发先预热一轮，再发送相同的 32 条短提示。每个请求固定生成 32 个 token，温度为 0，关闭 Qwen3 思考模式。一个请求结束即补下一个，直到 32 条发完。每档共生成 1,024 个 token。预热、读取监控指标和模型加载耗时均在计时窗口外。
- 流式客户端以 vLLM 返回的生成 token ID 标记首尾事件，跳过仅携带角色或 usage 的块。TTFT 从客户端发请求计到第一个生成 token 的到达时刻，包含网络和服务端排队。每个请求的 TPOT 等于首末生成 token 的到达时间差除以 31。表中 P95 取最近秩。
- 输出吞吐为成功生成的 token 总数除以整批墙钟时间。显存用 NVML 每 200 ms 采样并在批次结束时补采，表中为整张卡已用显存的采样峰值。桌面进程和 vLLM 预留的 KV cache 都占用这一读数。
- 服务端平均排队、prefill 和 decode 时间来自 `/metrics` 中各阶段 histogram 的前后差值。脚本校验每阶段计数增量为 32。原始 JSON 保留逐请求读数、显存时间序列、阶段均值和批次耗时。 [vLLM 指标定义](https://docs.vllm.ai/en/v0.11.2/design/metrics/)

## 面试中怎样解释 PD 分离

Prefill 处理整段输入并建立 KV cache，长输入能并行执行大量矩阵计算。Decode 每步生成一个 token，并反复读取权重和已有 KV cache，小 batch 时常受显存带宽影响。混合服务中，一次长 prefill 可能拖长其他请求的 token 间隔。把两个阶段放到不同实例可以分别配置资源、控制首 token 延迟与后续 token 间隔，同时需要支付 KV cache 传输成本。本项目实测范围是单卡合并部署，适合讨论排队和阶段耗时。验证 PD 分离收益需要增加分离部署对照。 [vLLM 分离式 Prefill 说明](https://docs.vllm.ai/en/v0.11.2/features/disagg_prefill/)
