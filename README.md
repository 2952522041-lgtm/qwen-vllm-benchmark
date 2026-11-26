# Qwen 单卡推理部署与调优

在一张 NVIDIA GPU 上用 vLLM 提供 OpenAI 兼容的对话接口，测量客户端从发出请求到收到首个 token 的时间，以及生成后续 token 的速度。实验比较三种配置，客户端依次维持 1、4、16 个在途请求。

## 运行环境与模型

- Linux 或可使用 NVIDIA GPU 的 WSL2 Ubuntu，Python 3.12、C 编译器及 Python 开发头文件、`uv`、`curl`、`ss`。运行前用 `nvidia-smi` 确认 GPU 0 可见。
- 浮点模型为 [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)，量化模型为[同一模型的官方 AWQ 4-bit 版本](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-AWQ)。模型修订号固定在 `serve.sh`。FP16 指推理时的权重与计算精度。
- 三组共用 55% 的显存预算、512 token 上下文、2048 token 的单轮调度预算、Triton attention 与 eager 模式。关闭前缀缓存，避免重复提示词命中缓存。基线允许每轮最多调度 4 条序列，调参组提高到 16，量化组保持 16。vLLM 从权重配置识别量化格式。

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
  -d '{"model":"qwen-lab","messages":[{"role":"user","content":"用一句中文解释什么是首 token 延迟。"}],"temperature":0,"max_tokens":80}'
```

## 复现实验

先在第一个终端按 Ctrl+C 关闭手动启动的服务，再在已激活的 Python 环境中执行以下命令。

```bash
bash run.sh
python report.py .local/results
```

`run.sh` 为每个配置启动独立服务，等待就绪，执行一次 curl 对话，然后压测并释放 GPU。可用 `bash run.sh tuned` 单独复跑一组。新结果写入 `.local/results/`，运行日志在 `.local/`。离线运行时可将 `MODEL_DIR` 指向包含上述两个模型同名目录的父目录，权重应对应脚本中的修订号。

## 读数口径

- 每个配置、每档并发先预热一轮，再发送相同的 48 条短提示。每个请求固定生成 64 个 token，温度为 0。一个请求结束即补下一个，直到 48 条发完。每档共生成 3,072 个 token。预热、读取监控指标和模型加载耗时均在计时窗口外。
- 流式客户端以 vLLM 返回的生成 token ID 标记首尾事件，跳过仅携带角色或 usage 的块。TTFT 从客户端发请求计到第一个生成 token 的到达时刻，包含网络和服务端排队。每个请求的 TPOT 等于首末生成 token 的到达时间差除以 63。表中 P95 取最近秩。
- 输出吞吐为成功生成的 token 总数除以整批墙钟时间。显存用 NVML 每 200 ms 采样并在批次结束时补采，表中为整张卡已用显存的采样峰值。桌面进程和 vLLM 预留的 KV cache 都占用这一读数。
- 服务端平均排队、prefill 和 decode 时间来自 `/metrics` 中各阶段 histogram 的前后差值。脚本校验每阶段计数增量为 48。原始 JSON 保留逐请求读数、显存时间序列、阶段均值和批次耗时。 [vLLM 指标定义](https://docs.vllm.ai/en/v0.11.2/design/metrics/)

## 实测结果

<!-- measured-results -->

## 面试中怎样解释 PD 分离

Prefill 处理整段输入并建立 KV cache，长输入能并行执行大量矩阵计算。Decode 每步生成一个 token，并反复读取权重和已有 KV cache，小 batch 时常受显存带宽影响。混合服务中，一次长 prefill 可能拖长其他请求的 token 间隔。把两个阶段放到不同实例可以分别配置资源、控制首 token 延迟与后续 token 间隔，同时需要支付 KV cache 传输成本。本项目实测范围是单卡合并部署，适合讨论排队和阶段耗时；验证 PD 分离收益需要增加分离部署对照。 [vLLM 分离式 Prefill 说明](https://docs.vllm.ai/en/v0.11.2/features/disagg_prefill/)
