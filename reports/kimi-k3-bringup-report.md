# Kimi K3 Vista Bring-up and Validation Report

Date: 2026-08-02 to 2026-08-03
Platform: TACC Vista GH200
Runtime: SGLang on Apptainer

This report preserves the investigation, failed experiments, fixes, and
measurements that were intentionally removed from the operational README.
Results describe this exact image, checkpoint, and Vista topology; they should
not be treated as general Kimi K3 performance claims.

## Executive summary

- The complete production checkpoint contains 96 shards and
  1,560,936,091,448 weight bytes (about 1.56 TB decimal).
- The selected runtime is the ARM64 CUDA-12 K3 image
  `lmsysorg/sglang:kimi-k3-cu12-74968e5653-arm64`, converted to a local SIF.
- The final serving topology is 32 Vista GH200 nodes with one GPU per node,
  TP32/EP32 over 400 Gb/s InfiniBand.
- Production job 884876 loaded the checkpoint, passed warmup and Hello
  generation, completed isolated recall tests through the configured 32K
  context envelope, and served three multi-turn chat requests.
- No OOM, NCCL failure, dropped rank, or GPU sampling failure occurred in the
  final run.
- Job 884876 ran for 44 minutes 1 second and consumed approximately 23.48 Vista
  SUs/node-hours.

## Capacity and topology decision

Kimi K3 has 2.8T total parameters and 104B active parameters. Active parameters
govern work per token; all expert weights still need storage. The native MXFP4
checkpoint is about 1,453.8 GiB.

| Shape | Aggregate nominal HBM | Bring-up assessment |
|---|---:|---|
| 1 GH200 | 96 GiB | Cannot hold the full checkpoint |
| 16 GH200 | 1,536 GiB | Weights leave insufficient runtime/cache headroom |
| 32 GH200 | 3,072 GiB | Practical fit and matches the published TP32/EP32 world size |

SGLang's published H200 lane assumes fewer multi-GPU NVSwitch nodes. Vista has
one GH200 per node, so all TP/EP communication crosses InfiniBand. The world
size fits the model, but decode latency is expected to be worse than the
published NVSwitch topology.

Linux exposes GH200 HBM as a memory-only NUMA node. The roughly 212 GiB shown by
some host-memory tools is Grace LPDDR plus HBM, not an additional 212 GiB of CPU
RAM available to the weight loader.

## Runtime/image investigation

The production SIF is:

```text
images/sglang-kimi-k3-cu12-74968e5653-arm64.sif
```

Recorded contents included SGLang 0.5.16, PyTorch 2.11.0 with CUDA 12.9, and
FlashInfer 0.6.15. The SIF SHA-256 was:

```text
0b97409f235ad8c5fdf21da29407348359e23a8899c9b85ddeeb13860d8f32b3
```

The ARM64 K3 image was selected because Vista's Grace CPU requires ARM64 and
the image includes K3 and Hopper support. Its missing FA3 `flash_ops` extension
was avoided by explicitly selecting FlashInfer for prefill and FlashMLA for
decode.

Other investigated paths were not selected for first bring-up:

- the CUDA-13 ARM64 SGLang K3 image was a less direct match for the site stack;
- TokenSpeed's published NVIDIA K3 recipe targeted eight B300 GPUs and did not
  provide a comparable multi-node Hopper recipe;
- older vLLM SIFs found under the account predated K3 support.

The jobs used the container's CUDA/NCCL/RDMA libraries under `--cleanenv`.
Mixing Vista's older host NCCL module into the image was intentionally avoided.

## Bring-up chronology

### Tiny and transport diagnostics

- Jobs 884391 and 884475 reached a healthy two-node TP2/EP2 HTTP server, but
  their first model forward timed out.
- Job 884475 showed that NCCL symmetric memory resolved the earlier lazy
  PyTorch NCCL communicator initialization problem.
- Job 884506 ran TP2/EP1. Ordinary and registered symmetric-memory PyNccl
  all-reduces passed, both ranks loaded the tiny model, and one token returned
  HTTP 200 in 31.65 seconds. This separated the transport/TP path from the tiny
  checkpoint's EP2 path.
- Job 884517 ran TP2/EP2. It reached the request while rank 0 compiled an
  EP2-specific Marlin MoE kernel and rank 1 waited on the shared TVM-FFI cache
  lock. The client timeout was therefore compilation-related, not an NCCL
  stall.
- After the shared kernel cache was completed, job 884586 reused it (`ninja: no
  work to do`) and TP2/EP2 returned one token with HTTP 200 in 31.42 seconds.

These experiments informed the persistent shared `TVM_FFI_CACHE_DIR`, bounded
timeouts, and production serving configuration. The temporary tiny-model and
NCCL diagnostic tooling was removed after bring-up.

### Production checkpoint download

Job 884613 downloaded the public checkpoint to `models/Kimi-K3`. The transfer
took 27 minutes 37 seconds after the site prolog and finished with all 96
shards, a valid index, and 1,560,936,091,448 weight bytes. `HF_HOME` was retained
as download/tokenizer cache rather than used as the deployment path.

### Production memory tuning

Job 884828 loaded the full model across 32 nodes, completed warmup, and served
short requests. A retained 4K cache followed by an 8K request exhausted a
Marlin MoE workspace when the static-memory fraction was 0.85.

The final profile changed the static-memory fraction to 0.80, bounded chunked
prefill at 4,096 tokens, isolated context-capacity tests with cache flushes, and
added automatic all-node GPU sampling. CUDA graphs remained disabled during
bring-up.

## Final validated configuration

| Setting | Value |
|---|---:|
| SGLang world size | 32 |
| Tensor parallelism | 32 |
| Expert parallelism | 32 |
| Context length | 32,768 |
| Maximum running requests | 16 |
| Chunked prefill | 4,096 |
| Static memory fraction | 0.80 |
| MoE backend | Marlin |
| Symmetric memory | Enabled |
| Prefill / decode attention | FlashInfer / FlashMLA |
| Mamba full-memory ratio | 0.45 |
| Mamba state | BF16 |
| Mamba radix strategy | `extra_buffer_lazy` |
| KV cache | FP8 E4M3 |
| CUDA graphs | Disabled for prefill and decode |

## Context validation from job 884876

Every row below used isolated cache state and exact-token needle recall. All
successful rows returned the marker correctly.

| Prompt tokens | Output budget | Completion tokens | Latency (s) | Result |
|---:|---:|---:|---:|---|
| 4,096 | 128 | 64 | 16.271 | Pass |
| 8,192 | 128 | 64 | 16.736 | Pass |
| 16,384 | 128 | 67 | 20.406 | Pass |
| 24,576 | 128 | 67 | 22.756 | Pass |
| 30,000 | 128 | 67 | 26.028 | Pass |
| 32,000 | 128 | 67 | 25.547 | Pass |
| 32,637 | 128 | 67 | 25.742 | Pass |
| 32,640 | 125 | 67 | 24.964 | Pass |

A 32,640-token prompt with a 128-token output budget was correctly rejected
with HTTP 400 because K3's chat-template guard counted a total of 32,771 tokens.
Reducing the output budget to 125 fit the configured limit. The server therefore
failed safely at the context guard rather than allocating an oversized request.

During the long-context window, the hottest sampled GPU reached 92,548 MiB out
of 97,871 MiB, leaving 5,323 MiB of sampled HBM headroom.

## Multi-turn chat observations from job 884876

The original chat client was non-streaming, so exact client TTFT was not
recorded. Server logs only showed a 6-8 second interval between the prefill log
and the first periodic decode log; this is not a valid end-to-end TTFT. The
current streaming client now records TTFT and time to first answer content for
future jobs.

Approximate request durations observed from server milestones were:

| Turn | Approx. request duration | Stable decode throughput |
|---:|---:|---:|
| 1 | 33 s | about 4.8 tokens/s after initial decode |
| 2 | 18 s | about 4.8 tokens/s |
| 3 | 72 s | about 4.7 tokens/s |

All three requests returned HTTP 200. During active generation, complete
15-second GPU samples averaged about 97% utilization across the 32 GPUs.

Per-GPU sampled HBM remained reserved after each request:

| State | 32-GPU average used | Hottest GPU used | Change from baseline |
|---|---:|---:|---:|
| Before chat | 88,534 MiB | 88,990 MiB | 0 |
| After turn 1 | 88,624 MiB | 89,080 MiB | +90 MiB/GPU |
| After turn 2 | 88,758 MiB | 89,214 MiB | +224 MiB/GPU |
| After turn 3 | 89,712 MiB | 90,168 MiB | +1,178 MiB/GPU |

The retained memory is consistent with radix/Mamba cache and CUDA allocator
reservation; it was not accompanied by an error. It should not be interpreted
as proof that indefinitely many turns are safe. A long-running multi-turn soak
and concurrent-client benchmark have not yet been run.

## Cost accounting

Job 884876 allocated 32 GH nodes from 17:22:46 to 18:06:47, or 2,641 seconds:

```text
32 nodes * 2641 seconds / 3600 = 23.4756 node-hours/SUs
```

The three chat turns occupied about 233 seconds including pauses between turns,
equivalent to about 2.07 allocated node-hours during that window. Vista charges
the full allocation while the GPUs are idle, which motivated cancelling the
server immediately after testing.

## Remaining validation work

- Measure true client TTFT and time to first answer content with the new
  streaming `k3-chat.csv` instrumentation.
- Run a bounded multi-turn/cache soak test before treating the service as
  long-running production infrastructure.
- Benchmark concurrent requests; the final validation used one serial client.
- Quantify the accuracy impact of FP8 KV and BF16 KDA state against a
  higher-precision baseline.
- Benchmark CUDA graphs separately only if sufficient HBM headroom remains.

## References consulted during bring-up

- <https://huggingface.co/moonshotai/Kimi-K3>
- <https://github.com/MoonshotAI/Kimi-K3>
- <https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3>
- <https://hub.docker.com/r/lmsysorg/sglang/tags>
- <https://lightseek.org/tokenspeed/recipes/models#kimi-k3>
- <https://hub.docker.com/r/lightseekorg/tokenspeed-runner/tags>
- <https://docs.tacc.utexas.edu/hpc/vista/>
