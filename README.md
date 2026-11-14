# chip-router

Pick the AI chip each job runs on, **on purpose**.

New accelerators keep showing up, and default cloud queues will happily put a job on whatever is left. chip-router is a small open-source tool for an AI agent or a developer: it reads a job, compares it to a chip catalog you can extend, and returns one rented GPU or accelerator, a plain-English reason, and the runners-up. It also logs which chip ran, what it was estimated to cost, and why.

It can only pick from chips you would rent. It does not control what Cursor or other AI companies run internally. Provider adapters are **dry-run only**: they print the launch they would make and never rent a GPU, call a paid API, or spend money.

## Quickstart

```bash
python -m pip install -e ".[dev]"
chip-router validate
chip-router list
chip-router pick examples/infer-8b.yaml
chip-router pick examples/infer-70b.yaml --plan
chip-router log
```

A job file looks like this:

```yaml
name: infer-8b
task: inference          # inference | fine-tune | training | signal-processing
precision: fp8           # fp8 | bf16 | fp16 | fp32
model_size_gb: 8         # used with a task headroom factor if vram_gb is omitted
# vram_gb: 16            # optional hard memory floor, wins over the estimate
budget_usd_per_hour: 3.00
budget_usd_per_job: 6.00
max_runtime_hours: 2
preferred_providers: [runpod, modal]
excluded_providers: []
allowed_providers: []    # optional allow-list
optimize: balanced       # cost | performance | balanced
software: any            # any | cuda | rocm | tpu
```

If `vram_gb` is omitted, the router estimates a memory floor as `model_size_gb` times a documented headroom factor: **1.2** for inference and signal-processing, **2.5** for fine-tunes, **4.0** for training. Those multiples are conservative starting points, not a guarantee of fit for a particular framework or batch size. Set `vram_gb` when you know the real requirement.

The library API is the same decision the CLI prints:

```python
from chip_router import pick_chip, load_job

decision = pick_chip(load_job("examples/infer-8b.yaml"))
print(decision.reason)
if decision.choice:
    print(decision.choice.chip.name, decision.choice.offer.provider)
```

Optional MCP server (stdio, exposes `pick_chip`):

```bash
python -m pip install -e ".[mcp]"
chip-router mcp
```

## How the router chooses

The picker is deterministic. Given the same catalog and job, it always returns the same winner.

1. **Hard filters.** The chip must have at least the memory floor, support the requested precision (FP8 jobs skip chips that do not publish FP8 Tensor Cores), match the software stack if you asked for `cuda`, `rocm`, or `tpu`, and have an offer on an allowed provider. Offers over the hourly or per-job budget are dropped.
2. **Score.** `cost` picks the cheapest remaining offer. `performance` picks the best published dense throughput (memory bandwidth for signal-processing). `balanced` (the default) picks the best published performance per dollar. For inference and signal-processing, balanced mode also keeps first rank inside a right-size memory window (`max(required * 3, required + 40 GB)`) so a 192 GB card is not chosen for a 16 GB job just because its efficiency looks high.
3. **Ties.** Preferred providers win ties. Then provider name, chip id, and SKU, so the result is stable.

Every decision includes a reason you can read, the runners-up, and a price footnote that says whether the number is a current public list price, a published future list price, or an example.

## Cost awareness

- Prices in the default catalog are **public on-demand list rates**, each with a retrieval date and a source URL. They are not quotes and they change.
- Tensor throughputs are stored as **dense** (no structured sparsity) TFLOPS when the vendor publishes both, so chips can be compared fairly. Sparse peaks are kept next to them for reference.
- Modal's published rate is per second and GPU-only; the hourly figure is `per_second * 3600`, and Modal also bills CPU and memory.
- Lambda's public instance prices in this catalog **take effect 16 November 2026, 00:00 UTC**. Until then, current Lambda prices are only shown after login.
- Estimated job cost is `usd_per_hour * max_runtime_hours * count`. There is no utilization model yet.
- The decision log (`chip-router-log.jsonl`, or `CHIP_ROUTER_LOG`) stores the job, chip, provider, estimated cost, reason, UTC timestamp, and an optional `actual` field you can fill in later.

## Add a chip

Adding a chip is adding an entry. Edit [`src/chip_router/data/chips.yaml`](src/chip_router/data/chips.yaml) and run `chip-router validate`.

A valid entry needs:

- a unique kebab-case `id`
- memory, memory bandwidth, and vendor **source URLs** (datasheet or official product page) with a retrieval date
- dense and, when published, sparse BF16/FP8 throughput — or `null` if the vendor did not publish that number
- zero or more `offers`, each with `provider`, `sku`, `usd_per_hour`, `pricing_kind`, `retrieved`, and `source_url`

`pricing_kind` must be one of:

| Kind | Meaning |
| --- | --- |
| `public_list` | A current public list price you retrieved from the provider |
| `published_future_list` | A public price that is not in effect yet (`effective_at` is required) |
| `example` | A clearly marked example value, never a pretended real price (`source_note` is required) |

Do not invent a list price. If a provider does not publish one, leave that offer out. The NVIDIA A10G entry is in the catalog with official specs and **no offers**, because RunPod, Modal, and Lambda did not list A10G on the public pages checked for v0.1 (Modal and Lambda list NVIDIA A10, a related but different SKU).

Then add or adjust a test if the new chip should change an example pick, and open a PR.

## Catalog in v0.1

| Chip | Memory | Bandwidth | Dense BF16 / FP8 | Priced on |
| --- | ---: | ---: | --- | --- |
| NVIDIA H100 SXM | 80 GB | 3.35 TB/s | 989.5 / 1979 TFLOPS | RunPod, Modal, Lambda (future) |
| NVIDIA H200 SXM | 141 GB | 4.8 TB/s | 989.5 / 1979 TFLOPS | RunPod, Modal |
| NVIDIA B200 SXM | 180 GB | up to 8 TB/s | unpublished BF16 / 4500 TFLOPS FP8 | RunPod, Modal, Lambda (future) |
| NVIDIA A100 80GB SXM | 80 GB | 2.039 TB/s | 312 / no FP8 | RunPod, Modal, Lambda 8x (future) |
| NVIDIA L40S | 48 GB | 864 GB/s | 362.05 / 733 TFLOPS | RunPod, Modal |
| NVIDIA A10G | 24 GB | 600 GB/s | 70 / no FP8 | *no public offer found* |
| NVIDIA GeForce RTX 4090 | 24 GB | 1008 GB/s | 330.25 / 660.5 TFLOPS (see note) | RunPod |
| AMD Instinct MI300X | 192 GB | 5.3 TB/s | 1307.4 / 2614.9 TFLOPS | RunPod |
| Google TPU v5p | 95 GB usable | 2765 GB/s | 459 / 459 TFLOPS | Google Cloud |

RTX 4090 memory and bandwidth come from NVIDIA's product page. The 1,321 AI TOPS figure is from NVIDIA's 40-series announcement; dense FP8 and BF16 in this catalog are derived from that number (half for 2:4 sparsity, half again for Ada FP8 = 2× BF16) so scoring stays on a dense basis. B200 BF16 is left `null` because NVIDIA's public DGX B200 table publishes FP8 and FP4, not a per-GPU BF16 peak.

Default RunPod prices are **Secure Cloud** on-demand rates from the pricing page updated 12 November 2026. Community Cloud rates are often lower and are not in this catalog.

## CLI

```text
chip-router pick job.yaml [--plan] [--json] [--no-log] [--catalog PATH]
chip-router list [--json]
chip-router log [-n 20] [--json]
chip-router validate [chips.yaml]
chip-router mcp
```

## What v0.1 does not do

- It never launches, rents, or bills anything.
- It does not check live availability or regional stock.
- It does not model multi-node topology, interconnect, or framework-specific KV cache.
- Lambda current (pre-16 November 2026) prices are not in the catalog, because they are not on the public pricing page.
- Google TPU is included as an optional catalog entry; it is not offered on RunPod, Modal, or Lambda.

## License

MIT © 2026 Joshua Almeida
