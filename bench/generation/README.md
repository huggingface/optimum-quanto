# Quanto generation benchmark

This script compares the latency per generated token for a specific model for several quantization configurations and implementations.

```
usage: benchmark.py [-h] [--model MODEL] [--device DEVICE] [--it IT] [--quantization {bnb_4bit,bnb_8bit,w8a16,w8a8}]

Generate bechmark

options:
  -h, --help            show this help message and exit
  --model MODEL         The model to use for benchmark
  --device DEVICE       The device to use for benchmark.
  --it IT               The number of benchmark iterations
  --quantization {bnb_4bit,bnb_8bit,w8a16,w8a8}
                        One of none, bnb_4bit, bnb_8bit, w8a16, w8a8.
```

Device: NVIDIA A10G (24Gb memory)
quanto: 0.0.10

| model                            | fp16  | w8a16 | w8a8   | bnb 8bit | bnb 4bit |
|----------------------------------|-------|-------|--------|----------|----------|
| princeton-nlp/Sheared-LLaMA-1.3B | 21 ms | 39 ms | 79 ms  | 115 ms   | 33 ms    |
| 01-ai/Yi-6B                      | 32 ms | 81 ms | 113 ms | 190 ms   | 44 ms    |
| NousResearch/Llama-2-7b-hf       | 37 ms | 93 ms | 107 ms | 164 ms   | 42 ms    |
| HuggingFaceH4/zephyr-7b-beta     | 38 ms | 98 ms | 118 ms | 207 ms   | 45 ms    |

At a quick glance, we can see that:

- the w8a16 latency per-token is quite far from the reference float16 latency,
- the w8a8 latency is not very far from the w8a16 latency, probably thanks to
the accelerated integer matmul (`torch._int_mm`).
- both quanto configurations are much faster than LLM.int8, but much slower than LLM.int4.

Note that quanto does not include (yet) any optimized kernels, and uses only vanilla
pytorch operations. There is therefore plenty of room for improvement.
