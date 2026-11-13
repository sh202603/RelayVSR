<h1 align="center">RelayVSR</h1>

<h3 align="center">Large-Small Model Collaboration for Efficient Real-World Video Super-Resolution</h3>

<p align="center">
  Xijun Wang · Xin Li · Zirui Lang · Suhang Yao · Haoran Li · Zhibo Chen
</p>

<p align="center">
  <a href="https://kopperx.github.io/RelayVSR/">Project Page</a> ·
  <a href="https://arxiv.org/abs/2609.37850">Paper</a> ·
  <a href="https://huggingface.co/kopper/RelayVSR">Model Weights</a>
</p>

RelayVSR performs collaborative streaming 4× video super-resolution with sparse
Wan references and a compact conditional decoder. This repository contains
inference code and the [project website source](docs/).

## About this fork

This is a fork of [kopperx/RelayVSR](https://github.com/kopperx/RelayVSR) that
runs on Windows and fits 4K output into 16 GB of VRAM. With the default options
the frames are identical to upstream.

- **Windows 11 support**: the same setup and commands as on Linux.
- **960×540 → 3840×2160 on a 16 GB GPU**: the Wan K/V cache can live in host
  memory, which upstream keeps on the GPU (about 25 GB in total at this size).
  See [Memory](#memory).
- **Optional FP8 weights and int8 cache**: less VRAM and faster keyframes, with
  a slightly different output.
- **Audio pass-through**: the input audio is copied without re-encoding.
- **Resume**: an interrupted run continues from the last finished part. See
  [Long videos](#long-videos).
- **Multi-threaded encoding** with `--encode-threads`, and per-stage timings
  and peak VRAM printed after every run.

On an RTX 5080, 960×540 → 4K runs at about 4.6 fps with
`--wan-weights fp8 --kv-cache gpu --kv-cache-dtype int8 --window-size 32 --encode-threads 0`.

![RelayVSR overview from the paper](assets/intro-figure.png)

## Demo

The input starts on the left and the 4× output on the right. The divider moves
from right to left until the output fills the frame.
The watermark strip is cropped from both views. The output area is not resized;
the MP4 is 3544×1832 at 24 fps with all 265 frames.

https://github.com/user-attachments/assets/937a6084-3a59-4ccb-9a9c-6f0462693186

[Download the MP4](assets/demo/relayvsr-comparison.mp4)

## Setup

Linux or Windows 11, Python 3.11–3.13 and an NVIDIA GPU:

```bash
uv sync --locked --extra cu128
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
```

On Windows the Triton kernels come from `triton-windows`, which is installed
automatically.

## Inference

Download the two checkpoints from the
[Hugging Face repository](https://huggingface.co/kopper/RelayVSR) into `weights/`:

```bash
uvx hf download kopper/RelayVSR flash.safetensors wan.safetensors --local-dir weights
```

The expected paths are `weights/wan.safetensors` and
`weights/flash.safetensors`, resolved relative to `inference.py`.
The bundled `assets/demo/input.mp4` is used when `--input` is omitted.

```bash
python inference.py \
  --output output_x4.mp4
```

Pass `--input path/to/video.mp4` to process another video, or provide an
ordered image directory together with `--fps`.

The default Flash checkpoint is **S, step 60000, EMA**, trained with a two-frame
video attention window (current frame and one predecessor) and current-relative
temporal RoPE. Override either path with `--wan-checkpoint` or `--flash-checkpoint`.

- `--window-size 16`: generate reference endpoints every 15 frames. Increase
  this value to let the small model process more intermediate frames.
- `--fps 24`: override the input FPS; required for image directories.
- `--max-frames 100`: process at most 100 frames.
- `--dtype bf16`: inference precision; `fp16` and `fp32` are also supported.
- `--encode-threads 0`: x264 threads; 0 lets the encoder choose. The default
  of 1 is the slowest but matches upstream; at 4K output, more threads roughly
  double the overall speed. The encoded bitstream depends on the thread count.
- `--overwrite`: replace an existing output.
- `--no-audio`: do not copy the input audio.

Video FPS is preserved. Output resolution is exactly 4× the input. The input
audio is copied without re-encoding. Each model loads its own safetensors directly, without runtime adapter
merging. Both files contain their model configuration; the Wan file also contains
the prompt embeddings.

## Memory

A 16 GB GPU can produce 3840×2160 from 960×540. The run prints per-stage
timings and peak VRAM when it ends.

- `--kv-cache auto|gpu|cpu`: where the Wan K/V cache lives. `cpu` keeps it in
  host memory (about 12 GB of RAM at 4K output) and moves one layer at a time to
  the GPU. The output is identical; each keyframe takes longer. `auto`, the
  default, uses the GPU only when the cache fits.
- `--kv-cache-dtype int8`: store the cache as int8, halving it.
- `--wan-weights fp8`: run the Wan block linears in FP8, roughly halving the
  weights on the GPU and speeding up each keyframe. Needs an RTX 40 series or
  newer GPU and `--dtype bf16`; otherwise it falls back to bf16 with a warning.

The last two change the output slightly, so they are off by default. On a 16 GB
GPU at 4K output, `--wan-weights fp8 --kv-cache gpu --kv-cache-dtype int8` is the
fastest combination and `--wan-weights fp8 --kv-cache cpu` uses the least VRAM.

## Long videos

`--resume` writes the video as parts into `<output>.parts/` and saves the model
state after each part. If the run is interrupted, repeat the same command: the
finished parts are reused and the result has the same frames as an
uninterrupted `--resume` run. `--resume-interval 600` sets the approximate part
length in frames. Each saved state is as large as the Wan K/V cache (about
12 GB at 4K output, half with `--kv-cache-dtype int8`); only the latest is kept.
Changing the input, the checkpoints or an option that affects the pixels
discards the saved parts.

## Citation

```bibtex
@misc{wang2026relayvsr,
  title={RelayVSR: Large-Small Model Collaboration for Efficient Real-World Video Super-Resolution},
  author={Xijun Wang and Xin Li and Zirui Lang and Suhang Yao and Haoran Li and Zhibo Chen},
  year={2026},
  eprint={2609.37850},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2609.37850}
}
```

## License

Apache-2.0. The Wan implementation retains its upstream copyright notice.
Acknowledgments: [Diffusers](https://github.com/huggingface/diffusers),
[Wan](https://github.com/Wan-Video/Wan2.2),
[FlashVSR](https://github.com/OpenImagingLab/FlashVSR) and
[flashvsr-sm89-ops](https://github.com/aireet/flashvsr-sm89-ops) (FP8 kernels).
