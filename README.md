<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo.svg">
  <img src="assets/logo-light.svg" alt="OpenLongTail" width="560">
</picture>
<h2>Generative Scaling of Long-Tail Driving Data</h2>

[Lulin&nbsp;Liu](https://lulinliu.github.io/)<sup>&ast;</sup> · [Nuo&nbsp;Chen](https://nuochen1203.github.io/)<sup>&ast;</sup> · [Yan&nbsp;Wang](https://yanwang.org/) · [Bangya&nbsp;Liu](https://phddirectory.ece.wisc.edu/staff/liu-bangya/) · [Wenyan&nbsp;Cong](https://www.wenyancong.com/) · [Hezhen&nbsp;Hu](https://alexhu.top/)<br>
[Boris&nbsp;Ivanovic](https://research.nvidia.com/labs/avg/author/boris-ivanovic/) · [Hao&nbsp;Wang](https://haohww.github.io/) · [Ziyao&nbsp;Zeng](https://adonis-galaxy.github.io/homepage/) · [Xinyu&nbsp;Gong](https://gongxinyuu.github.io/) · [Yang&nbsp;Zhou](https://engineering.tamu.edu/civil/profiles/zhou-yang.html) · [Zixiang&nbsp;Xiong](https://people.engr.tamu.edu/zixiang-xiong/index.html)<br>
[Dilin&nbsp;Wang](https://wdilin.github.io/) · [Zhangyang&nbsp;Wang](https://vita-group.github.io/) · [Weisong&nbsp;Shi](https://www.cis.udel.edu/people/faculty/weisong-shi/) · [Ruohan&nbsp;Zhang](https://ai.stanford.edu/~zharu/) · [Marco&nbsp;Pavone](https://research.nvidia.com/person/marco-pavone) · [Zhiwen&nbsp;Fan](https://zhiwenfan.github.io/)<sup>†</sup>

Texas A&M University · NVIDIA · University of Wisconsin–Madison · UT Austin · Yale University · Adobe · Meta · University of Delaware · Stanford University

<sup>&ast;</sup>Equal contribution&emsp;<sup>†</sup>Corresponding author

<a href="https://arxiv.org/abs/2607.09655"><img src="https://img.shields.io/badge/arXiv-2607.09655-b31b1b?logo=arxiv&logoColor=white"></a>
<a href="https://openlongtail.github.io/"><img src="https://img.shields.io/badge/Project%20Page-OpenLongTail-1f8acb?logo=googlechrome&logoColor=white"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow"></a>

</div>

> **TL;DR** — From a single <u>front-camera</u> driving video, **OpenLongTail** synthesizes the <u>five missing rig cameras</u> at the <u>same timestamps</u> — turning monocular long-tail clips into **pose-grounded, multi-view training data** for **robust VLA driving policies**.

## TODO

- [ ] **Checkpoint** — *will be released in July 2026*
- [x] **Inference code**
- [x] **Preprocessing & training code**

## Table of Contents

- [Installation](#installation)
- [Inference](#inference)
- [Building Training Data](#building-training-data)
- [Training](#training)
- [Citation](#citation)

## Installation

```bash
# 1. Clone
git clone https://github.com/lulinliu/longt.git openlongtail
cd openlongtail

# 2. Create the environment (Python 3.10, CUDA, bf16)
conda create -n openlongtail python=3.10 -y
conda activate openlongtail

# 3. Install a CUDA-matched PyTorch, then the rest of the deps
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Download the **base backbone** and the **OpenLongTail checkpoint** from Hugging Face in one step:

```bash
bash scripts/download.sh
```

## Inference

Generate the 5 synchronized non-front views from a single front-camera clip. Each clip produces the 5 generated views, the front input, and a preview grid.

```bash
# raw dash-cam clips (front video only)
bash scripts/infer.sh --input path/to/front_clips --output outputs/openlongtail_infer

# already-preprocessed clips
bash scripts/infer.sh --input path/to/cache --output outputs/openlongtail_infer --cached
```

The rear views reuse the front camera's earlier view of the same scene and are steered by an automatically generated scene caption, so the synthesized surround stays consistent with the input. Model and sampler settings can be overridden via environment variables (see the top of [`scripts/infer.sh`](scripts/infer.sh)).

## Building Training Data

Preprocessing turns raw front-camera clips into a ready-to-train multi-view cache in one step: it recovers a metric ego-trajectory for each clip and reprojects the front-view evidence into every target camera (same-time for the side views, temporal look-back for the rear views).

```bash
bash scripts/preprocess.sh --input path/to/raw_clips --output data/openlongtail_cache

# shard across GPUs (run i = 0..N-1 in parallel)
bash scripts/preprocess.sh --input path/to/raw_clips --output data/openlongtail_cache --shards 8 --shard 0

# if your source already provides camera poses, skip pose recovery
bash scripts/preprocess.sh --input path/to/raw_clips --output data/openlongtail_cache --have-poses
```

The resulting cache is consumed directly by training.

## Training

```bash
# Single node (torchrun), 8 GPUs → outputs/openlongtail_train
LATENT_CACHE_ROOT=data/openlongtail_cache OUTPUT_DIR=outputs/openlongtail_train bash scripts/train.sh

# short run on fewer GPUs
NUM_GPUS=4 NUM_STEPS=5000 OUTPUT_DIR=outputs/smoke bash scripts/train.sh

# multi-node (SLURM)
sbatch scripts/train.slurm
```

`train.sh` auto-resumes from the latest checkpoint under the output directory.

## Citation

```bibtex
@misc{liu2026openlongtailgenerativescalinglongtail,
      title={OpenLongTail: Generative Scaling of Long-Tail Driving Data}, 
      author={Lulin Liu and Nuo Chen and Yan Wang and Bangya Liu and Wenyan Cong and Hezhen Hu and Boris Ivanovic and Hao Wang and Ziyao Zeng and Xinyu Gong and Yang Zhou and Zixiang Xiong and Dilin Wang and Zhangyang Wang and Weisong Shi and Ruohan Zhang and Marco Pavone and Zhiwen Fan},
      year={2026},
      eprint={2607.09655},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.09655}, 
}
```

## Acknowledgements

OpenLongTail builds on the following open-source projects:

- [Wan2.1-VACE](https://github.com/Wan-Video/Wan2.1) for the video-diffusion backbone.
- [DepthCrafter](https://github.com/Tencent/DepthCrafter) for video-coherent depth used by the depth warp.
- [MapAnything](https://github.com/facebookresearch/map-anything) for metric camera-pose recovery on in-the-wild clips.
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) for front-view scene captioning.

We thank the authors of these projects for releasing their code and models. Please refer to the corresponding files under `third_party/` for upstream licenses and notices.

## License

Released under the [MIT License](LICENSE); base models remain under their respective licenses.
