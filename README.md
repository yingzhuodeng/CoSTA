# CoSTA: Compressed Speech Token Adaptation for Efficient Spoken Dialogue Modeling

<div align="center">
<br>
<a>Yingzhuo Deng</a><sup><span>1,2</span></sup>,
<a>Chong Li</a><sup><span>1,2</span></sup>,
<a>Bingxin Liu</a><sup><span>1,2</span></sup>,
<a>Tianyu Peng</a><sup><span>1,2</span></sup>,
<a>Yi Shu</a><sup><span>2</span></sup>,
<br>
<a>Can Xie</a><sup><span>1,2</span></sup>,
<a>Siyao Song</a><sup><span>1,2</span></sup>,
<a>Shuo Ren</a><sup><span>1</span></sup>,
<a>Jiajun Zhang</a><sup><span>1,2,3*</span></sup>
<br>

\* Corresponding author

<sup>1</sup> Institute of Automation, Chinese Academy of Sciences<br>
<sup>2</sup> School of Artificial Intelligence, University of Chinese Academy of Sciences<br>
<sup>3</sup> Wuhan AI Research<br>

![Spoken-Dialogue-Modeling](https://img.shields.io/badge/Task-Spoken--Dialogue--Modeling-red)
![Speech-Token-Compression](https://img.shields.io/badge/Method-Speech--Token--Compression-blue)
<a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/Paper-arXiv-red"></a>
</div>

## Overview

We introduce **CoSTA** (**Co**mpressed **S**peech **T**oken **A**daptation), a practical framework for improving the inference efficiency of spoken dialogue models (SDMs). Existing SDMs often generate long and redundant discrete speech-token sequences, which increases autoregressive generation cost, slows streaming response, and limits the number of sustainable interaction turns under a fixed generation budget.

CoSTA adapts an existing SDM to generate **losslessly compressed speech tokens** while keeping the original speech tokenizer and speech decoder unchanged. It applies byte-pair encoding (BPE) to speech-token sequences, extends the model vocabulary with merged speech tokens, and uses pronunciation adaptation followed by dialogue adaptation to teach the model to generate compact speech-token sequences. During inference, the generated compressed tokens are deterministically expanded back to the original token space before waveform synthesis.

Experiments on bilingual spoken-dialogue benchmarks show that CoSTA substantially improves speech-token generation efficiency while preserving response quality, achieving **172.43%** and **165.94%** relative expanded speech-token generation speed with **66.81%** and **70.17%** compression rates on Chinese and English benchmarks, respectively. Further analyses show that CoSTA reduces first-packet latency and improves multi-turn sustainability under a limited generation budget.

## 🔥 Update

- [23/06/2026] CoSTA is released.
