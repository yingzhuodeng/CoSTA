# Third-party code

`Step-Audio2/` contains code and demonstration assets from [stepfun-ai/Step-Audio2](https://github.com/stepfun-ai/Step-Audio2), including CosyVoice and FlashCosyVoice components. Its [Apache-2.0 license](Step-Audio2/LICENSE) and source notices are retained. CoSTA adds the batch inference entry points and configurable model/decoder paths.

`train/step_audio2_model/` contains the Step-Audio 2 mini model/configuration definitions used for CoSTA's validated experiments, based on [stepfun-ai/Step-Audio-2-mini](https://huggingface.co/stepfun-ai/Step-Audio-2-mini). They include training-loss support, embedding and LM-head resizing interfaces, and inference timing accumulation. The [upstream license](train/step_audio2_model/LICENSE) is retained.

`train/my_trainer.py` adapts Hugging Face Transformers' Apache-2.0-licensed Trainer implementation, including gradient accumulation and training-step bookkeeping. The upstream license is included as [train/TRANSFORMERS_LICENSE](train/TRANSFORMERS_LICENSE).

These upstream licenses apply to the corresponding third-party components. No repository-wide license for the original CoSTA code has been selected in this release preparation.
