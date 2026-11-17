# ASGD schedule repair on pretrained Qwen3.5-9B-Base

This is an additional model-level verification of the existing ASGD historical repair. It does not establish a new defect in the installed Torch runtime.

## Workload

The complete pretrained text model has 8,953,803,264 parameters and 32 layers. Its checkpoint revision is `68c46c4b3498877f3ef123c856ecfde50c39f404`. Each branch executes three LoRA SFT steps on natural-language text, with BF16 base weights and 557,056 trainable FP32 adapter parameters. The model configuration and layers are unchanged. The runtime is Torch 2.9.0+cu126, Transformers 5.3.0 and PEFT 0.20.0 on the user-confirmed H200 platform (runtime name NVIDIA H20Z).

## Defect and patch

Unmodified PyTorch v2.3.0 ASGD source is the affected implementation. Its foreach path places `alpha` on `step` instead of on the whole learning-rate denominator. The candidate changes that single expression to `lr / ((1 + lambd * lr * step) ** alpha)`. The original single-tensor implementation supplies a native control; an independent NumPy recurrence checks updates at each branch's actual input state.

Settings are `lr=0.2`, `lambd=0.01`, `alpha=0.75`, `t0=1`, no weight decay and no dropout. The source hash and upstream landed fix are pinned in the JSON evidence. The installed libraries and checkpoint files remain unchanged.

## Results

| Measurement against native control | Affected | Candidate |
|---|---:|---:|
| Maximum stored-parameter error | 0.00032670795917510986 | 2.7939677238464355e-9 |
| Maximum stored-update error | 0.0003258325159549713 | 1.862645149230957e-9 |
| Maximum gradient error over the training trajectory | 0.0015920773148536682 | 7.450580596923828e-9 |
| Step 3 training-loss absolute error | 0.020531892776489258 | 0 |
| Combined array RMSE | 4.109886592512926e-6 | 3.058415570085951e-10 |

Each comparison covers 384 arrays and 6,684,672 elements across all trainable tensors, three steps, and parameter/gradient/update/averaged-state captures. All values are finite. Original and candidate foreach dispatch each execute three times; the control executes the original single-tensor path three times.

The same-state recurrence error reaches 8.72e-6 before repair and at most 1.33e-9 afterward. The regression fails on the original foreach source and passes on the candidate. Final evaluation target scores for all three texts match the control exactly after repair. Full array and optimizer-state snapshots stay in the native run; the JSON contains their paths, collected metrics, execution receipt and verifier code.
