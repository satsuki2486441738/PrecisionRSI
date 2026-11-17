# Native CUDA evaluation supplement — 22 October 2026

This supplement adds 22 measured, single-GPU native cases as 14 history entries. The source evidence was published first at commit [`a9fe8dff25de1c3bf806955b06e85d56a9c0a26b`](https://github.com/satsuki2486441738/precision-skills/tree/a9fe8dff25de1c3bf806955b06e85d56a9c0a26b/tests/results/gpt-6/compare-gpurecovery-20261021T180045Z) on `validation/gpt-6-cold-skill-20261019`. Existing evaluations and repair records are preserved.

## What ran

- **SFT:** TRL SFTTrainer, LLaMAFactory, ms-swift and Axolotl each completed native data preprocessing, one training step and a framework-owned SGD update. Each used a seeded tiny Qwen2 and the pinned Qwen3-0.6B checkpoint in FP32 and BF16 execution modes: 16 cases, eight history entries and 80 comparison groups.
- **Inference:** LMDeploy PyTorch CUDA and llama.cpp GGML_CUDA each ran SmolLM2-135M-Instruct and Qwen2.5-0.5B-Instruct. Every case includes a prefill and two actual cached single-token decode calls: four cases, four history entries and 32 comparison groups, including stage-specific views.
- **RL:** ms-swift GRPO completed generation, synthetic rewards, group advantages, loss, full-model backward, clipping and native AdamW updates for tiny FP32 and pretrained BF16 cases: two cases, two history entries and eight comparison groups.

The hardware was one NVIDIA RTX 5060 with 8 GB memory. The original full-vocabulary logits, target scores, gradients, updates and execution observations are retained for each applicable comparison. These cases do not establish four-GPU DDP, convergence, throughput or model quality.

## Independent references and coverage

CPU references run independently from the captured initial model values and effective inputs. They do not use native current-policy scores, gradients or updates as the numerical reference. The RL objective conditions on the actual generated histories, masks, rewards and frozen native old-policy score anchors. Both retained Swift cases have FP32 `[2,4]` old-score tensors; CPU computation independently recomputes the current policy, rather than recomputing both old and current policies. References reproduce the native objective's effective denominator; the objective is distinct from selected-token NLL. Requested FP32/BF16 modes and actual storage/computation dtypes are recorded separately.

SFT comparisons retain the complete trainable parameter inventory, including zero gradients and unchanged coordinates. RL native optimization covers the complete model, while the numerical gradient/update comparison samples **98,304 coordinates** from the first 12 optimizer parameters. The tiny and pretrained inventories contain 19,709,312 and 751,632,384 parameter elements respectively. The CPU reference computes full-model backward and clipping, then applies elementwise AdamW at the selected coordinates. This is explicitly sampled numerical coverage.

There are **120 original comparison groups** and **96 portable replay groups**: 80 SFT, eight overall inference and eight RL. The 24 additional inference stage views are measured slices of the retained logits and target arrays. They do not add portable replay commands to the archive index.

`status: completed` means that the stated native workflow and comparison scope finished. Every numerical `acceptance` remains `null`. The pretrained BF16 RL native objective is zero, whereas the CPU reference is approximately 0.03235; that difference is retained without an accuracy-pass claim. Synthetic rewards validate execution only.

## Provenance and replay

The source run's `GUIDE.md`, `GPU_PROGRESS.json`, `RUN_VALIDATION.json`, `HARDWARE_AUDIT.json` and `SOURCE_IDENTITY_AUDIT.json` describe execution, remaining work, archive validation and post-execution source checks. The SFT environment/log ZIP contains native commands, stdout, resolver/check/freeze receipts and failed preparation attempts. Three new SFT environments pass their own dependency checks; the older read-only TRL environment retains unrelated installed-package conflicts.

Public ZIPs contain indexed portable arrays and replay instructions. Private text paths and machine identifiers were removed; captured array/tensor bytes were preserved and their hashes checked. Public RL ZIP hashes are recorded independently from the pre-export archive hashes. Installed ms-swift 4.6.0 has its original module hashes and null checkout identity; a separate audit matched its 538 Python source files to the official 4.6 release commit. Post-execution audits do not rewrite the collectors' original provenance.

The individual history entries link directly to the immutable evidence, summaries and appropriate replay packages. No generated chart substitutes for a raw measurement. Only recorded, unequal values on two comparable sides receive bars; scalar errors, similarity statistics and coverage remain numerical.

An append-only [fixed-input supplement](https://github.com/satsuki2486441738/precision-skills/blob/5e439f054c192f8beb2f89bf4462a801da993b97/tests/results/gpt-6/compare-gpurecovery-20261021T180045Z/swift-rl/fixed-inputs-index.json) provides 11 original-bit input tensors per Swift RL case, a typed JSON tree and a loader for independent CPU forward/backward. This replay reproduces the retained reference loss and full clipping norm. The original comparison ZIPs and numerical values remain unchanged; the supplement adds no native cases or acceptance decisions.

## Work still in progress

An append-only [H100 continuation](h100-native-20261022-guide.md) now closes Slime, the selected NeMo untied-config route and the four-rank TRL fixture. The paragraph below records this earlier supplement's publication state; the continuation retains its own counts, conditions and immutable evidence.

OpenRLHF, slime, NeMo RL, AReaL, ROLL, official NVIDIA Megatron Bridge and four-rank TRL DDP still need their selected full CUDA workflows and independently compared evidence in this supplement. A new four-H100 training allocation is preparing those workflows; dependency resolution or an import alone does not complete a case.

The selected MindIE-LLM, MindSpeed-RL and historical MindSpeed-LLM routes require Ascend hardware. Their earlier NPU results remain separate from CUDA results. A newer MindSpeed-LLM FSDP2 CUDA initialization branch has been inspected but has not been measured as an alternative route. Earlier complete TensorRT-LLM CUDA evidence is preserved in its existing history; these 22 cases do not claim a new serialized TensorRT engine test.
