# H100 native CUDA continuation — 22 October 2026

This supplement adds **eight native experiments as six history entries**, retaining **102 comparison groups**. Evidence was committed to the original validation branch before history publication: [`fa8e0631759b4f1ead41b6fb39f49daf67546379`](https://github.com/satsuki2486441738/precision-skills/tree/fa8e0631759b4f1ead41b6fb39f49daf67546379/tests/results/gpt-6/compare-h100native-20261021T200953Z). The [source Guide](https://github.com/satsuki2486441738/precision-skills/blob/fa8e0631759b4f1ead41b6fb39f49daf67546379/tests/results/gpt-6/compare-h100native-20261021T200953Z/GUIDE.md), progress snapshot, public hashes and independent validation receipts remain available there. Earlier history and the Repair interface are preserved.

## What ran

- **Slime:** two models, two history entries and fourteen comparison groups. The official controller runs SGLang rollout, grouped rewards, FSDP2 microbatch training, clipping and AdamW, followed by a second NCCL synchronization. One H100 trains and one H100 generates; this does not establish two-rank FSDP sharding. Recorded weight versions are `[1, 2]`.
- **NeMo RL:** two untied-config model variants, two history entries and eight comparison groups. The official `run_grpo_math.py` path runs HF generation, reward/group construction and native clipping/AdamW. Its selected `FSDP1PolicyWorker` DP=1 branch skips FSDP wrapping; the evidence is single-GPU and does not claim distributed sharding.
- **TRL SFTTrainer:** tiny/pretrained × FP32/BF16, four native experiments and two model history entries. Each native experiment runs four H100 ranks. Sixteen rank captures each have five comparisons and an independent single-process CPU reference: eighty comparison groups. Rank captures are not separate experiments; CPU references do not execute DDP.

The model checkpoint is pinned Qwen3-0.6B where applicable; each seeded tiny checkpoint has its own recorded hash. TRL trains the tiny model fully and uses q_proj/v_proj LoRA rank 4 on the pretrained model. Actual captured dtypes are retained independently of requested trainer precision modes.

## Reference conditions and coverage

Slime independently recomputes **both old and current** CPU policies on the captured generated histories. Its native forward uses BF16 with FP32 parameter storage; CPU uses FP32 eager. Every experiment retains four trajectories, two groups and 24 valid actions. Tiny arrays cover all 361,344 parameter elements. Pretrained arrays cover six entire tensors, **1,051,904 / 596,049,920 elements**, while both execution paths run full backward, clipping and AdamW. FSDP current/old scores and SGLang rollout scores are separate boundaries.

NeMo uses fixed recorded native old-policy anchors for both objectives; only the current CPU policy is independently recomputed. Both paths use a derived config with exactly one change, `tie_word_embeddings=False`. Original config/weight bytes and the derivation diff are retained. Each case has four trajectories, one group and 32 actions. Numerical gradient/update coverage is the first twelve optimizer parameters with up to 8,192 elements each: **65,856 tiny / 67,840 pretrained coordinates**. Full-model training is not full-model elementwise comparison. The original CPU clipping return scalar was not captured and is not inferred from a diagnostic norm.

For TRL, all four native ranks receive identical initial values and ordered tokens, labels and masks. The audit verifies synchronized native gradient replicas are exactly equal and retains each rank's native/CPU scalar and norm ratios. This fixture does not generalize to different rank-local batches. Every trainable coordinate, zero gradient and unchanged coordinate is retained; frozen tensors are checked separately. Each CPU process recomputes the original objective denominator. Trainer item count, shifted valid target count and selected-token NLL are distinct quantities.

The actual `clip_grad_norm_` returned norm determines clipping. FP64 full-gradient norms are separate diagnostics. In Slime's pretrained CPU case these differ (`60.67780303955` returned versus `60.6939123357` diagnostic); replacing one with the other changes the update. Near-zero group-averaged loss at initial ratio=1 can still have nonzero gradients and an actual optimizer update.

## Evidence and interpretation

All **102 portable comparison replays matched** in an independent NumPy-only run. Public archives contain original-bit arrays, manifests, conditions, reports and replay specifications. Fixed-input archives preserve dtype, shape, scalar/list structure and configuration derivation for independent CPU model recomputation. Comparison replay alone does not rerun a model. Private PT and compiled Python files are excluded; numerical bytes are unchanged.

Recorded failures remain separate: shell variable scope corrupted dispatch before training; metadata-only `TorchVersion` loading failed after successful DDP training; controller-installed passive hooks failed to capture serialized Ray workers. Recovery records identify what changed and preserve the earlier attempts. No failed attempt contributes a completed experiment.

`completed` means the stated workflow and retained comparison scope finished. Every numerical `acceptance` remains `null`; no threshold, convergence, throughput or answer-quality claim is added. Synthetic rewards validate execution. The interface draws bars only for meaningful unequal recorded values on two comparable sides; scalar errors, similarity statistics and coverage stay numerical.

OpenRLHF, AReaL, ROLL and NVIDIA Megatron Bridge continue separately and are not counted as complete in this snapshot. AReaL's earlier effective attention boundary differed from the CPU reference, so that attempt is explicitly incomparable. Historical MindSpeed-RL, MindIE-LLM and MindSpeed-LLM Ascend routes remain separate; a newer MindSpeed-LLM CUDA candidate is being checked through its own dependencies and native entrypoint.
