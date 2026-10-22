/* Display-only evaluation names. Recorded titles and evidence remain authoritative. */
(function (global) {
  'use strict';
  const list = (value) => Array.isArray(value) ? value : [];
  const text = (value) => typeof value === 'string' ? value.trim() : '';
  const frameworkNames = [
    [/Megatron Bridge|^megatron-bridge-/i, 'Megatron Bridge'],
    [/MindSpeed-RL|^mindspeed-rl-/i, 'MindSpeed-RL'],
    [/MindSpeed-LLM|^mindspeed-llm-/i, 'MindSpeed-LLM'],
    [/TensorRT-LLM|^tensorrt-llm-/i, 'TensorRT-LLM'],
    [/NeMo RL|^nemo(?:-rl)?-/i, 'NeMo RL'],
    [/OpenRLHF|^openrlhf-/i, 'OpenRLHF'],
    [/LLaMAFactory|^(?:factory|llamafactory)-/i, 'LLaMAFactory'],
    [/ms-swift|^(?:ms-swift|swift)-/i, 'ms-swift'],
    [/llama\.cpp|^llamacpp-|^llama-cuda-/i, 'llama.cpp'],
    [/LMDeploy|^lmdeploy-/i, 'LMDeploy'],
    [/MindIE-LLM|^mindie-/i, 'MindIE-LLM'],
    [/Axolotl|^axolotl-/i, 'Axolotl'],
    [/\bTRL\b|^trl-/i, 'TRL'],
    [/AReaL|^areal-/i, 'AReaL'],
    [/\bROLL\b|^roll-/i, 'ROLL'],
    [/\bslime\b|^slime-/i, 'slime'],
    [/PyTorch|^torch-/i, 'PyTorch'],
  ];

  function settingRows(value, path = '') {
    if (value && typeof value === 'object') {
      return Object.entries(value).flatMap(([key, child]) => settingRows(child, path ? `${path}.${key}` : key));
    }
    return [{ path, value }];
  }
  function frameworkFor(record) {
    const title = text(record.title);
    const id = text(record.id);
    const known = frameworkNames.find(([pattern]) => pattern.test(title) || pattern.test(id));
    if (known) return known[1];
    return text(record.framework?.name) || text(record.framework) ||
      title.split(/[：:]|\s+·\s+/)[0].match(/^[A-Za-z][A-Za-z0-9_.-]*/)?.[0] || '评测';
  }
  function platformFor(record) {
    const hardware = list(record.sides).map((side) => text(side.hardware)).join(' ');
    const ascend = /Ascend|\bNPU\b|昇腾/i.test(hardware);
    const cuda = /NVIDIA|\bCUDA\b|\bRTX\b|\bH[12]00\b/i.test(hardware);
    if (ascend && cuda) return 'CUDA ↔ Ascend';
    if (ascend) return 'Ascend';
    if (cuda) return 'CUDA';
    return /\bCPU\b/i.test(hardware) ? 'CPU' : '';
  }
  function factsFor(record) {
    const rows = list(record.sides).flatMap((side) => settingRows(side.settings || {}));
    // Scope/mode text may describe a real route. Ignore hashes, source paths and
    // measured values, which must never become display-name heuristics.
    const modes = rows.filter((row) => /scope|execution|mode|forward_shape|backend|engine|任务|执行|入口|训练|training|算法|loss_type/i.test(row.path))
      .map((row) => typeof row.value === 'string' ? row.value : '').join(' ');
    const title = text(record.title);
    return { rows, title, modes, context: [title, text(record.summary), ...list(record.common),
      ...list(record.differences), modes].join(' ') };
  }
  function trainingFor(facts) {
    const declared = facts.rows.filter((row) => /training|训练$|task$/.test(row.path)).map((row) => String(row.value));
    if (declared.some((value) => /^(?:full|完整参数训练)|full-parameter/i.test(value))) return '全参';
    if (/LoRA/i.test(facts.title) || declared.some((value) => /LoRA/i.test(value))) return 'LoRA';
    if (facts.rows.some((row) => /adapter_dtype$/.test(row.path)) && /LoRA/i.test(facts.context)) return 'LoRA';
    return '';
  }
  function routeFor(title) {
    const routes = [];
    if (/\bkernel\b|内核路径|算子路径/i.test(title)) routes.push('kernel');
    if (/parameteri[sz]ed|参数化/i.test(title)) routes.push('参数化');
    return routes.join(' / ');
  }
  function inferencePurpose(record, framework, platform, facts) {
    const { title, context, rows } = facts;
    if (/BF16/i.test(title) && /FP16/i.test(title)) return 'BF16 ↔ FP16 完整前向 / 缓存推理';
    const cold = /冷上下文/.test(title) || (rows.some((row) => /cross_forward_cached_decode$/.test(row.path) && row.value === false) && /clear native memory|prefix.*清空/i.test(context));
    if (cold) return [platform, '冷上下文推理'].filter(Boolean).join(' ');
    const repeated = /repeated.prefill|重复.?prefill|重复预填充|fresh full-prefix|fresh full.prefix|fresh full-prefix requests/i.test(context);
    if (repeated) {
      const engines = framework === 'LMDeploy' && /turbomind/i.test(facts.modes) && /pytorch/i.test(facts.modes) ? '双引擎' : '';
      return [platform, `${engines}重复预填充`].filter(Boolean).join(' ');
    }
    const cached = rows.some((row) => /(?:cache_replay|cross_forward_cached_decode|kv_cache_retained)$/.test(row.path) && row.value === true) ||
      /cached decode|缓存解码|cached fixed.history|固定历史缓存|prefill.*cached/i.test(context);
    if (cached) {
      const backend = framework === 'llama.cpp' && platform === 'Ascend' ? 'CPU ↔ CANN' : platform;
      const route = framework === 'TensorRT-LLM' && /TRTLLM/i.test(facts.modes) ? 'TRTLLM ' : '';
      return [backend, `${route}缓存推理`].filter(Boolean).join(' ');
    }
    if (/full.forward|full.sequence|完整前向|完整序列/i.test(context)) return [platform, '完整前向推理'].filter(Boolean).join(' ');
    return '';
  }
  function sftPurpose(record, framework, platform, facts) {
    const training = trainingFor(facts);
    const worldSize = Math.max(1, ...facts.rows.filter((row) => /world_size$/.test(row.path) && Number.isInteger(row.value)).map((row) => row.value));
    if (/\bDDP\b|四卡/.test(facts.title) || (worldSize > 1 && /\bDDP\b/.test(facts.context))) {
      return [`${worldSize > 1 ? worldSize : 4}卡 DDP`, training, 'SFT'].filter(Boolean).join(' ');
    }
    if (/checkpoint|恢复/i.test(facts.title)) return [training, 'checkpoint 恢复'].filter(Boolean).join(' ');
    if (framework === 'Megatron Bridge' && /checkpoint|冷恢复|新进程恢复/i.test(facts.context)) {
      return [platform, /\bfork\b/i.test(facts.context) ? 'fork' : '', 'checkpoint 冷恢复 SFT'].filter(Boolean).join(' ');
    }
    if (/pretrain|预训练/i.test(facts.title)) return [platform, '预训练'].filter(Boolean).join(' ');
    if (/trainer component|训练组件/i.test(facts.title)) return [platform, 'SFT 组件'].filter(Boolean).join(' ');
    if (/\bSFT\b/i.test(facts.title)) return [platform, training, 'SFT'].filter(Boolean).join(' ');
    return '';
  }
  function rlPurpose(record, framework, platform, facts) {
    const { title, context, rows } = facts;
    const primaryAlgorithms = rows.filter((row) => /(?:algorithm|算法)$/.test(row.path));
    const declaredAlgorithms = (primaryAlgorithms.length ? primaryAlgorithms : rows.filter((row) => /loss_type$/.test(row.path))).map((row) => row.value).join(' ');
    const algorithm = /GRPO/i.test(title + ' ' + declaredAlgorithms) ? 'GRPO' : /PPO/i.test(context) ? 'PPO' : '';
    if (/worker API/i.test(title)) return [platform, 'PPO worker'].filter(Boolean).join(' ');
    if (/RL 组件/i.test(title)) return [platform, algorithm || (/GRPOTrainer/.test(context) ? 'GRPO' : '策略优化'), '组件'].filter(Boolean).join(' ');
    if (/reward.driven CLI/i.test(title)) return [platform, '奖励驱动 GRPO'].filter(Boolean).join(' ');
    const untied = /untied/i.test(`${title} ${record.model?.name || ''}`) ? 'untied' : '';
    const dtype = title.match(/\b(?:BF16|FP16|FP32)\b/i)?.[0].toUpperCase() || '';
    const twoDevices = /两\s*(?:GPU|卡)|2\s*×\s*NVIDIA/i.test(context + ' ' + list(record.sides).map((side) => side.hardware).join(' ')) ||
      rows.some((row) => /执行\.GPU$/.test(row.path) && row.value === 2);
    const objective = algorithm || (rows.some((row) => /算法$/.test(row.path) && /GRPO/i.test(String(row.value))) ? 'GRPO' : /策略优化/.test(title) ? 'PPO' : '');
    if (objective || untied) return [platform, twoDevices ? '两卡' : '', dtype, untied, objective || 'GRPO'].filter(Boolean).join(' ');
    return '';
  }
  function fallbackPurpose(record, framework, modelLabel) {
    let purpose = text(record.title);
    for (const literal of [framework, text(record.model?.name), modelLabel].filter(Boolean)) purpose = purpose.split(literal).join(' ');
    purpose = purpose.replace(/\b20\d{2}[-/]?\d{2}[-/]?\d{2}(?:T\d{6}Z)?\b/gi, ' ')
      .replace(/(?:seed\s*[:=]?\s*\d+|种子\s*\d+)/gi, ' ')
      .replace(/原生|已完成|精度评测/gi, ' ')
      .replace(/full[- ](?:forward|sequence)/gi, '完整前向')
      .replace(/parameteri[sz]ed/gi, '参数化')
      .replace(/\binference\b/gi, '推理')
      .replace(/[·：:()（）]/g, ' ').replace(/\s+/g, ' ').trim();
    return purpose || ({ inference: '推理', sft: 'SFT', rl: '策略优化' }[record.task] || '实验');
  }

  // Model normalization belongs to PrecisionHistory.evaluationModel. The caller
  // passes that label so grouping, filters and card titles use the same identity.
  function titleFor(record = {}, modelLabel = '') {
    record = record || {};
    modelLabel = text(modelLabel) || text(record.model?.name);
    const framework = frameworkFor(record);
    const platform = platformFor(record);
    const facts = factsFor(record);
    let purpose = record.task === 'inference' ? inferencePurpose(record, framework, platform, facts) :
      record.task === 'sft' ? sftPurpose(record, framework, platform, facts) :
        record.task === 'rl' ? rlPurpose(record, framework, platform, facts) : '';
    if (purpose) purpose = [routeFor(facts.title), purpose].filter(Boolean).join(' ');
    else purpose = fallbackPurpose(record, framework, modelLabel);
    return [framework, purpose, modelLabel].filter(Boolean).join(' · ');
  }

  const api = { titleFor };
  if (typeof module === 'object' && module.exports) module.exports = api;
  global.EvaluationPresentation = api;
})(globalThis);
