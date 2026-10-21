/* Present archived settings as readable fields without changing their values. */
(function (global) {
  'use strict';

  const labels = {
    model: '模型', inputs: '输入', numerics: '数值精度', execution: '执行',
    scope: '范围', state: '状态', name: '名称', id: '标识', family: '模型系列',
    vocab_mapping: '词表映射', logits: 'Logits', logsoftmax: 'Log-softmax',
    framework: '框架', native_cache_replay: '原生缓存回放',
    native_entrypoint: '原生入口', score_processors: '分数处理器',
    revision: '版本 revision', tokenizer_revision: '分词器 revision', source_commit: '源码提交',
    source_pins: '源码固定版本', source_files: '源文件', checkpoint_file_hashes: 'Checkpoint 文件校验',
    checkpoint_safetensors: 'Checkpoint 文件校验', checkpoint_safetensors_sha256: 'Checkpoint SHA-256',
    tokenizer_sha256: '分词器 SHA-256', config_sha256: '配置 SHA-256', gguf_sha256: 'GGUF SHA-256',
    path: '路径', bytes: '字节数', sha256: 'SHA-256', version: '版本',
    dtype: '数据类型', base_dtype: '基座数据类型', adapter_dtype: 'Adapter 数据类型',
    model_dtype: '模型数据类型', parameter_dtype: '参数数据类型', parameter_storage_dtype: '参数存储类型',
    parameters: '参数', gradients: '梯度', weights: '权重', initial_weights: '初始权重',
    requested_precision: '请求精度', logits_dtype: 'Logits 数据类型', raw_logits_dtype: '原始 Logits 数据类型',
    raw_score_dtype: '原始分数数据类型', gradient_dtype: '梯度数据类型', loss_dtype: 'Loss 数据类型',
    objective_dtype: '目标函数数据类型', logprob_dtype: 'Logprob 数据类型',
    stored_parameter_dtypes: '存储参数类型', all_model_parameter_dtypes: '全部模型参数类型',
    captured_gradient_dtypes: '已采集梯度类型', actual_gradient_dtype: '实际梯度类型',
    device: '设备', hardware: '硬件', backend: '后端', engine: '引擎',
    attention: '注意力实现', attention_backend: '注意力后端', attn_backend: '注意力后端',
    flash_attention: 'Flash Attention', tf32: 'TF32', dropout: 'Dropout',
    deterministic_algorithms: '确定性算法', seed: '随机种子', temperature: '温度',
    prompt: '提示 token', prompt_ids: '提示 token ID', continuation: '续写 token',
    continuation_ids: '续写 token ID', generated_continuation_ids: '生成续写 token ID',
    request_input_ids: '请求输入 token ID', forwarded_input_ids: '前向输入 token ID',
    actual_native_generated_token_ids: '实际原生生成 token ID',
    attention_mask: '注意力掩码', position_ids: '位置 ID', positions: '位置',
    prediction_positions: '预测位置', logical_prediction_positions: '逻辑预测位置',
    sample: '样本', samples: '样本数', token: 'Token ID', position: '位置',
    parameter: '参数', index: '索引', history: '历史 token', histories: '历史序列', targets: '目标坐标',
    add_special_tokens: '添加特殊 token', forward_shape: '前向执行形态',
    forward: '前向计算', forward_calls: '前向调用次数', mode: '模式', task: '任务',
    batch_size: '批大小', micro_batch_size: 'Microbatch 大小', global_batch_size: '全局批大小',
    global_microbatch_size: '全局 Microbatch 大小', vocab_size: '词表大小',
    rank: '进程 rank', world_size: '进程数', tp: '张量并行度', requests: '请求数',
    selected_rows: '选取行数', paired_prediction_rows: '配对预测行数', target_count: '目标数',
    requested_targets: '请求目标数', requested_distributions: '请求分布数',
    valid_targets: '有效目标数', valid_actions: '有效动作数', actions: '动作数',
    trajectories: '轨迹数', groups: '组数', group_count: '组数', group_size: '每组大小',
    complete_groups: '完整组数', training: '训练', steps: '步数', optimizer_steps: '优化器步数',
    optimizer_step: '优化器步数', optimizer: '优化器', learning_rate: '学习率', lr: '学习率',
    betas: 'Adam betas', eps: 'Epsilon', momentum: '动量', weight_decay: '权重衰减',
    clip_grad: '梯度裁剪', max_grad_norm: '梯度范数上限', clipping: '裁剪',
    gradient_clipping: '梯度裁剪', gradient_accumulation: '梯度累积',
    gradient_accumulation_steps: '梯度累积步数', accumulation: '梯度累积',
    cache: '缓存', use_cache: '使用缓存', cache_dtype: '缓存数据类型', KV_dtype: 'KV 数据类型',
    kv_storage: 'KV 存储类型', logits_storage: 'Logits 存储类型', logit_storage: 'Logits 存储类型',
    weights_format: '权重格式', quantization: '量化', unquantized: '未量化',
    arithmetic: '计算精度', implementation: '实现', capture: '采集', coordinates: '坐标',
    requested_settings: '请求配置', requested_cache: '请求缓存配置', effective_settings: '生效配置',
    backend_settings: '后端配置', kwargs: '参数', role: '角色', threads: '线程数', torch_threads: 'Torch 线程数',
    loss: 'Loss', objective: '目标函数', loss_config: 'Loss 配置', reduction: '聚合方式',
    normalization: '归一化', mask: '掩码', stages: '阶段', sampled_elements: '采样元素数',
    interpreter: '解释器', execution_stack: '执行栈', CANN_version: 'CANN 版本',
    CANN_ATB_versions: 'CANN / ATB 版本', source_version: '源码版本', source_case: '来源 Case'
  };
  const openGroups = new Set(['model', 'inputs', 'numerics', 'execution', 'scope', 'state']);
  const isObject = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
  const isProvenance = (key, value) => /(?:revision|commit|sha256|hash|source_pins|source_files|checkpoint_safetensors)/i.test(key) ||
    typeof value === 'string' && /^[a-f0-9]{32,}$/i.test(value);

  function settingLabel(key) {
    const name = String(key);
    if (Object.prototype.hasOwnProperty.call(labels, name)) return labels[name];
    const suffix = name.match(/^(.+)_sha256$/i);
    if (suffix) return `${settingLabel(suffix[1])} SHA-256`;
    return name.replace(/([a-z0-9])([A-Z])/g, '$1 $2').replace(/[_-]+/g, ' ').trim() || '字段';
  }

  function formatSettingValue(value, key = '') {
    if (value === null) return '未记录（null）';
    if (value === undefined) return '未提供（undefined）';
    if (value === true) return '是（true）';
    if (value === false) return '否（false）';
    if (typeof value === 'number' || typeof value === 'bigint') return String(value);
    if (Array.isArray(value)) {
      return value.length ? `[${value.map((item) => formatSettingValue(item, key)).join(', ')}]` : '空数组（0 项）';
    }
    if (isObject(value)) return Object.keys(value).length ? `对象（${Object.keys(value).length} 个字段）` : '空对象（0 项）';
    const content = String(value);
    if (!content.length) return '空字符串';
    if (key === 'forward_shape') {
      if (content === 'one6-token prefill plus two single-token cached decode calls') {
        return '1 次 6 token 预填充，随后 2 次单 token 缓存解码';
      }
      return content.replace(/\b(one|lengths|rows)(?=\d)/gi, '$1 ').replace(/(\d),(?=\d)/g, '$1, ');
    }
    const knownValues = {
      continuous: '连续执行', restored: 'Checkpoint 恢复', 'not recorded': '未记录',
      'not measured': '未测量', 'not executed': '未执行', disabled: '禁用', enabled: '启用'
    };
    return Object.prototype.hasOwnProperty.call(knownValues, content) ? knownValues[content] : content;
  }

  function renderSettings(settings, helpers, options = {}) {
    const { node } = helpers;
    const root = node('div', `evaluation-settings${options.compact ? ' evaluation-settings-compact' : ''}`);

    function scalar(target, key, value) {
      const formatted = formatSettingValue(value, key);
      if (isProvenance(key, value) && typeof value === 'string' && value.length > 24) {
        const details = node('details', 'evaluation-setting-value-details');
        details.append(node('summary', '', `${value.slice(0, 12)}… · 查看完整值`),
          node('code', 'evaluation-setting-code', formatted));
        target.append(details);
      } else {
        target.append(node('span', 'evaluation-setting-text', formatted));
      }
    }

    function array(target, key, values, depth) {
      const tokenList = /(?:token|prompt|continuation|history|input_ids)/i.test(key) && values.every((item) => typeof item === 'number');
      const count = `${values.length} ${tokenList ? '个 token' : '项'}`;
      if (!values.length) {
        scalar(target, key, values);
        return;
      }
      const simple = values.every((value) => value === null || typeof value !== 'object');
      const content = simple ? formatSettingValue(values, key) : '';
      if (simple && values.length <= 12 && content.length <= 160 && !isProvenance(key, '')) {
        target.append(node('span', 'evaluation-setting-text evaluation-setting-array', content),
          node('span', 'evaluation-setting-count', count));
        return;
      }
      const details = node('details', 'evaluation-setting-value-details');
      details.append(node('summary', '', `${count} · 展开明细`));
      if (simple && values.every((value) => typeof value === 'number')) {
        details.append(node('div', 'evaluation-setting-text evaluation-setting-array', content));
      } else {
        const list = node('ol', 'evaluation-setting-items');
        values.forEach((value) => {
          const item = node('li');
          if (isObject(value)) object(item, value, depth + 1);
          else if (Array.isArray(value)) array(item, key, value, depth + 1);
          else scalar(item, key, value);
          list.append(item);
        });
        details.append(list);
      }
      target.append(details);
    }

    function object(target, value, depth) {
      const entries = Object.entries(value);
      if (!entries.length) {
        scalar(target, '', value);
        return;
      }
      let fields;
      entries.forEach(([key, child]) => {
        // This is an archive wrapper, not a user-facing configuration group.
        if (key === 'recorded_conditions' && isObject(child)) {
          fields = null;
          object(target, child, depth);
          return;
        }
        if (isObject(child) && Object.keys(child).length) {
          fields = null;
          const group = node('details', 'evaluation-settings-group');
          group.dataset.settingKey = key;
          group.open = openGroups.has(key) && depth < 2 && !isProvenance(key, '');
          group.append(node('summary', '', settingLabel(key)));
          const body = node('div', 'evaluation-settings-group-body');
          object(body, child, depth + 1);
          group.append(body);
          target.append(group);
          return;
        }
        if (!fields) {
          fields = node('dl', 'evaluation-setting-fields');
          target.append(fields);
        }
        const row = node('div', 'evaluation-setting-row');
        row.dataset.settingKey = key;
        const term = node('dt', '', settingLabel(key));
        const definition = node('dd', 'evaluation-setting-value');
        if (Array.isArray(child)) array(definition, key, child, depth);
        else scalar(definition, key, child);
        row.append(term, definition);
        fields.append(row);
      });
    }

    if (isObject(settings)) object(root, settings, 0);
    else if (Array.isArray(settings)) array(root, '', settings, 0);
    else scalar(root, '', settings);
    return root;
  }

  const api = { renderSettings, settingLabel, formatSettingValue };
  if (typeof module === 'object' && module.exports) module.exports = api;
  global.PrecisionEvaluationSettings = api;
})(globalThis);
