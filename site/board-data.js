/* Derive the board dataset from the site's own published records.
 *
 * The page reads `data/evaluations.json` and `data/repairs.json` directly, so
 * the interface can never show a frozen copy that has drifted from the data
 * the rest of the site serves. This is a port of tools/build_board.py; the two
 * are kept equivalent by tests/test_board_data.js.
 *
 * Note on word boundaries: Python's `\b` is Unicode-aware, so `\bloss\b` does
 * not match inside "loss值". JavaScript's `\w` is ASCII-only and would match
 * there, silently reclassifying rows — hence the explicit lookarounds below.
 */
(function (global) {
  "use strict";

  var NB = "[^\\p{L}\\p{N}_]"; // Unicode-aware non-word, standing in for \b
  function rx(source) { return new RegExp(source, "u"); }

  var PLATFORMS = [
    [rx("ascend\\s*910c|ascend910_9382|ascend910c"), "Ascend 910C", "npu"],
    [rx("ascend\\s*910b"), "Ascend 910B", "npu"],
    [rx("ascend|npu"), "Ascend NPU", "npu"],
    [rx("h200|h20z"), "NVIDIA H200", "gpu"],
    [rx("h100"), "NVIDIA H100", "gpu"],
    [rx("5090"), "RTX 5090", "gpu"],
    [rx("5060"), "RTX 5060", "gpu"],
    [rx("nvidia|cuda|geforce"), "NVIDIA GPU", "gpu"],
    [rx("cpu|aarch64|x86_64|macos|wsl"), "CPU", "cpu"],
  ];
  function platformOf(text) {
    var low = (text || "").toLowerCase();
    for (var i = 0; i < PLATFORMS.length; i++)
      if (PLATFORMS[i][0].test(low)) return [PLATFORMS[i][1], PLATFORMS[i][2]];
    return [((text || "未记录").trim().slice(0, 26) || "未记录"), "other"];
  }

  // What quantity the comparison is about.
  var SUBJECTS = [
    ["distribution", rx("词表|分布|distribution|logits|probability mass|top-?k"), "词表分布"],
    ["score", rx("目标分数|target score|nll|logprob|目标 logprob|score"), "目标分数"],
    ["gradient", rx("梯度|gradient|grad(?![\\p{L}\\p{N}_])"), "梯度"],
    ["update", rx("存储更新|参数更新|optimizer|adamw|sgd|更新|update"), "参数/优化器更新"],
    ["checkpoint", rx("checkpoint|检查点|恢复|resume|restore"), "Checkpoint 往返"],
    ["tensor", rx("观测张量|activation|tensor|张量|hidden"), "张量/激活"],
    ["loss", rx("(?<=^|" + NB + ")loss(?![\\p{L}\\p{N}_])|损失"), "Loss"],
  ];
  // What single difference the comparison isolates.
  var AXES = [
    ["precision", rx("fp32|bf16|fp16|精度|dtype"), "跨精度"],
    ["exec_mode", rx("连续训练|恢复|resume|continuous|modes?(?![\\p{L}\\p{N}_])|hooks|observer|on/?off|开关"), "跨执行模式"],
    ["reference", rx("独立|参考|reference|原生|native"), "原生 ↔ 独立参考"],
    ["skills", rx("skills"), "Skills 开关"],
  ];
  function match(table, text) {
    var low = (text || "").toLowerCase();
    for (var i = 0; i < table.length; i++)
      if (table[i][1].test(low)) return [table[i][0], table[i][2]];
    return [null, null];
  }

  var METRICS = [
    ["max_abs", rx("最大绝对误差|max(imum)? absolute (error|difference)"), "最大绝对误差", "lower"],
    ["rmse", rx("^rmse|均方根"), "RMSE", "lower"],
    ["l2", rx("^l2 误差|^l2 error|l2 范数误差"), "L2 误差", "lower"],
    ["cosine", rx("余弦相似度|cosine similarity"), "余弦相似度", "higher"],
    ["nll_abs", rx("nll (绝对差|absolute difference)|目标 nll"), "NLL 绝对差", "lower"],
    ["logprob_mae", rx("logprob mae|logprob 平均绝对|target logprob"), "logprob MAE", "lower"],
    // "Top-1" is a prefix of "Top-100", so require no following digit. Only the
    // fraction form is kept: the raw "agreements / positions" count is a
    // different quantity and must not share a column with a ratio.
    ["top1", rx("top-?1(?![0-9]).*(agreement fraction|一致率|一致比例)"), "Top-1 一致率", "higher"],
  ];
  function metricKey(label) {
    var low = (label || "").toLowerCase();
    for (var i = 0; i < METRICS.length; i++)
      if (METRICS[i][1].test(low)) return [METRICS[i][0], METRICS[i][2], METRICS[i][3]];
    return [null, null, null];
  }

  var FRAMEWORKS = ["vLLM", "SGLang", "TRL", "PEFT", "verl", "Megatron", "MindSpeed",
                    "Transformers", "Accelerate", "PyTorch"];
  function frameworksOf(sides) {
    var found = [];
    sides.forEach(function (side) {
      (side.libraries || []).forEach(function (lib) {
        FRAMEWORKS.forEach(function (fw) {
          if (fw.toLowerCase() === String(lib.name == null ? "" : lib.name).toLowerCase()
              && found.indexOf(fw) < 0) found.push(fw);
        });
      });
    });
    // PyTorch/Transformers are near-universal; name them only when nothing sharper exists.
    var sharp = found.filter(function (f) {
      return f !== "PyTorch" && f !== "Transformers" && f !== "Accelerate";
    });
    return sharp.length ? sharp : found.slice(0, 1);
  }

  function isPlainObject(v) {
    return v !== null && typeof v === "object" && !Array.isArray(v);
  }
  /* Nested settings as ordered groups, so the UI never prints an object dump. */
  function flattenSettings(settings, limit) {
    limit = limit || 30;
    // A settings value can be an object (a file manifest entry, say). Naming it
    // by its identifying field keeps the panel readable; dumping every key —
    // or letting String() produce "[object Object]" — does not.
    function scalar(value) {
      if (value === null || value === undefined) return "null";
      if (Array.isArray(value)) return value.map(scalar).join(", ");
      if (isPlainObject(value)) {
        var keys = Object.keys(value);
        var id = ["name", "path", "file", "id", "label"].filter(function (k) {
          return typeof value[k] === "string";
        })[0];
        if (id) return String(value[id]);
        return keys.slice(0, 3).map(function (k) {
          var v = value[k];
          return k + "=" + (isPlainObject(v) || Array.isArray(v) ? "…" : String(v));
        }).join(" ");
      }
      return String(value);
    }
    function render(value) {
      if (Array.isArray(value)) {
        if (value.length > 8)
          return "[" + value.length + " 项] " + value.slice(0, 6).map(scalar).join(", ") + " …";
        return value.map(scalar).join(", ");
      }
      return scalar(value);
    }
    var groups = [];
    function walk(mapping, path) {
      var fields = [];
      Object.keys(mapping).forEach(function (key) {
        var value = mapping[key];
        if (isPlainObject(value) && Object.keys(value).length) walk(value, path.concat([key]));
        else fields.push([key, render(value)]);
      });
      if (fields.length) groups.push({ group: path.join(" · "), fields: fields.slice(0, limit) });
    }
    walk(settings, []);
    // Python sorts on (group != "", group): the unnamed root group comes first.
    groups.sort(function (a, b) {
      var ka = a.group !== "" ? 1 : 0, kb = b.group !== "" ? 1 : 0;
      if (ka !== kb) return ka - kb;
      return a.group < b.group ? -1 : a.group > b.group ? 1 : 0;
    });
    return groups.slice(0, 10);
  }

  /* Counter.most_common(): count desc, first-seen order preserved on ties. */
  function mostCommon(items) {
    var counts = new Map();
    items.forEach(function (x) { counts.set(x, (counts.get(x) || 0) + 1); });
    return Array.from(counts.entries()).sort(function (a, b) { return b[1] - a[1]; });
  }
  function median(values) {
    var s = values.slice().sort(function (a, b) { return a - b; }), n = s.length;
    if (!n) return null;
    return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
  }
  var isNum = function (v) { return typeof v === "number" && isFinite(v); };

  function deriveBoard(data, repairs) {
    var evaluations = data.evaluations || [];
    var rows = [], unclassified = {};
    var bump = function (k) { unclassified[k] = (unclassified[k] || 0) + 1; };

    evaluations.forEach(function (ev) {
      var sides = {};
      (ev.sides || []).forEach(function (s) { sides[s.id] = s; });
      var sideList = Object.keys(sides).map(function (k) { return sides[k]; });
      sideList.forEach(function (side) {
        var p = platformOf(side.hardware || "");
        side._platform = p[0]; side._kind = p[1];
      });
      var fws = frameworksOf(sideList);

      (ev.comparisons || []).forEach(function (cmp) {
        var picked = (cmp.side_ids || []).map(function (id) { return sides[id]; })
                                         .filter(Boolean);
        var text = [cmp.title || "", cmp.scope || "", ev.title || ""]
          .concat(picked.map(function (s) { return s.label || ""; })).join(" ");

        var sub = match(SUBJECTS, text), subjectKey = sub[0], subject = sub[1];
        if (!subjectKey) bump("subject");

        var kinds = new Set(picked.map(function (s) { return s._kind; }));
        var platforms = [];
        picked.forEach(function (s) {
          if (platforms.indexOf(s._platform) < 0) platforms.push(s._platform);
        });
        var axisKey, axis;
        if (kinds.size > 1) { axisKey = "platform"; axis = "跨平台"; }
        else {
          var ax = match(AXES, text); axisKey = ax[0]; axis = ax[1];
          if (!axisKey) bump("axis");
        }

        var metrics = {};
        (cmp.metrics || []).forEach(function (m) {
          var mk = metricKey(m.label || ""), key = mk[0], better = mk[2];
          if (key === null || !isNum(m.value)) return;
          var value = Number(m.value), old = metrics[key];
          // Several source metrics can share one canonical id; keep the worst
          // reading so a row never looks better than it is.
          if (old === undefined
              || (better === "lower" && value > old.value)
              || (better === "higher" && value < old.value))
            metrics[key] = { value: value, unit: m.unit || "", label: m.label || "", better: better };
        });

        var observed = 0, expected = 0;
        (cmp.coverage || []).forEach(function (c) {
          observed += c.observed || 0; expected += c.expected || 0;
        });

        rows.push({
          eval_id: ev.id, date: ev.date || "", task: ev.task || "",
          status: ev.status || "", model: (ev.model || {}).name || "",
          eval_title: ev.title || "", title: cmp.title || ev.title || "",
          scope: cmp.scope || "",
          subject: subject || "未分类", subject_key: subjectKey || "other",
          axis: axis || "未分类", axis_key: axisKey || "other",
          platforms: platforms, pair: platforms.join(" ↔ ") || "—",
          frameworks: fws, metrics: metrics,
          coverage: { observed: observed, expected: expected,
                      ratio: expected ? observed / expected : null },
          sides: picked.map(function (s) {
            return {
              label: s.label || "", platform: s._platform, hardware: s.hardware || "",
              libraries: (s.libraries || []).slice(0, 8).map(function (l) {
                return (l.name + " " + (l.version || "")).trim();
              }),
              settings: flattenSettings(s.settings || {}),
            };
          }),
          notes: (ev.notes || []).slice(0, 4).map(function (n) {
            return typeof n === "string" ? n : (n.text || n.label || "");
          }),
          evidence: (ev.evidence || []).slice(0, 3).map(function (x) {
            return { label: x.label || "", url: x.url || "" };
          }),
        });
      });
    });

    // Hero: a worked example that overturns the naive assumption. Pulled from
    // the records rather than written by hand, so it cannot drift from them.
    var hero = null;
    evaluations.forEach(function (ev) {
      if (ev.id !== "gemma4-h100-ascend-partitions-20261107") return;
      var pick = {};
      (ev.comparisons || []).forEach(function (cmp) {
        (cmp.metrics || []).forEach(function (m) {
          if ((m.label || "").indexOf("Native KV global maximum absolute error") === 0)
            pick[cmp.title] = m.value;
          if ((cmp.title || "").indexOf("Within-process control: gpu-full-r1 vs gpu-full-r2") === 0
              && m.label === "Target maximum absolute difference")
            pick.control = m.value;
        });
      });
      var at = function (k) { return pick[k] === undefined ? null : pick[k]; };
      hero = {
        eval_id: ev.id, model: (ev.model || {}).name || "",
        bars: [
          { label: "同一张 H100 · 重复跑", value: at("control"), note: "位级一致", kind: "same" },
          { label: "跨平台 · 同为 token 路径", value: at("Native final KV: gpu-token vs npu-token"),
            note: "H100 ↔ Ascend", kind: "cross" },
          { label: "同一张 H100 · 换 KV 分块", value: at("Native final KV: gpu-token vs gpu-chunk63-65"),
            note: "token vs chunk63-65", kind: "worse" },
          { label: "同一张 Ascend · 换 KV 分块", value: at("Native final KV: npu-token vs npu-chunk63-65"),
            note: "token vs chunk63-65", kind: "worse" },
        ],
        unit: "KV 最大绝对误差",
      };
    });

    // Repairs carry one headline metric with its direction, so the figure never
    // draws a higher-is-better reading on the same axis as an error.
    var higher = rx("相似度|similarity|一致率|agreement|covered|通过");
    (repairs.repairs || []).forEach(function (rp) {
      var pairs = ((rp.validation || {}).metrics || []).filter(function (m) {
        return isNum(m.before) && isNum(m.after);
      });
      var pickM = null;
      pairs.forEach(function (m) {                       // Python max(): keeps the first maximum
        if (pickM === null || Math.abs(m.before) > Math.abs(pickM.before)) pickM = m;
      });
      rp.headline = pickM === null ? null : {
        label: pickM.label || "", unit: pickM.unit || "",
        before: pickM.before, after: pickM.after,
        better: higher.test(pickM.label || "") ? "higher" : "lower",
      };
      rp.lib = rp.affected_library || rp.library_group_id || "其它";
    });

    var facet = function (field) {
      return mostCommon(rows.map(function (r) { return r[field]; }));
    };
    // Compare view: per platform × subject × metric, how far that platform sat
    // from its counterpart. The number describes the pair it took part in,
    // never the platform on its own.
    var agg = new Map();
    rows.forEach(function (r) {
      new Set(r.platforms).forEach(function (plat) {
        Object.keys(r.metrics).forEach(function (key) {
          var k = plat + "\u0000" + r.subject + "\u0000" + key;
          if (!agg.has(k)) agg.set(k, []);
          agg.get(k).push(r.metrics[key].value);
        });
      });
    });
    var compare = {};
    agg.forEach(function (values, k) {
      var parts = k.split("\u0000"), plat = parts[0], subject = parts[1], key = parts[2];
      var better = "lower";
      METRICS.forEach(function (m) { if (m[0] === key) better = m[3]; });
      if (!compare[subject]) compare[subject] = {};
      if (!compare[subject][plat]) compare[subject][plat] = {};
      compare[subject][plat][key] = {
        n: values.length, median: median(values),
        worst: better === "lower" ? Math.max.apply(null, values) : Math.min.apply(null, values),
        better: better,
      };
    });

    var metricsTotal = 0;
    evaluations.forEach(function (e) {
      (e.comparisons || []).forEach(function (c) { metricsTotal += (c.metrics || []).length; });
    });

    var summary = {
      updated_at: data.updated_at || "",
      compare: compare, evaluations: evaluations.length, comparisons: rows.length,
      metrics_total: metricsTotal,
      tasks: facet("task"), subjects: facet("subject"), axes: facet("axis"),
      platforms: mostCommon([].concat.apply([], rows.map(function (r) { return r.platforms; }))),
      frameworks: mostCommon([].concat.apply([], rows.map(function (r) { return r.frameworks; }))),
      metric_columns: METRICS.map(function (m) {
        return { key: m[0], label: m[2], better: m[3],
                 count: rows.filter(function (r) { return m[0] in r.metrics; }).length };
      }),
      unclassified: unclassified,
      partial: rows.filter(function (r) { return r.status === "partial"; }).length,
      cross_platform: rows.filter(function (r) { return r.axis_key === "platform"; }).length,
      hero: hero,
    };
    return { summary: summary, rows: rows, repairs: repairs };
  }

  function buildBoard(base) {
    base = base || "./data/";
    var get = function (name) {
      return fetch(base + name).then(function (r) {
        if (!r.ok) throw new Error(name + ": HTTP " + r.status);
        return r.json();
      });
    };
    return Promise.all([get("evaluations.json"), get("repairs.json")])
      .then(function (both) { return deriveBoard(both[0], both[1]); });
  }

  global.deriveBoard = deriveBoard;
  global.buildBoard = buildBoard;
  if (typeof module !== "undefined" && module.exports)
    module.exports = { deriveBoard: deriveBoard, buildBoard: buildBoard };
})(typeof window !== "undefined" ? window : globalThis);
