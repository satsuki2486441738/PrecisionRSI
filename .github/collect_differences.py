#!/usr/bin/env python3
"""Collect immutable public survey evidence into the Differences dashboard.

Usage: python -B .github/collect_differences.py --output site/data/differences.json
       --raw-output site/data/differences-raw.json

Standard library only. Downloads public, SHA-pinned evidence; verifies Git blob
hashes and every member of the shared paper archive's SHA-256 manifest. Source
files are deduplicated by content. Measurement views can overlap: their counts
are NOT independent sample or experiment counts. No model execution occurs.
"""
import argparse
import concurrent.futures
import hashlib
import io
import json
import math
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

REPO = "satsuki2486441738/PrecisionRSI"
PINS = {
    "survey/gemma4-gpu-npu-delivery-20261107": "118b50032dabd6f771b3a8391e24cef1d02e9a1a",
    "survey/deepseek41-gpu-npu-delivery-20261108": "0ed4019febaec0d501d74382e054060a54fa30c3",
    "survey/qwen35-gpu-npu-delivery-20261108": "0c1241e220c1a2f2855e86b57f1a5eaffb189d7e",
    "survey/qwen38-gpu-npu-delivery-20261108": "90d9a0218e01c31dce55178601eca52e3a9d4383",
    "survey/model-precision-gpu-npu-delivery-20261110": "ff4c7472f448d179602d4090ba9daf7b7ff2fb88",
}
ARCHIVE_SHA256 = "f66fdb64f996087bac2761a9d25a6d922df655c66c433351427c96ecee14d54f"
PINNED_FILES = [
  {
    "sha": "bd09898fe06d6dae4ca70374f7117cda9224df4c",
    "path": "site/data/evaluations.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107"
    ]
  },
  {
    "sha": "52be5c0568c3ec84fdbd6c6bcf135675241a0f55",
    "path": "site/evidence/gemma4-ascend-tp1-repeat-20261106/report.md",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "1c7adeea5b2ea6e92cd967327d948b09b14f12da",
    "path": "site/evidence/gemma4-ascend-tp1-repeat-20261106/selected-targets.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "4cdcb4f28bdcd304507286d72798d29dac4193d6",
    "path": "site/evidence/gemma4-h100-ascend-batch-padding-20261107/coverage.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "6d043fab8bad981c545fdb88427a3ca6b85c81b0",
    "path": "site/evidence/gemma4-h100-ascend-batch-padding-20261107/execution-cost.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "9fa7d5c4d5ecae47431ecd027e52045dbf9a37da",
    "path": "site/evidence/gemma4-h100-ascend-batch-padding-20261107/independent-checks.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "d237fa314199fd7af8cd1f6104b4d82f832f150b",
    "path": "site/evidence/gemma4-h100-ascend-batch-padding-20261107/report.md",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "358087988796783afcc0c1b30a689290d4da0809",
    "path": "site/evidence/gemma4-h100-ascend-batch-padding-20261107/selected-metrics.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "9edc9c75944b3f16c8ab13da0e2bb93af18e2b3d",
    "path": "site/evidence/gemma4-h100-ascend-batch-padding-20261107/selected-targets.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "6db59bd111f0f88ff39c0579201f06ea4869e7ff",
    "path": "site/evidence/gemma4-h100-ascend-cacheoff-20261106/coverage.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "4ffd33a46a3fc67a21d9c08f942d8b154f8649b4",
    "path": "site/evidence/gemma4-h100-ascend-cacheoff-20261106/report.md",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "f718efd2f944a266ee0f3e85ca497de0253665d1",
    "path": "site/evidence/gemma4-h100-ascend-cacheoff-20261106/selected-metrics.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "92ce588a28d7c791e9e1ea1ba19823443bcd8f72",
    "path": "site/evidence/gemma4-h100-ascend-cacheoff-20261106/selected-targets.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "cafb5d0a5909b380bd9ac74fc575b769df2125ae",
    "path": "site/evidence/gemma4-h100-ascend-fixed-20261106/coverage.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "da010d70bce9f84fc1ffcb4555d2d4c9e6190c47",
    "path": "site/evidence/gemma4-h100-ascend-fixed-20261106/report.md",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "226fe069b134063b68a976e591d9d2aade42f957",
    "path": "site/evidence/gemma4-h100-ascend-fixed-20261106/selected-metrics.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "7bef61a4d582dbce49631ff7299f0bd1e82a9eb8",
    "path": "site/evidence/gemma4-h100-ascend-fixed-20261106/selected-targets.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "bd685edcd5f6d190694215d374491222d89883a4",
    "path": "site/evidence/gemma4-h100-ascend-observer-transparency-20261107/cost.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "fd5fe38fe46a6c9e5dcc9f2d13a654cdbd4dd2a6",
    "path": "site/evidence/gemma4-h100-ascend-observer-transparency-20261107/coverage.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "6f85b9b2bc69c097a6941dd3b8a053a2b888435a",
    "path": "site/evidence/gemma4-h100-ascend-observer-transparency-20261107/e02-summary.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "63e0042a18d73c4f4071701f5f782a97ea5bdf52",
    "path": "site/evidence/gemma4-h100-ascend-observer-transparency-20261107/report.md",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "4395b7cb16ed52e16f34e34c4122644755550666",
    "path": "site/evidence/gemma4-h100-ascend-observer-transparency-20261107/selected-metrics.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "4404c53762af368ca91c5d34cb31a021fa4467a4",
    "path": "site/evidence/gemma4-h100-ascend-observer-transparency-20261107/selected-targets.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "5f2a0a270db44ea8397ecc3a12b601d2ca199567",
    "path": "site/evidence/gemma4-h100-ascend-partitions-20261107/coverage.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "4d300e2d19e46cd5b7984d63801af35a20809997",
    "path": "site/evidence/gemma4-h100-ascend-partitions-20261107/execution-cost.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "ab4a8943c8c0d7d84ecbe775ef475526afe1f23b",
    "path": "site/evidence/gemma4-h100-ascend-partitions-20261107/independent-checks.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "32b6dae9aa0034d0630719e219a98b740d2311de",
    "path": "site/evidence/gemma4-h100-ascend-partitions-20261107/report.md",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "8e933836a799ca0d18d5e7814ad587749c0d1dd8",
    "path": "site/evidence/gemma4-h100-ascend-partitions-20261107/selected-metrics.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "e58584f03a68a74b81ed0fb919f1625e450e49e1",
    "path": "site/evidence/gemma4-h100-ascend-partitions-20261107/selected-targets.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "38a90951ddc43436f51240169f3e2516671e8be3",
    "path": "site/evidence/gemma4-h100-ascend-reserved-p128-20261107/report.md",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "a453e6fd4cbe644ce9e578f48e10e9b88bcc79f4",
    "path": "site/evidence/gemma4-h100-ascend-reserved-p128-20261107/summary.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "46a49ae8eff7d9b4f3945da9062dd13280178954",
    "path": "site/evidence/gemma4-h100-four-independent-20261105.md",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "973011d8848fefe9bb2c21585a35278620b1f01d",
    "path": "site/evidence/gemma4-h100-observer-transparency-20261107/cost.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "e79524a975c4161e47687ee9c1a4c2d4e6817994",
    "path": "site/evidence/gemma4-h100-observer-transparency-20261107/coverage.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "b5e6bed3e4f4be9af6ea768427dd866c7309662d",
    "path": "site/evidence/gemma4-h100-observer-transparency-20261107/e02-summary.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "1bd3c13055594747bed94749fe2c14ef03d28e8a",
    "path": "site/evidence/gemma4-h100-observer-transparency-20261107/report.md",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "e8b25ba9a1f797ab41f749e69e36194ba6262600",
    "path": "site/evidence/gemma4-h100-observer-transparency-20261107/selected-metrics.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "ddbe7dbfd9d143bfa029fcf2272eb8d4e4438f5a",
    "path": "site/evidence/gemma4-h100-observer-transparency-20261107/selected-targets.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "16b7a7f71a9d46f3f0bf404ecbb6d80ad7d3ac0d",
    "path": "site/evidence/gemma4-h100-reserved48-three-bins-20261107/report.md",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "3f87ac77c851da1e39a5049135bf0e43057f8a81",
    "path": "site/evidence/gemma4-h100-reserved48-three-bins-20261107/summary.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "a0a4d043bfff5a606a4c9830c76537277a4a170c",
    "path": "site/evidence/gemma4-h100-rtx5090-pilot-20261105/report.md",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "d923e24c1de6dfad1be399f61ab6c2b8dee88a79",
    "path": "site/evidence/gemma4-h100-rtx5090-pilot-20261105/target-logprob-evidence.json",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "59bd0399855ea54e329f58c249e458a125955ad2",
    "path": "site/evidence/gemma4-scoped-delivery-20261107.md",
    "branches": [
      "survey/gemma4-gpu-npu-delivery-20261107",
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108",
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "c9d5ef30285fc0f2aaf482c383158a3ce3cf1df4",
    "path": "site/evidence/deepseek41-gpu-npu-delivery-20261108.md",
    "branches": [
      "survey/deepseek41-gpu-npu-delivery-20261108"
    ]
  },
  {
    "sha": "41b52517c4171de7666e58003f9bafaf794fb8ec",
    "path": "site/evidence/paper-writing-results-20261108.zip",
    "branches": [
      "survey/deepseek41-gpu-npu-delivery-20261108",
      "survey/qwen35-gpu-npu-delivery-20261108",
      "survey/qwen38-gpu-npu-delivery-20261108"
    ]
  },
  {
    "sha": "2ef1930c3acb76acd22301a3f4b849c8fe76a00b",
    "path": "site/evidence/qwen35-gpu-npu-delivery-20261108.md",
    "branches": [
      "survey/qwen35-gpu-npu-delivery-20261108"
    ]
  },
  {
    "sha": "eb8c72878362e9d55c3e831edfff8ed6547d8037",
    "path": "site/evidence/qwen38-gpu-npu-delivery-20261108.md",
    "branches": [
      "survey/qwen38-gpu-npu-delivery-20261108"
    ]
  },
  {
    "sha": "8514fd92ace70c486157f47f16fbf7447e501902",
    "path": "site/evidence/model-precision-gpu-npu-20261110/README.md",
    "branches": [
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  },
  {
    "sha": "e2e691c85a9e5fe6c0dfe1fe5edaef7764334982",
    "path": "site/evidence/model-precision-gpu-npu-20261110/results.json",
    "branches": [
      "survey/model-precision-gpu-npu-delivery-20261110"
    ]
  }
]
MODELS = {
    "gemma4": "Gemma 4 12B", "qwen35": "Qwen3.5 9B Base",
    "qwen38": "Qwen3.8 Flash Next", "deepseek41": "DeepSeek V4.1 Flash 1017",
    "gemma3": "Gemma 3 27B IT", "gpt-oss": "gpt-oss-120b",
    "nemotron": "Nemotron 3.5 Bolt 74B-A9B BF16", "operators": "Operator references",
}
GLOBAL_NOTES = [
    "Counts describe recorded measurement views, not independent experiments. Aggregate and detailed views may share inputs and observations.",
    "Native/reference comparisons on each platform are not direct GPU–NPU comparisons. Different dtype, layout, software or placement are retained as configuration differences.",
    "Numerical acceptance is unspecified unless explicitly recorded. Missing evidence is not a pass; GPU is not a mathematical oracle.",
    "Only already-public summaries are collected. Private raw tensors, model weights, datasets and execution tools are not included.",
]

def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "PrecisionRSI-survey-collector"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()

def pointer(path):
    return "".join("/" + str(p).replace("~", "~0").replace("/", "~1") for p in path)

def slug(value):
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")

def number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

def model_id(name):
    s = str(name).lower().replace("-", "").replace(" ", "")
    for fragments, key in [(('gemma4',), 'gemma4'), (('gemma3',), 'gemma3'),
                           (('qwen3.5','qwen35'), 'qwen35'), (('qwen3.8','qwen38'), 'qwen38'),
                           (('deepseek',), 'deepseek41'), (('gptoss',), 'gpt-oss'),
                           (('nemotron',), 'nemotron')]:
        if any(x in s for x in fragments):
            return key
    return "operators"

def pretty(value):
    return str(value).replace("_", " ").replace("/", " · ")

def unit(key):
    s = key.lower()
    if "nll" in s or "nat_per_token" in s: return "nat/token"
    if "nat" in s or "js" in s: return "nat"
    if "cosine" in s or "fraction" in s or "overlap" in s: return "fraction"
    if "probability" in s: return "probability"
    if "tokens_per_second" in s: return "tokens/s"
    if "seconds" in s or "latency" in s: return "seconds"
    if "bytes" in s: return "bytes"
    return "recorded units"

class Collector:
    def __init__(self):
        self.sources = []
        self.raw = {}
        self.by_hash = {}
        self.records = []
        self.record_ids = set()
        self.trees = {}
        self.eval_ids = set()
        self.archive_files_verified = 0

    def register(self, content, path, commit, branch, blob=None, archive_member=None):
        sha = hashlib.sha256(content).hexdigest()
        origin = {"branch": branch, "commit": commit, "path": path}
        if archive_member: origin["archive_member"] = archive_member
        if sha in self.by_hash:
            source = self.by_hash[sha]
            if origin not in source["origins"]: source["origins"].append(origin)
            return source
        sid = "source-" + sha[:16]
        source = {"id": sid, "branch": branch, "commit": commit, "path": path,
                  "sha256": sha, "git_blob_sha": blob, "bytes": len(content),
                  "url": f"https://github.com/{REPO}/blob/{commit}/{path}",
                  "raw_url": f"https://raw.githubusercontent.com/{REPO}/{commit}/{path}",
                  "origins": [origin]}
        if archive_member:
            source["archive_member"] = archive_member
            source["url"] = f"https://github.com/{REPO}/blob/{commit}/{path}"
        self.sources.append(source)
        self.by_hash[sha] = source
        if (archive_member or path).endswith(".json"):
            self.raw[sid] = json.loads(content)
        return source

    def record(self, source, path, mid, metrics, label=None, kind="reported-diagnostic", platform="mixed",
               sides=None, scope=None, notes=None, config=None, status="reported", coverage=None):
        rid = slug(mid + "-" + source["id"] + "-" + "-".join(map(str,path)))
        if rid in self.record_ids: return
        self.record_ids.add(rid)
        src = {k: source[k] for k in ("id","branch","commit","path","url")}
        if source.get("archive_member"): src["archive_member"] = source["archive_member"]
        src["json_pointer"] = pointer(path)
        rec = {"id": rid, "model_id": mid, "model": MODELS[mid],
               "label": label or pretty(" / ".join(map(str,path))) or pretty(Path(source.get('archive_member',source['path'])).stem), "comparison_kind": kind,
               "platform": platform, "stage": str(path[-2] if len(path)>1 else path[0] if path else "survey"),
               "mode": str(path[-1]) if path else "summary", "metrics": metrics,
               "scope": scope or "Recorded diagnostic view; inspect the source configuration and coverage before interpreting or combining values.",
               "status": status, "acceptance": None, "source": src, "notes": notes or [],
               "configuration": config or {}, "coverage": coverage or []}
        if sides:
            rec["side_a"], rec["side_b"] = sides[0], sides[1]
        self.records.append(rec)

    def metric(self,key,value,path,label=None,units=None):
        return {"key": key, "label": label or pretty(key), "value": value,
                "unit": units or unit(key), "source_pointer": pointer(path)}

    def evaluations(self, source, only_gemma=False):
        data = self.raw[source['id']]
        for ei,e in enumerate(data.get("evaluations",[])):
            if only_gemma and not e['id'].startswith('gemma4-'): continue
            if e['id'] in self.eval_ids: continue
            self.eval_ids.add(e['id'])
            mid=model_id(e.get('model',{}).get('name',e['id']))
            by_id={s['id']:s for s in e.get('sides',[])}
            for ci,c in enumerate(e.get('comparisons',[])):
                path=['evaluations',ei,'comparisons',ci]
                sides=[by_id.get(x,{'id':x,'label':x}) for x in c.get('side_ids',[])[:2]]
                hardware=' '.join(s.get('hardware','') for s in sides).lower()
                if any('ascend' in s.get('hardware','').lower() for s in sides) and any('nvidia' in s.get('hardware','').lower() for s in sides):
                    kind,platform='cross-backend','GPU ↔ NPU'
                else:
                    kind='within-backend'
                    platform='Ascend' if 'ascend' in hardware else 'GPU' if 'nvidia' in hardware else 'mixed'
                metrics=[]
                for mi,m in enumerate(c.get('metrics',[])):
                    if not number(m.get('value')): continue
                    met=self.metric(m.get('id',str(mi)),m['value'],path+['metrics',mi,'value'],m.get('label'),m.get('unit'))
                    if isinstance(m.get('side_values'),dict):
                        met['side_values']=[{'label':by_id.get(k,{}).get('label',k),'value':v,'source_pointer':pointer(path+['metrics',mi,'side_values',k])} for k,v in m['side_values'].items() if number(v)]
                    for key in ('coordinate','threshold','status'):
                        if key in m: met[key]=m[key]
                    metrics.append(met)
                self.record(source,path,mid,metrics,label=e['title']+' · '+c.get('title',c.get('id','comparison')),
                            kind=kind,platform=platform,sides=sides if len(sides)==2 else None,scope=c.get('scope'),
                            config={'common_conditions':e.get('common',[]),'differences':e.get('differences',[]),'model':e.get('model',{})},
                            notes=e.get('notes',[]),status=e.get('status','reported'),coverage=c.get('coverage',[]))

    def flatten_metrics(self, data, base, prefix=""):
        """Flatten recorded statistics, never coordinates, IDs or tensor payloads."""
        out=[]
        skip={'max_coordinate','max_flat_index','max_values','findings','shape','index','coordinates','mismatch_coordinates','source_sha256','evidence_sha256','scores','predictions','answer','choices','side_a_order','side_b_order','side_a_winners','side_b_winners',
              'count','total','target_count','elements','compared_elements','layer','position','sample','seed','replicates','samples','bootstrap_replicates','warmup','measured','world_size','schema_version','articles','independent_articles','distribution_positions',
              'configuration','configurations','settings','model','hardware','backend_software','source_pins','run','runs','attempt','returncode','numeric_observations_admitted'}
        if not isinstance(data,dict): return out
        for key,value in data.items():
            if key in skip or key.endswith('sha256') or key.endswith('_hash') or 'index_zero_based' in key: continue
            name=(prefix+'.' if prefix else '')+key
            if number(value): out.append(self.metric(name,value,base+[key]))
            elif isinstance(value,dict):
                if number(value.get('value')):
                    out.append(self.metric(name,value['value'],base+[key,'value']))
                elif {'count','total','fraction'} <= value.keys() and number(value['fraction']):
                    m=self.metric(name,value['fraction'],base+[key,'fraction'],units='fraction')
                    m.update(numerator=value['count'],denominator=value['total'])
                    out.append(m)
                elif {'count','total'} <= value.keys() and number(value['count']) and ('agreement' in name or 'top1' in name):
                    m=self.metric(name,value['count'],base+[key,'count'],units='positions')
                    m.update(numerator=value['count'],denominator=value['total'])
                    out.append(m)
                else: out.extend(self.flatten_metrics(value,base+[key],name))
            elif isinstance(value,list) and ('range' in key or 'interval' in key) and len(value)==2 and all(number(v) for v in value):
                for i,suffix in enumerate(('lower','upper')):
                    out.append(self.metric(name+'.'+suffix,value[i],base+[key,i]))
        return out

    def walk_measurements(self, source, obj, path, mid, config=None, notes=None, forced_kind=None, sides=None):
        """Discover standard public comparison outputs at every configuration path."""
        if isinstance(obj,list):
            for i,v in enumerate(obj): self.walk_measurements(source,v,path+[i],mid,config,notes,forced_kind,sides)
            return
        if not isinstance(obj,dict): return
        keys=set(obj)
        recognized=(isinstance(obj.get('metrics'),dict) and any(x in obj['metrics'] for x in ('rmse','cosine','max_abs','error_norm')))
        recognized=recognized or bool(keys & {'mae_nat','probability_rmse','gpu_rmse','rmse'})
        recognized=recognized or ('top1_equal' in keys and 'positions' in keys)
        recognized=recognized or ('native' in keys and 'reference' in keys and any(number(obj[k]) for k in keys))
        recognized=recognized or bool(keys & {'maximum_JS_nat','observed_fixed_over_stock_median_ratio','first_id_agreement_before','max_shared_full_vocabulary_JS_nat','maximum_absolute_target_logprob_difference_nat','fullcache_vs_chunk_logit_max_abs','full_vocab_probability_rmse'})
        if recognized:
            metrics=self.flatten_metrics(obj.get('metrics',obj),path+(['metrics'] if isinstance(obj.get('metrics'),dict) else []))
            if metrics:
                kind=forced_kind or 'reported-diagnostic'; platform='mixed'
                p='/'.join(map(str,path)).lower()
                if 'gpu_rmse' in obj and 'npu_rmse' in obj:
                    kind='operator-reference';platform='GPU ↔ NPU'
                elif forced_kind=='native-reference':
                    platform='CUDA' if 'cuda' in path else 'Ascend'
                elif any(x in p for x in ('repeatability','partition_checks','padding_batch','collector_transparency','continuation')):
                    kind='within-backend';platform='Ascend' if '/npu' in p else 'GPU' if '/gpu' in p else 'mixed'
                elif mid in ('qwen38','deepseek41') and (path[:1]==['numerical'] and path[1] in ('target_scores','reserved_target_scores','structured_scores','targets','target_groups','full_forward_distributions','fixed_history_distributions','by_corpus_and_length','reserved_scores_by_length','reserved_scores_by_sample','structured_scores_by_sample','structured_scores_by_length','structured_distributions','structured_by_length','selected_position_agreement','mhc_layers','blocks','block_by_layer','operator_cross','routing_cross','routing_natural_cross')):
                    kind='cross-backend';platform='GPU ↔ NPU'
                if kind=='operator-reference':
                    pair=[{'label':'GPU reference RMSE','value':obj['gpu_rmse'],'source_pointer':pointer(path+['gpu_rmse'])},{'label':'NPU reference RMSE','value':obj['npu_rmse'],'source_pointer':pointer(path+['npu_rmse'])}]
                    metrics[0]['side_values']=pair
                if 'maximum_JS_nat' in obj and 'cross_peer_distributions' in path:
                    kind,platform='cross-backend','GPU ↔ NPU'
                elif 'observed_fixed_over_stock_median_ratio' in obj:
                    kind,platform='within-backend','Ascend'
                elif 'fullcache_vs_chunk_logit_max_abs' in obj:
                    kind,platform='within-backend','GPU' if 'gpu' in path else 'Ascend'
                for met in metrics:
                    if met['key'] in ('first_id_agreement_before','first_id_agreement_after') and 'positions' in obj:
                        met.update(numerator=met['value'],denominator=obj['positions'],unit='positions')
                    if met['key']=='firstmax_agreement' and 'rows' in obj:
                        met.update(numerator=met['value'],denominator=obj['rows'],unit='positions')
                identity=obj.get('id',obj.get('sample'))
                label=pretty(' / '.join(map(str,path)))+(' · '+str(identity) if identity is not None else '')
                self.record(source,path,mid,metrics,label=label,kind=kind,platform=platform,sides=sides,config=config,notes=notes,
                            scope=obj.get('scope') if isinstance(obj.get('scope'),str) else None,
                            status=obj.get('status','reported'),coverage=obj.get('coverage'))
                return
        for k,v in obj.items():
            if k in ('findings','max_coordinate','mismatch_coordinates','provenance','source_hashes','runtime','source_pins','evidence_hashes','private_selected_evidence','replay'): continue
            self.walk_measurements(source,v,path+[k],mid,config,notes,forced_kind,sides)

    def model_campaign(self,source):
        for platform,models in self.raw[source['id']]['platforms'].items():
            for name,e in models.items():
                base=['platforms',platform,name];mid=model_id(e['model']['repository'])
                sides=[{'id':k,'label':k.title(),'hardware':f"{s['count']} × {s['accelerator']}",'libraries':s['libraries'],'settings':{'backend':s['backend']}} for k,s in e['sides'].items()]
                config={'common_conditions':['Same checkpoint revision and frozen inputs; native/reference peers use the same assigned accelerator set within this platform.'],'differences':['Native backend versus reference backend. This is not a direct GPU–NPU comparison.'],'model':e['model']}
                for mode,m in e['text'].items():
                    p=base+['text',mode];metrics=self.flatten_metrics(m,p)
                    for met in metrics:
                        if met['key']=='nll_abs_diff_nat_per_token':
                            met['side_values']=[{'label':'Native','value':m['side_a_nll_nat_per_token'],'source_pointer':pointer(p+['side_a_nll_nat_per_token'])},{'label':'Reference','value':m['side_b_nll_nat_per_token'],'source_pointer':pointer(p+['side_b_nll_nat_per_token'])}]
                    self.record(source,p,mid,metrics,label=pretty(mode),kind='native-reference',platform='CUDA' if platform=='cuda' else 'Ascend',sides=sides,config=config,
                                scope='Native versus reference on the same assigned devices within one platform; exact stage and execution mode retained.',notes=e.get('notes',[]),status=e['status'])
                held=e['heldout'];p=base+['heldout'];metrics=[]
                delta_key=next(k for k in held if 'delta' in k and number(held[k]))
                metrics.append(self.metric('heldout-delta-nll',held[delta_key],p+[delta_key],'Held-out mean native minus reference NLL','nat/token'))
                interval=next((k for k in held if 'interval' in k),None)
                if interval:
                    for i,label in enumerate(('95% interval lower','95% interval upper')):
                        metrics.append(self.metric('heldout-ci-'+str(i),held[interval][i],p+[interval,i],label,'nat/token'))
                self.record(source,p,mid,metrics,label='Held-out ΔNLL · 48 reserved requests',kind='native-reference',platform='CUDA' if platform=='cuda' else 'Ascend',sides=sides,config=config,
                            scope=held.get('statistic','Equal-input native minus reference NLL, equally weighted length bins; not pooled token NLL.'),notes=e.get('notes',[]),status=e['status'])
                # Preserve auxiliary comparisons individually. Before/after interventions
                # and same-operand replays are not relabeled as whole-model pairs.
                self.walk_measurements(source,e.get('auxiliary',{}),base+['auxiliary'],mid,config,e.get('notes',[]))
                for key in ('quality','performance','modalities'):
                    auxiliary=e.get('auxiliary',{}).get(key)
                    if not auxiliary: continue
                    ap=base+['auxiliary',key];ams=[]
                    if key=='modalities':
                        for i,v in enumerate(auxiliary.get('exact_answers',[])):
                            met=self.metric('exact-answers-'+str(i),v,ap+['exact_answers',i],sides[i]['label']+' exact answers','answers')
                            met.update(numerator=v,denominator=auxiliary.get('cases'))
                            ams.append(met)
                    else:
                        peers=auxiliary.get('sides',[])
                        for i,peer in enumerate(peers):
                            for met in self.flatten_metrics(peer,ap+['sides',i]):
                                if key=='quality' and met['key']=='correct':
                                    met.update(numerator=met['value'],denominator=peer['total'],unit='answers')
                                if key=='performance' and i==0 and len(peers)==2:
                                    relative=met['source_pointer'][len(pointer(ap+['sides',i])):]
                                    other=peers[1]
                                    for part in relative[1:].split('/'):
                                        other=other[part.replace('~1','/').replace('~0','~')]
                                    met['side_values']=[{'label':'Native','value':met['value'],'source_pointer':met['source_pointer']},{'label':'Reference','value':other,'source_pointer':pointer(ap+['sides',1])+relative}]
                                met['label']=sides[i]['label']+' · '+met['label']
                                met['key']=key+'-'+str(i)+'-'+met['key']
                                ams.append(met)
                        if key=='quality' and len(peers)==2:
                            ams[0]['side_values']=[{'label':sides[i]['label'],'value':v['correct'],'source_pointer':pointer(ap+['sides',i,'correct'])} for i,v in enumerate(peers)]
                    extra_config={**config,'measurement_coverage':{k:auxiliary[k] for k in ('cases','by_modality','warmup','measured') if k in auxiliary}}
                    self.record(source,ap,mid,ams,label=pretty(key)+' · native / reference',kind='native-reference',platform='CUDA' if platform=='cuda' else 'Ascend',sides=sides,config=extra_config,
                                scope='Recorded paired '+key+' probe; preserve the source sample count and measured workload. Small quality probes do not establish general accuracy.',notes=e.get('notes',[]),status=e['status'])
                self.record(source,base+['tasks'],mid,[],label='Task coverage · measured, complete and deferred scopes',platform='CUDA' if platform=='cuda' else 'Ascend',
                            config={**config,'task_status':{k:v['status'] for k,v in e.get('tasks',{}).items()}},status='coverage',scope='Recorded task coverage; measured and complete tasks are distinguished from deferred or missing numerical chains.')

    def run(self):
        # This frozen manifest was discovered from complete recursive trees for
        # all five pinned commits. No live API/rate-limit-dependent discovery.
        jobs={}
        for item in PINNED_FILES:
            for branch in item['branches']:
                jobs.setdefault(item['sha'],[]).append((branch,item['path'],item))
        def download(group):
            branch,path,item=group[0];commit=PINS[branch]
            content=fetch(f'https://raw.githubusercontent.com/{REPO}/{commit}/{path}')
            blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
            if blob!=item['sha']: raise ValueError('Git blob mismatch: '+path)
            return group,content
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            downloaded=list(pool.map(download,jobs.values()))
        archive=None
        for group,content in downloaded:
            for branch,path,item in group:
                source=self.register(content,path,PINS[branch],branch,item['sha'])
            if path.endswith('.zip'): archive=(source,content)
        if archive is None: raise ValueError('Missing shared paper archive')
        source,content=archive
        if hashlib.sha256(content).hexdigest()!=ARCHIVE_SHA256: raise ValueError('Archive SHA-256 mismatch')
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            manifest=json.loads(z.read('manifest.json'))
            for name,expected in manifest.items():
                if hashlib.sha256(z.read(name)).hexdigest()!=expected: raise ValueError('Archive member mismatch: '+name)
            self.archive_files_verified=len(manifest)
            for name in sorted(z.namelist()):
                if name.startswith('sources/') and name.endswith('.json'):
                    member=self.register(z.read(name),source['path'],source['commit'],source['branch'],archive_member=name)
                    for origin in source['origins']:
                        self.register(z.read(name),origin['path'],origin['commit'],origin['branch'],archive_member=name)
        # Explicit evaluation hierarchy first; one copy per recorded evaluation ID.
        for s in list(self.sources):
            if s['path']=='site/data/evaluations.json': self.evaluations(s,only_gemma=True)
            if s.get('archive_member')=='sources/model-evaluations.json': self.evaluations(s)
        for s in list(self.sources):
            if s['id'] not in self.raw: continue
            path=s.get('archive_member',s['path']);d=self.raw[s['id']]
            if path=='site/data/evaluations.json' or path=='sources/model-evaluations.json': continue
            if path.endswith('model-precision-gpu-npu-20261110/results.json'):
                self.model_campaign(s); continue
            if 'gemma4-' in path:
                # Explicit evaluation comparisons already cover primary metrics;
                # structured source views retain additional per-mode/boundary data.
                if path.endswith(('independent-checks.json','e02-summary.json','selected-targets.json','target-logprob-evidence.json')):
                    continue  # replay views of primary measurements; original bytes remain in raw sources
                if path.endswith('gemma4-h100-ascend-reserved-p128-20261107/summary.json'):
                    for i,family in enumerate(d['pair_families']):
                        pair=family['pair'];sides=[{'label':pair[k+'_backend']+' / '+pair[k+'_mode'],'hardware':pair[k+'_backend']} for k in ('a','b')]
                        for key in ('all_target_metrics','continuation_only_metrics','distribution_aggregate'):
                            p=['pair_families',i,key]
                            self.record(s,p,'gemma4',self.flatten_metrics(family[key],p),label='Reserved P128 · '+sides[0]['label']+' ↔ '+sides[1]['label']+' · '+pretty(key),
                                        kind='cross-backend' if pair['a_backend']!=pair['b_backend'] else 'within-backend',platform='GPU ↔ NPU' if pair['a_backend']!=pair['b_backend'] else pair['a_backend'],sides=sides,
                                        scope='12 paired reserved articles, P128 + 32 forced continuation tokens. The 9 families share these same 12 inputs; not independent cohorts.',
                                        config={'hardware':d['hardware'],'libraries':d['backend_software'],'settings':d['settings']},notes=['P512/P2048/P8192 on Ascend remain unpaired. All49 full-plan completion flags remain false.'])
                    continue
                if path.endswith('gemma4-h100-reserved48-three-bins-20261107/summary.json'):
                    for i,group in enumerate(d['selected_within_gpu_metrics']):
                        p=['selected_within_gpu_metrics',i]
                        self.record(s,p,'gemma4',self.flatten_metrics(group,p),label='H100 '+group['modes'][0]+' ↔ '+group['modes'][1]+' · 36/48 reserved articles',kind='within-backend',platform='GPU',
                                    scope=d['scope'],config={'coverage':d['counts'],'bins':d['bins']},notes=['Only36 of48 planned articles across P128/P512/P2048. Both P8192 attempts ended before numerical admission. MAE endpoints are per-article extrema, not a pooled mean.'])
                    self.record(s,['p8192_attempts'],'gemma4',[],label='H100 reserved P8192 · no numerical admission',kind='within-backend',platform='GPU',status='incomplete',scope='Both attempts ended before numerical admission; interruption cause remains unknown. This is not a numerical pass.')
                    continue
                self.walk_measurements(s,d,[], 'gemma4',notes=['Detailed source view; may overlap evaluation summaries.'])
            elif 'qwen38-precision' in path:
                config={'common_conditions':['Frozen model revision and inputs; cohort-specific settings recorded in the source.'],'differences':['Final selected inference uses GPU automatic placement versus NPU TP8. Historical reserved, training, RL and performance use the original automatic-placement cohort; these configurations must not be pooled.'],'model':d.get('model',{}),'execution':d.get('numerical',{}).get('inference_execution_configuration',{})}
                self.walk_measurements(s,d.get('numerical',{}),['numerical'],'qwen38',config,d.get('limits',[])+['Selected fixed-history token agreement is not generation accuracy; historical and TP8 cohorts differ.'])
                for key in ('serving',):
                    self.record(s,[key],'qwen38',[],label='Serving admissions and missing coverage',config=config,status='incomplete',scope='Two NPU engine admissions failed before scoring; no numerical serving pass is implied.')
            elif 'deepseek41-precision' in path:
                config={'common_conditions':['Same pinned model and frozen inputs as recorded.'],'differences':['Original GPU mixed-FP8/FP4 TP4 versus adapted NPU BF16 TP8/EP8 with CPU offload; not a hardware-only effect.'],'backends':d.get('configurations',{})}
                for key in ('numerical','bounded_architecture','head_transparency','performance'):
                    self.walk_measurements(s,d.get(key,{}),[key],'deepseek41',config,[GLOBAL_NOTES[2]])
            elif path=='sources/qwen-diagnostics.json':
                for key,mid in [('Qwen35','qwen35'),('Qwen38','qwen38'),('DeepSeek41','deepseek41')]:
                    section=d.get(key,{})
                    self.walk_measurements(s,section,[key],mid,{'configuration':section.get('configuration',{}),'model_revision':section.get('model_revision')},['Interim diagnostic snapshot; preserve the primary report scope.'])
            elif path=='sources/qwen-followup.json':
                self.walk_measurements(s,d.get('secondary_cache',{}),['secondary_cache'],'qwen35',{'configuration':d.get('secondary_cache',{}).get('configurations',{})},['Secondary-host cohort; does not replace the missing primary-host NPU cache capture.'])
                self.walk_measurements(s,d.get('q38_matched_cost',{}),['q38_matched_cost'],'qwen38',{'configuration':d.get('q38_matched_cost',{}).get('configuration',{})},['Order and configuration limit cost interpretation.'])
            else:
                mid='operators' if 'operator-' in path else 'deepseek41' if 'deepseek' in path else 'qwen35'
                self.walk_measurements(s,d,[],mid,{'source_scope':d.get('scope',{})} if isinstance(d,dict) else {},['Supplementary diagnostic view; shared samples and coordinates are dependent.'])

    def write(self,output,raw_output):
        self.sources.sort(key=lambda x:(x['path'],x.get('archive_member',''),x['id']))
        self.records.sort(key=lambda x:(x['model_id'],x['comparison_kind'],x['label'],x['id']))
        raw={'schema_version':1,'sources':self.raw}
        payload={'schema_version':1,'updated_at':'2026-11-13','snapshot_date':'2026-11-13',
                 'branches':[{'name':b,'branch':b,'commit':c,'url':f'https://github.com/{REPO}/tree/{c}'} for b,c in PINS.items()],
                 'models':[{'id':k,'label':v} for k,v in MODELS.items() if any(r['model_id']==k for r in self.records)],
                 'records':self.records,'sources':self.sources,'notes':GLOBAL_NOTES,
                 'raw_results_url':'./data/'+raw_output.name,
                 'coverage':{'branches':len(PINS),'unique_source_files':len(self.sources),'raw_json_sources':len(self.raw),
                             'recorded_views':len(self.records),'numeric_metrics':sum(len(r['metrics']) for r in self.records),
                             'by_comparison_kind':dict(Counter(r['comparison_kind'] for r in self.records)),
                             'archive_files_verified':self.archive_files_verified,'archive_sha256':ARCHIVE_SHA256,
                             'counting_unit':'Recorded measurement views; overlapping views are not independent experiments.'}}
        for path,obj in ((output,payload),(raw_output,raw)):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8')
        print(json.dumps(payload['coverage'],indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('site/data/differences.json'))
    parser.add_argument('--raw-output',type=Path,default=Path('site/data/differences-raw.json'))
    args=parser.parse_args()
    collector=Collector();collector.run();collector.write(args.output,args.raw_output)

if __name__=='__main__': main()
