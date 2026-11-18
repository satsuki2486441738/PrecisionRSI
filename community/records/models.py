"""Model search facets derived from declared metadata and recorded model names."""

import re


UNCLASSIFIED = 'unclassified'
TINY = 'tiny'
UNKNOWN_SIZE = 'unknown'

ARCHITECTURES = (
    (r'\bqwen2\.5[-_ ]?vl\b', 'Qwen2.5-VL'),
    (r'\bqwen2[-_ ]?vl\b', 'Qwen2-VL'),
    (r'\bqwen2(?:\.5)?[-_ ]?moe\b', 'Qwen2-MoE'),
    (r'\bqwen3\.5\b.*(?:\bmoe\b|-a\d+(?:\.\d+)?b\b)', 'Qwen3.5-MoE'),
    (r'\bqwen3\.5\b', 'Qwen3.5'),
    (r'\bqwen3\.8\b', 'Qwen3.8'),
    (r'\bqwen3\b(?!\.).*(?:\bmoe\b|-a\d+(?:\.\d+)?b\b)', 'Qwen3-MoE'),
    (r'\bqwen3\b(?!\.)', 'Qwen3'),
    (r'\bqwen2(?:\.5)?\b(?!\.)', 'Qwen2'),
    (r'\bdeepseek[- ]v4\.1\b', 'DeepSeek V4.1'),
    (r'\bgemma[- ]4\b', 'Gemma 4'),
    (r'\bgemma[- ]3\b', 'Gemma 3'),
    (r'\bsmollm2\b', 'SmolLM2'),
    (r'\bn(?:vidia-)?nemotron[- ]3\.5\b|\bnemotron[- ]3\.5\b', 'Nemotron 3.5'),
    (r'\bgpt-oss\b', 'GPT-OSS'),
    (r'\bgpt[- ]?2\b', 'GPT-2'),
    (r'\bllama\b', 'Llama'),
)
TINY_NAME = re.compile(r'(?<![a-z0-9])tiny(?![a-z0-9])|随机|微型|固定双', re.I)
SIZE_IN_NAME = re.compile(r'-(\d+(?:\.\d+)?[KMBT])(?=-|$)', re.I)
DECLARED_SIZE = re.compile(r'(?:[1-9][0-9]*|(?:0|[1-9][0-9]*)(?:\.[0-9]+)?[KMBT]|tiny|unknown)', re.I)
ACTIVE_EXPERT_SIZE = re.compile(r'-a\d+(?:\.\d+)?b\b', re.I)


def model_group(model, architecture):
    """Derive a browse label from declared or identifiable model topology."""
    name = model.get('name') or ''
    if re.search(r'[-_ ]moe$', architecture, re.I):
        return re.sub(r'[-_ ]moe$', ' MoE', architecture, flags=re.I)
    if re.search(r'[-_ ]dense$', architecture, re.I):
        return re.sub(r'[-_ ]dense$', ' Dense', architecture, flags=re.I)
    if architecture != UNCLASSIFIED and ACTIVE_EXPERT_SIZE.search(name):
        return architecture + ' MoE'
    if architecture in ('Qwen2', 'Qwen3', 'Qwen3.5'):
        return architecture + ' Dense'
    return architecture


def model_facets(model):
    """Return a model architecture and size bucket without rewriting its identity."""
    name = model.get('name') or ''
    architecture = model.get('architecture')
    if 'architecture' in model and (not isinstance(architecture, str) or not architecture.strip()):
        raise ValueError('model.architecture must be a non-empty name')
    if not architecture:
        architecture = next((label for pattern, label in ARCHITECTURES if re.search(pattern, name, re.I)), UNCLASSIFIED)
    if 'size' in model:
        size = model['size']
        if not isinstance(size, str) or not DECLARED_SIZE.fullmatch(size):
            raise ValueError('model.size must be a parameter count, a scale like 0.5B, tiny, or unknown')
        return architecture.strip(), size.lower() if size.lower() in (TINY, UNKNOWN_SIZE) else size.upper()
    if TINY_NAME.search(name):
        return architecture.strip(), TINY
    match = SIZE_IN_NAME.search(name)
    return architecture.strip(), match[1].upper() if match else UNKNOWN_SIZE
