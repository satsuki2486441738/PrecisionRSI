# Hardware and model identity marks

These marks identify the hardware vendor in configuration selectors. They are
local static assets; the page does not contact vendor servers when rendering.

- `nvidia.svg`: the unchanged **Eye Mark** path from NVIDIA's official header
  `n24-nvidia-logo` symbol, with a viewBox around that mark.
  Source: https://www.nvidia.com/en-us/about-nvidia/legal-info/logo-brand-usage/
- `ascend.svg`: the unmodified Ascend community header logo.
  Source: https://www.hiascend.com/_static3/logo1.BQ2XZIjU.svg

Retrieved 2026-11-17. Brand marks belong to their respective owners and are used
for identification. CPU uses a neutral inline chip icon.

## Model families

`model-logos.svg` bundles the Qwen, DeepSeek, Gemma, Hugging Face, Meta and
OpenAI static marks from Lobe Icons. Source revision: `34270bc208d7122778dc00b30bd460f97ea03216`.
Source: https://github.com/lobehub/lobe-icons/tree/34270bc208d7122778dc00b30bd460f97ea03216/packages/static-svg/icons

The upstream MIT notice is retained in `LOBE-ICONS-LICENSE.txt`. Internal gradient
IDs are namespaced when assembling the SVG symbols. Nemotron reuses NVIDIA’s mark;
unnamed synthetic fixtures use a neutral model icon. Marks identify model families,
including explicitly named random/tiny configurations; the model labels retain
the actual configuration. All assets are served locally.
