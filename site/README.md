# Repair history website

GitHub Pages serves this directory. Preview from the repository root with `python3 -m http.server 8000 --directory site`, then open `http://localhost:8000`.

## Record a repair

Append an entry to `data/repairs.json`. Give each repair a stable ID and date, model identity, actual hardware/dependency versions, trigger, cause, change, verification stages and evidence links. A later repair experiment gets a new entry; preserve earlier measurements and their evidence. Use `status: "validated"` for completed verification and keep `validation.acceptance` separate (`null` means pending user judgment).

Metrics carry before/after values and units. State observed/expected coverage and verification scope. Keep unattributed residual differences in `validation.remaining`. Publish concise evidence under `evidence/`; retain full execution artifacts with their private cases. Add libraries, environments and models through data entries without changing the page.

Run `python3 -B .github/validate_history.py` before publishing. The Pages workflow validates data, uploads only `site/` and deploys on changes to `main`.
