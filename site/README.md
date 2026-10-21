# Repair history website

GitHub Pages serves this directory. Preview from the repository root with `python3 -m http.server 8000 --directory site`, then open `http://localhost:8000`.

## Record a repair

Organize `data/repairs.json` as library groups → independent fixes → verification records. Declare each group in `library_groups` with a stable `id`, `title` and `components` (one library or an interacting combination). An independent repair sets `library_group_id`; `affected_library` identifies where the defect is located. A verification follow-up sets `related_repair_id` to the independent repair and inherits its group. `verification_title` labels that round of verification.

Append an entry to `repairs` for each experiment. Give each repair a stable ID and date, model identity, actual hardware/dependency versions, trigger, cause, change, verification stages and evidence links. A later verification gets a new entry linked to the same repair; a separate defect gets a new independent repair entry. Preserve earlier measurements, IDs, links and evidence. The page counts independent fixes separately from verification rounds. Use `status: "validated"` for completed verification and keep `validation.acceptance` separate (`null` means pending user judgment).

Metrics carry before/after values and units. State observed/expected coverage and verification scope. Keep unattributed residual differences in `validation.remaining`. Publish concise evidence under `evidence/`; retain full execution artifacts with their private cases. Add libraries, environments and models through data entries without changing the page.

Run `python3 -B .github/validate_history.py` before publishing. The Pages workflow validates data, uploads only `site/` and deploys on changes to `main`.
