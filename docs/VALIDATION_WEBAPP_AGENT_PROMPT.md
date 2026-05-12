# Agent instructions — implement one phase of the validation web app

> **You are reading this file because the user told you to.** They will give you a single phase number (0–5). Implement **only** that phase, validate it, and stop.

---

## 0. Orientation — read these first, in order

1. `docs/VALIDATION_WEBAPP_PLAN.md` — full plan; this is the source of truth.
2. `docs/VALIDATION_WEBAPP_PROGRESS.md` — current status. **Verify every earlier phase is fully ticked (both agent + human boxes) before you start.** If an earlier phase is incomplete, stop and tell the user.
3. `README.md` — pipeline outputs and `runs/<run>/` layout.
4. The actual `runs/` directories (`hdbscan`, `kmeans_coarse`, `leiden_k50_res1`) — only the files relevant to your phase. Use `ls -lh` first; do **not** load multi-GB `.npy` intermediates into memory.

Then re-read **section 4 → "Phase <N>"** of the plan, including its **Validation** subsection, and write a short todo list before coding.

---

## 1. Hard rules

- Implement **only** the requested phase. If you finish early, stop. Do not start the next phase, do not "while I'm here" cleanups.
- Webapp code is **additive only**. Do not modify existing pipeline code under `src/embedcluster/{cli,config,io,validation,gpu,preprocessing,metrics,export,neighbors,graph,pipelines}` or `scripts/run_{leiden,hdbscan,kmeans}.py`.
- Do not add dependencies beyond `requirements-webapp.txt` unless the phase explicitly calls for one. If you must, add it with a loose lower-bound pin and justify it in your final summary.
- Do not run `git commit`, `git push`, or any state-changing git command.
- Do not install or modify CUDA / RAPIDS / NVIDIA drivers.
- Follow the global rules in `~/.claude/CLAUDE.md`:
  - Conda env discipline. If no env is active, ask the user which to use. Use `python -m pip`, never bare `pip`.
  - Never read multi-GB files fully — `np.load(..., mmap_mode='r')` and sample.
  - Loose version pins only.
- Match existing repo style. Minimal, surgical edits. No speculative abstractions, no decorative comments.

---

## 2. Validation — required before you tick anything

Every phase has both **agent self-validation** and **human validation** steps in `VALIDATION_WEBAPP_PLAN.md` under that phase. You must:

1. Run **all agent self-validation steps** for your phase. They must all pass.
2. If something fails, fix it. If you cannot fix it (missing input, environment issue, ambiguous spec), **stop and ask the user** — do not work around it silently.
3. Only after every agent self-validation step passes may you tick **the agent box** for your phase in `VALIDATION_WEBAPP_PROGRESS.md`.
4. **Never tick the human validation box.** That is the user's job. Leave it as `[ ]`.

If a self-validation step requires the live UI and you cannot launch Streamlit from your environment (e.g. headless, no display, port blocked), say so explicitly in your final summary — do **not** tick the agent box for any UI-dependent step you could not run. List exactly what the user needs to verify manually.

Always-applicable agent self-validation (in addition to the per-phase steps):

- `python -c "import embedcluster.webapp"` and imports for any submodule you touched succeed with no errors.
- `pytest -q` passes (existing pipeline tests). If there are no tests yet for the webapp, that's fine — do not invent tests unless the phase requires them.
- `git status` shows only the files you intended to add/edit — no stray artifacts, no `__pycache__` committed, no `.streamlit` cache files.

---

## 3. When finished

1. Edit `docs/VALIDATION_WEBAPP_PROGRESS.md`:
   - Tick the **agent self-validation** checkbox for this phase.
   - Leave the **human validation** checkbox unchecked.
   - Add a 1–3 line note covering: what was deferred, any deviation from the plan, and any UI-dependent steps you could not run.
2. Reply to the user with:
   - A 3–6 line summary of what changed.
   - Files added / edited (paths only).
   - The exact list of human-validation steps the user still needs to run (copy them from the plan).
   - Any acceptance criteria you could **not** verify and why.

If the plan itself was wrong or ambiguous, surface that in your summary — do not silently improvise. The user will update the plan and re-run you.
