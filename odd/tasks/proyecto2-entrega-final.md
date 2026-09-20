# Project 2 final delivery — Deliverables 1 and 3

## Objective

Prepare and verify only the executed notebook (Deliverable 1) and the public
Streamlit MVP link (Deliverable 3) for Project 2.

## Authorized scope

- Make `notebooks/proyecto2.ipynb` reproducible from a clean Google Colab T4
  runtime within the required 30-minute limit.
- Remove the unimplemented processed-data fallback and preserve a robust raw
  data path.
- Verify the existing Streamlit MVP locally and deploy it from branch
  `proyecto2` only when the authorized existing Streamlit Cloud/GitHub session
  is actually available.
- Create the required URL-only `.txt` delivery file only after the public MVP
  URL is confirmed.

## Explicit exclusion

The executive-report PDF is out of scope. Do not create, edit, or commit any
PDF report.

## Constraints

- Preserve the existing MVP artifact-reading contract in `mvp/app.py`.
- Do not fabricate notebook outputs, T4 measurements, public URLs, or remote
  deployment evidence.
- TDD is disabled, as recorded in `odd/tasks/mvp-streamlit.md`; use ordinary
  functional checks instead.
- Preserve unrelated untracked `.atl/`, `.codegraph/`, and `docs/` files.
- Do not push, open a PR, or merge changes.

## Acceptance criteria

1. The notebook has no unimplemented processed-data route and runs end-to-end
   from a clean Colab T4 runtime without exceptions in under 30 minutes.
2. Its executed cells show the required data, model, ablation, and artifact
   outputs.
3. The Streamlit app starts locally and loads the committed artifacts.
4. The MVP is publicly reachable from Streamlit Cloud and a `.txt` file contains
   only its confirmed URL.

## Delivery strategy

`single-pr`. Keep each completed work unit in a focused Conventional Commit on
`proyecto2`; do not create a pull request in this task.

## Estimated change size

Approximately 80–130 changed lines, primarily notebook source and delivery
evidence; no generated outputs will be invented.

## Tasks

- [x] T1 — Record the authorized scope, exclusions, acceptance criteria, and
  verification plan in this feature document.
- [x] T2 — Remove the dead processed-data route and apply the smallest
  evidence-based runtime reduction while retaining the two-stage, five-config,
  three-seed methodology.
- [ ] T3 — Run available local notebook and artifact checks; execute the full
  notebook in the authorized Colab T4 session if it is available and record the
  real duration and outputs.
- [x] T4 — Run the Streamlit MVP locally against committed artifacts.
- [ ] T5 — Deploy from `proyecto2` through the authorized Streamlit Cloud/GitHub
  session, confirm the public URL, and create the URL-only delivery file.

## Verification log

| Task | Command or observation | Result |
| --- | --- | --- |
| T1 | Read `docs/CC3092_Proyecto2.md`, `docs/Documentacion_Notebook.md`, `mvp/`, and the notebook's recorded execution. | Confirmed scope: Deliverables 1 and 3 only. The recorded notebook execution is 1966.1 s (32.8 min) and ends in an `AssertionError`; its `PROCESSED_URL` is a TODO marker that falls back to raw data. |
| T2 | `uv run --with nbformat python -c 'from pathlib import Path; import nbformat; path=Path("notebooks/proyecto2.ipynb"); notebook=nbformat.read(path, as_version=4); source="\n".join("".join(cell.source) for cell in notebook.cells if cell.cell_type == "code"); assert "TODO_RELEASE_ASSET_URL" not in source; assert "PROCESSED_URL" not in source; assert "REBUILD_FROM_RAW" not in source; assert "EPOCHS_B=10" in source; nbformat.validate(notebook); print(f"validated {len(notebook.cells)} cells; raw-only data path and EPOCHS_B=10 confirmed")'` | Passed: 55 cells validate; the notebook contains no `TODO_RELEASE_ASSET_URL`, `PROCESSED_URL`, or `REBUILD_FROM_RAW` source and uses the direct Kaggle raw-data path. The recorded per-epoch ablation timings project that reducing Stage B from 12 to 10 epochs should recover the 2.8-minute overrun, but this is not a T4 measurement. |
| T3 | Authorized Colab T4 execution | Blocked: this environment exposes no Google Colab/browser session tool. The notebook is prepared for a clean T4 run, but no T4 duration or new outputs can be claimed until that session is accessible. |
| T4 | `uv run --with-requirements mvp/requirements.txt python -c 'from streamlit.testing.v1 import AppTest; at=AppTest.from_file("mvp/app.py").run(); assert not at.exception, at.exception; assert len(at.radio)==1 and len(at.selectbox)==1 and len(at.metric)==4 and len(at.dataframe)==1; assert "ALERTA" in at.metric[2].value; print("default alert case rendered with 4 metrics and sequence table"); [(at.radio[0].set_value(value).run(), (not at.exception) or (_ for _ in ()).throw(AssertionError(at.exception)), print(f"{value}: rendered")) for value in ["Todos", "Solo alertados", "Solo lavado confirmado", "Falsos negativos"]]'` | Passed: the default alert case rendered four metrics and a sequence table. `Todos`, `Solo alertados`, `Solo lavado confirmado`, and `Falsos negativos` each rendered without exceptions. |
| T4 | `uv run --with nbformat python -c 'from pathlib import Path; import json; artifacts=Path("notebooks/artifacts"); required=["mvp_data.json", "metrics.json", "preprocessing.json"]; assert all((artifacts/name).is_file() for name in required); records=json.loads((artifacts/"mvp_data.json").read_text()); metrics=json.loads((artifacts/"metrics.json").read_text()); preprocessing=json.loads((artifacts/"preprocessing.json").read_text()); assert records and "official_threshold" in metrics and "max_len" in preprocessing; print(f"artifacts valid: {len(records)} MVP records, max_len={preprocessing[\"max_len\"]}")'` | Passed: all required artifacts exist and parse; `mvp_data.json` contains 2,000 records and `preprocessing.json` reports `max_len=29`. |
| T5 | Authorized Streamlit Cloud/GitHub deployment | Blocked: this environment exposes no Streamlit Cloud/GitHub session tool. No public URL or URL-only `.txt` file was created. Deployment must use branch `proyecto2`, main file `mvp/app.py`, and the existing authorized session. |

## Work-unit evidence

| Task | Commit | Focused check | Runtime check | Rollback boundary |
| --- | --- | --- | --- | --- |
| T1 | `ea9aaa3` | Markdown structure reviewed. | N/A — planning document only. | Remove `odd/tasks/proyecto2-entrega-final.md`. |
| T2 | `bec0330` | `uv run --with nbformat …` passed; `git diff --check` passed. | Full notebook run is pending authorized Colab T4 access. | Revert the raw-path simplification and Stage B epoch setting in `notebooks/proyecto2.ipynb`. |
| T4 | `3839496` | Streamlit `AppTest` passed across all four filters; artifact parse check passed. | Same `AppTest` renders the application boundary without exceptions. | Revert this verification evidence only; no application source changed. |
