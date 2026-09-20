# Project 2 final delivery — Deliverables 1, 2, and 3

## Objective

Prepare and verify the executed notebook (Deliverable 1), the executive PDF
report (Deliverable 2), and the public Streamlit MVP link (Deliverable 3) for
Project 2.

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
- Create and verify `docs/Reporte_Ejecutivo_Proyecto2.pdf` as the executive
  report deliverable.

## Constraints

- Preserve the existing MVP artifact-reading contract in `mvp/app.py`.
- Do not fabricate notebook outputs, T4 measurements, public URLs, or remote
  deployment evidence.
- TDD is disabled, as recorded in `odd/tasks/mvp-streamlit.md`; use ordinary
  functional checks instead.
- Preserve unrelated untracked `.atl/`, `.codegraph/`, and `docs/` files.
- Do not push, open a PR, or merge changes.

## Executive-report requirements

- Write 2,000-3,000 body words in professional, neutral Spanish, excluding
  references.
- Cover business and legislative context, model design, the ablation, five
  named cases, limitations, and a production path.
- Include at least three academic references from 2020-2025 and an AI-use
  declaration of no more than 200 words.
- Validate text extraction, body-word count, and rendered visual legibility
  before marking the report task complete.

## Acceptance criteria

1. The notebook has no unimplemented processed-data route and runs end-to-end
   from a clean Colab T4 runtime without exceptions in under 30 minutes.
2. Its executed cells show the required data, model, ablation, and artifact
   outputs.
3. The Streamlit app starts locally and loads the committed artifacts.
4. The MVP is publicly reachable from Streamlit Cloud and a `.txt` file contains
   only its confirmed URL.
5. The executive report PDF satisfies all executive-report requirements and is
   readable after extraction and rendering validation.

## Delivery strategy

`single-pr`. Keep each completed work unit in a focused Conventional Commit on
`proyecto2`; do not create a pull request in this task.

## Estimated change size

Approximately 350 changed lines plus the generated PDF, primarily report source
and delivery evidence; no experimental outputs will be invented.

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
- [x] T5 — The user confirmed the public Streamlit URL opens correctly in an
    incognito window; create the URL-only delivery file.
- [x] T6 — Create and verify the executive PDF report, including factual scope,
   references, AI-use declaration, text extraction, word count, and visual review.
- [x] T7 — Convert the executive report to LaTeX with
  `docs/Reporte_Ejecutivo_Proyecto2.tex` as the sole canonical source; regenerate
  and verify the PDF without changing factual content. The first section and its
  label must be exactly `Resumen`, never `Resumen ejecutivo`.
- [x] T8 — Add and link the confirmed public repository and MVP resources in the
  executive report, then regenerate and verify the PDF.

## Verification log

| Task | Command or observation | Result |
| --- | --- | --- |
| T1 | Read `docs/CC3092_Proyecto2.md`, `docs/Documentacion_Notebook.md`, `mvp/`, and the notebook's recorded execution. | Confirmed scope: Deliverables 1 and 3 only. The recorded notebook execution is 1966.1 s (32.8 min) and ends in an `AssertionError`; its `PROCESSED_URL` is a TODO marker that falls back to raw data. |
| T2 | `uv run --with nbformat python -c 'from pathlib import Path; import nbformat; path=Path("notebooks/proyecto2.ipynb"); notebook=nbformat.read(path, as_version=4); source="\n".join("".join(cell.source) for cell in notebook.cells if cell.cell_type == "code"); assert "TODO_RELEASE_ASSET_URL" not in source; assert "PROCESSED_URL" not in source; assert "REBUILD_FROM_RAW" not in source; assert "EPOCHS_B=10" in source; nbformat.validate(notebook); print(f"validated {len(notebook.cells)} cells; raw-only data path and EPOCHS_B=10 confirmed")'` | Passed: 55 cells validate; the notebook contains no `TODO_RELEASE_ASSET_URL`, `PROCESSED_URL`, or `REBUILD_FROM_RAW` source and uses the direct Kaggle raw-data path. The recorded per-epoch ablation timings project that reducing Stage B from 12 to 10 epochs should recover the 2.8-minute overrun, but this is not a T4 measurement. |
| T3 | Authorized Colab T4 execution | Blocked: this environment exposes no Google Colab/browser session tool. The notebook is prepared for a clean T4 run, but no T4 duration or new outputs can be claimed until that session is accessible. |
| T4 | `uv run --with-requirements mvp/requirements.txt python -c 'from streamlit.testing.v1 import AppTest; at=AppTest.from_file("mvp/app.py").run(); assert not at.exception, at.exception; assert len(at.radio)==1 and len(at.selectbox)==1 and len(at.metric)==4 and len(at.dataframe)==1; assert "ALERTA" in at.metric[2].value; print("default alert case rendered with 4 metrics and sequence table"); [(at.radio[0].set_value(value).run(), (not at.exception) or (_ for _ in ()).throw(AssertionError(at.exception)), print(f"{value}: rendered")) for value in ["Todos", "Solo alertados", "Solo lavado confirmado", "Falsos negativos"]]'` | Passed: the default alert case rendered four metrics and a sequence table. `Todos`, `Solo alertados`, `Solo lavado confirmado`, and `Falsos negativos` each rendered without exceptions. |
| T4 | `uv run --with nbformat python -c 'from pathlib import Path; import json; artifacts=Path("notebooks/artifacts"); required=["mvp_data.json", "metrics.json", "preprocessing.json"]; assert all((artifacts/name).is_file() for name in required); records=json.loads((artifacts/"mvp_data.json").read_text()); metrics=json.loads((artifacts/"metrics.json").read_text()); preprocessing=json.loads((artifacts/"preprocessing.json").read_text()); assert records and "official_threshold" in metrics and "max_len" in preprocessing; print(f"artifacts valid: {len(records)} MVP records, max_len={preprocessing[\"max_len\"]}")'` | Passed: all required artifacts exist and parse; `mvp_data.json` contains 2,000 records and `preprocessing.json` reports `max_len=29`. |
| T5 | User-provided confirmation: `https://labsdeep-proyecto2.streamlit.app/` opens correctly in an incognito window. | Confirmed by the user; `docs/Enlace_MVP_Proyecto2.txt` contains only that public URL. No automated deployment or public-URL check was performed in this environment. |
| T6 | `uv run --with reportlab python docs/generar_reporte_ejecutivo_proyecto2.py` | Passed: created `docs/Reporte_Ejecutivo_Proyecto2.pdf`; source body count is 2,072 words and the AI-use declaration is 65 words. |
| T6 | `uv run --with pypdf python -c '...'` extracted all PDF pages and checked required factual markers. | Passed: 5 pages, 2,232 extracted body words, 17,005 extracted characters, and all required markers present. The extracted count includes headings and title; both it and the source-body count satisfy 2,000-3,000 words. |
| T6 | `pdftoppm -png -r 120 docs/Reporte_Ejecutivo_Proyecto2.pdf /tmp/opencode/proyecto2-reporte`, manual review of all five rendered pages, `pdfinfo`, and `qpdf --check`. | Passed: five 993x1404 rendered pages are visually legible; the ablation table and references render without clipping. PDF is A4, unencrypted, and qpdf reports no syntax or stream-encoding errors. |
| T7 | `pdflatex -interaction=nonstopmode -halt-on-error -output-directory docs docs/Reporte_Ejecutivo_Proyecto2.tex` (two passes), followed by log inspection. | Passed: produced the five-page LaTeX PDF; the final log contains no LaTeX errors, box warnings, or warnings. |
| T7 | `qpdf --check docs/Reporte_Ejecutivo_Proyecto2.pdf`; `pdftotext -layout`; extraction assertions; and `pdftoppm -png -r 120` review of all five pages. | Passed: qpdf reported no syntax or stream-encoding errors; extraction has 2,223 body words and 2,461 total words, preserves accented text and all required factual markers, and contains no `Resumen ejecutivo`. Rendered pages show a legible table, references, and AI-use declaration without clipping. The ReportLab generator was removed, leaving the `.tex` file as the canonical source. |
| T8 | Two-pass `pdflatex -interaction=nonstopmode -halt-on-error -output-directory docs docs/Reporte_Ejecutivo_Proyecto2.tex`; `qpdf --check`; `pdftotext -layout`; and a `pypdf` assertion over PDF link annotations. | Passed: regenerated a five-page PDF. The source and PDF contain the labeled repository and MVP links; PDF annotations resolve to `https://github.com/Branuvg/labs_deep` and `https://labsdeep-proyecto2.streamlit.app/`. Text extraction includes `Enlaces del proyecto` and excludes `Resumen ejecutivo`. |

## Work-unit evidence

| Task | Commit | Focused check | Runtime check | Rollback boundary |
| --- | --- | --- | --- | --- |
| T1 | `ea9aaa3` | Markdown structure reviewed. | N/A — planning document only. | Remove `odd/tasks/proyecto2-entrega-final.md`. |
| T2 | `bec0330` | `uv run --with nbformat …` passed; `git diff --check` passed. | Full notebook run is pending authorized Colab T4 access. | Revert the raw-path simplification and Stage B epoch setting in `notebooks/proyecto2.ipynb`. |
| T4 | `3839496` | Streamlit `AppTest` passed across all four filters; artifact parse check passed. | Same `AppTest` renders the application boundary without exceptions. | Revert this verification evidence only; no application source changed. |
| T6 | `docs(proyecto2): add executive project report` | ReportLab generation, pypdf extraction and word count, rendered-page visual review, `pdfinfo`, and `qpdf --check` all passed. | N/A - this deliverable is a static PDF, validated through generation, extraction, rendering, and PDF integrity checks. | Revert `docs/Reporte_Ejecutivo_Proyecto2.pdf`, `docs/generar_reporte_ejecutivo_proyecto2.py`, and this task evidence only. |
| T7 | `docs(proyecto2): convert executive report to LaTeX` | Two-pass `pdflatex`, log inspection, `qpdf --check`, extraction/word-count assertions, phrase exclusion, and all-page render review passed. | N/A - this deliverable is a static PDF, validated through compilation, extraction, rendering, and PDF integrity checks. | Revert `docs/Reporte_Ejecutivo_Proyecto2.tex`, regenerated PDF, obsolete generator removal, and this task evidence only. |
| T5 | `docs(proyecto2): record public MVP URL` | Exact URL-only content check passed. | N/A — public reachability was confirmed by the user in an incognito window; no automated external check was performed here. | Remove `docs/Enlace_MVP_Proyecto2.txt` and revert the T5 task evidence only. |
| T8 | `docs(proyecto2): link public project resources` | Two-pass LaTeX compilation, qpdf integrity check, text extraction, link-annotation assertions, and phrase exclusion all passed. | N/A — this deliverable is a static PDF, validated through compilation, extraction, annotations, and PDF integrity checks. | Revert the new report section, regenerated PDF, LaTeX auxiliary files if tracked, and T8 task evidence only. |
