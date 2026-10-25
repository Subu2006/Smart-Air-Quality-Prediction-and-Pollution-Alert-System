# Requirement-to-deliverable checklist (source review and build record)

| Requirement | Source | Deliverable / evidence | Status at build time | Caveat |
|---|---|---|---|---|
| Six CSE273 units and course outcomes | `ML Syllabus.pdf` | `docs/syllabus_coverage_matrix.csv`; report Appendix A | Mapped | Theoretical concepts are explicitly separated from implemented ones. |
| Ten prescribed practicals | `ML Syllabus.pdf`, pp. 1–2 | Ten `practicals/*.py` scripts and practical guide | Implemented as scripts; run-tested | Practical 03/04/08 use labelled fallback datasets unless sourced files are supplied. |
| Required report sequence and page allocation | `ml report.txt`; faculty photo | Word/PDF report; static contents and headings | Implemented approximately | Faculty outline overlaps Methodology and Results on page 20; resolved with Methodology to p. 20 and Results starting p. 21. |
| Student/institution details | AIRWATCH INDIA reference report, title page | Project title page | Adapted | LPU, CSE, Subrat Panigrahi, 26143073, N1 retained. Old INT374 code and old guide intentionally excluded. |
| Source CSV and provenance | AIRWATCH INDIA reference context + public data pages | README, schema guide, ingestion utility | Partially implemented | Actual India source CSV was not attached and could not be downloaded in the disconnected build environment. |
| Working ML pipeline and saved artifact | User brief | `src/.../models.py`, CLI, `models/demo_model.joblib` | Implemented/tested on synthetic demo | No real-data model can be claimed until the official CSV is supplied and retrained. |
| App navigation/visuals/prediction/alerts | User brief | `app.py` | Code implemented; syntax checked | Browser-level interaction not executed because Streamlit was not installed in the build environment. |
| Optional latest data | User brief | `src/smart_air_quality/openaq.py`, UI action | Implemented conditionally, not live-tested | Requires OpenAQ v3 API key, location ID and internet access. |
| Automated tests | User brief | `tests/test_pipeline.py` | Run-tested | Does not replace manual/browser or live API tests. |
| Real analytical results and figures | User brief | `reports/demo_results_summary.json`, `models/demo_model.csv`, `figures/*.png` | Generated from code | All shipped figures/metrics are labelled synthetic, not field observations. |
| Word report, PDF, visual QA | User brief | `reports/Smart_Air_Quality_Project_Report.docx/.pdf` | To be verified after generation | Screenshots of a live Streamlit session are not supplied; no fabricated screenshot is inserted. |
| Portable package | User brief | README, requirements, source, tests, data/demo, docs, ZIP | Packaged | External dataset download/licensing validation remains user-side because source archive was unavailable. |
