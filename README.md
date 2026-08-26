# AUD phenotyping: analysis code

Analysis code for *Language model phenotyping recovers alcohol use disorder missed by
diagnosis codes across health systems* (submitted).

Start with [`provenance.md`](provenance.md): it maps each result in the paper to the
script that produced it. [`SKILL.md`](SKILL.md) is the same map written for an agent.

## Scope

The study data are protected health information and are not in this repository. What
is here is every analysis step from the chart ratings and the model output panel
onward: the reference standard, the estimators, the figure and table producers, and
the checks that hold the paper to them.

Not here: the language model and how it was run, and the chart-review application. The
phenotyping instructions given to the model are printed verbatim in Supplementary
Note 1 of the paper.

## Layout

| path | |
|---|---|
| `downstream/npj/rebuild_all.sh` | the pipeline, in dependency order |
| `downstream/npj/ref_standard.py` | the reference standard, derived in one place |
| `downstream/stats_core.py` | the estimators, defined in one place |
| `downstream/data.yaml` | the pre-specified analysis rule |
| `downstream/npj/` | figure and table producers |
| `downstream/` | comparator arm, registry model, subgroup screen |
| `paper/manuscript_numbers.csv` | the paper's counts and proportions, recomputed |

## Requirements

Python 3.12, pandas 2.2, numpy 2.2, scipy 1.14, matplotlib 3.9. The comparator arm
also needs scikit-learn, xgboost, torch and transformers; the registry model needs
arviz.

Without the data, one thing runs:

```sh
cd downstream && python3 -c "import stats_core; stats_core._selftest()"
```

`ruff check` is clean; see `pyproject.toml`, which records why the import order and the
formatting are left alone.

## Licence

Code is released under the MIT Licence ([`LICENSE`](LICENSE)).

Documentation — `README.md`, `provenance.md` and `SKILL.md` — is released under
CC BY 4.0 ([`LICENSE-docs`](LICENSE-docs)).

`downstream/npj/nature_style.py` is a third-party helper, retained unmodified; its
origin is recorded in its header.

## Manuscript

[Wang-Haining/aud-llm-manuscript](https://github.com/Wang-Haining/aud-llm-manuscript)
