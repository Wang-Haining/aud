# Provenance

Which script produced which result in the paper.

This file is written for a reader who has the manuscript open and wants to find the
code behind a specific number, panel or table. It is also written to be read by an
agent, so every claim below names a file rather than describing one.

Everything here runs against protected health information on a university cluster.
The data are not in this repository and cannot be released.

The analysis begins from two inputs, and both are outside this repository:

- the **chart ratings**, one file per reading clinician, which the reference standard
  is derived from;
- **`output/canonical_panel.csv`**, one row per patient per model, holding the label
  each model returned.

How that panel was produced — the model, how it was served, how the notes were
batched — is not released. The boundary is deliberate: everything downstream of the
panel treats a model's label as data, and nothing downstream depends on how the label
was obtained.

---

## 1. Start here

```
downstream/npj/rebuild_all.sh
```

One script. It rebuilds every figure and table the paper carries, in dependency
order, and its last step refuses to finish if two products disagree with each other.
Reading it top to bottom is the fastest way to see the shape of the analysis; the
section headings in it match the sections below.

Three steps in it cannot run on a login node and are excluded, with a note at the top
of the file: the registry measurement model, which is seven jobs at six-hour walltime,
and the two note baselines, one of which exceeds the login-node time limit and one of
which needs a GPU.

---

## 2. The reference standard, and why one file owns it

```
downstream/npj/ref_standard.py        the reference standard
downstream/npj/consistency_gate.py    refuses to finish when two products disagree
```

`ref_standard.build()` is the single place the reference standard is derived. Every
other script calls it rather than re-deriving the labels, because they did re-derive
them once and drifted: on 13 August, Table 1 and the participant flow rested on the
labels from before the standard closed while Figure 1 rested on the closed ones, and
each set was internally consistent, so nothing caught it.

`consistency_gate.py` is what caught it and what prevents it now. It checks two
things. First, staleness: a product older than the rating file it derives from is
wrong whatever it says. Second, identity: where two products carry the same quantity
they must agree, and both must agree with `ref_standard.build()`.

It writes `resources/canonical_counts.json`, which carries the cohort counts and the
md5 of every figure the manuscript includes. The manuscript repository's own check
reads that file, so a figure edited on one side and not the other fails there.

The counts it publishes, and which the paper is held to:

| quantity | value |
|---|---|
| roster | 499 |
| analysis cohort | 487 |
| sampled, AUD code present | 243 |
| sampled, no AUD code | 244 |
| reference AUD present | 251 |
| reference AUD absent | 174 |
| reference indeterminate | 62 |
| determinate, scored for accuracy | 425 |
| referred to a second reader | 65 |
| never referred | 422 |
| no AUD code, reference positive | 39 |

---

## 3. Figure by figure

The pipeline writes six figures. The paper prints four. The four are assembled from
the six outside this repository, by hand, which is why no script here writes
`fig1_overall.pdf` and why the paper's figure numbers do not match the pipeline's.
Every panel and every number in the printed figures comes from the producers below.

| paper | composed from | subject |
|---|---|---|
| Figure 1 | pipeline figure 1 | the LLM against the reference standard, and every comparator |
| Figure 2 | pipeline figures 2 and 3 | under-coding across health systems, and at registry scale |
| Figure 3 | pipeline figures 4 and 5 | model choice, prompt wording, and whether a quotation is in the chart |
| Figure 4 | pipeline figure 6 | how the code's accuracy varies across patient groups |

### Figure 1 — agreement with the reference standard

```
downstream/npj/f01_accuracy_against_reference.py   the four rates, the six-category matrix
downstream/npj/panel_a.py                          panel a, one worked patient and the scale
downstream/npj/panels_de.py                        panels d and e
downstream/npj/panel_g.py                          panel g, every comparator on one axis
downstream/npj/build_supp_table6.py                the comparator table panel g draws from
downstream/npj/f01_figure1.py                      assembles the panels into figure1.pdf
downstream/render_fig2_headline.py                 the confusion matrix and alluvial panels
```

The comparator arm, the readers Figure 1 puts the LLM against:

```
downstream/structured_baselines.py     could structured data alone have found these patients
downstream/note_bert_baseline.py       a fine-tuned clinical encoder on the same notes
```

Both note baselines are excluded from `rebuild_all.sh` and submitted separately:
`note_ml_baseline.py` exceeds the login-node time limit and `note_bert_baseline.py`
needs a GPU. Their folds are split on a list of patients, not of note windows, so no
patient appears in both training and test.

Until 26 August the exclusion block at the top of `rebuild_all.sh` named only one of
them, and the sequence called a note baseline by a filename that no longer existed. That
step failed on every run, so the script never exited 0.

Supporting output:

```
downstream/npj/f01_agreement_definitions.py   every agreement definition the study supports
downstream/npj/f01_kappa_inventory.py         every agreement value the paper reports
downstream/npj/f01_restricted.py              the verification-independent standard
downstream/npj/f01_supp_temporal.py           where in time the evidence sits
downstream/npj/supp_confusion_six.py          the six-category matrix on its own
downstream/npj/supp_table8.py                 the hepatology fellow's independent review
```

### Figure 2 — across health systems, and at registry scale

```
downstream/site_figure_data.py            assembles what the external-validation panels need
downstream/site_external_validation.py    the three contributing hospital systems
downstream/npj/f02_figure2.py             assembles the site panels
downstream/npj/f02_supp_era.py            transportability across calendar era
downstream/purity_and_weighting.py        site purity, and the design weights
downstream/npj/f03_figure3.py             the registry panels
downstream/npj/f03_registry_burden.py     how many registry patients carry AUD, by what identifies them
downstream/npj/f03_supplement.py          the two panels the registry figure gave up
downstream/npj/f03_extract_window.py      the 2015 floor on the index date, and what rests on it
```

The registry measurement model, which supplies the posterior the registry panels
draw. Six-hour walltime, seven jobs, excluded from `rebuild_all.sh`:

```
downstream/fig6_bayes_me_v2.py      the Bayesian misclassification model
downstream/fig6_bayes_strata.py     the same, over the three code strata
downstream/render_fig6_strata.py    renders the stratum panels
```

The information runs one way: the chart labels enter the measurement model and the
measurement model enters the outcome model, and the treatment data never revise who
has AUD. Extended Data Figure 1 in the paper sets this out.

The treatment contrast on a time axis, Figure 2 panels d to f. Same measurement model and
cut; the outcome is two cause-specific piecewise-exponential hazards, first treatment and
death, among new users, censored at the end of the extract. Four chains of 1,200 outer
iterations, one Slurm task each, excluded from `rebuild_all.sh` for the same reason as the
registry model:

```
downstream/fig2_time_to_rx.py       the time-to-event registry model, and its summary
downstream/npj/jobs_fig2_time.sh    one chain per task, both censoring rules
downstream/npj/f02_time_figure.py   a proof of the three time panels
```

The printed panels are drawn by producers kept with the manuscript, from the aggregate
table `fig2_time_to_rx.py --summarise 1` writes.

### Figure 3 — model choice, prompt wording, and grounding

```
downstream/analyze_goal5_robustness.py    the prompt and temperature conditions
downstream/npj/f04_figure4.py             assembles the robustness panels
downstream/npj/panel_reader_swarm.py      every stratum as a point on one axis
downstream/run_npj_provenance.py          quote grounding: is each quotation in the chart
downstream/grounding_finish.py            the unresolved quotations, and the abstentions
downstream/npj/f05_figure5.py             assembles the grounding panels
downstream/npj/f05_supp_alluvial.py       which trigger carried each evidence category
```

### Figure 4 — how accuracy varies across patient groups

```
downstream/run_parity.py                       the subgroup screen, precision-gated
downstream/npj/f06_subgroup_frame.py           what each subgroup can answer, before any analysis
downstream/npj/f06_figure6.py                  assembles the subgroup panels
downstream/render_fig5_equity.py               the equity panels
downstream/npj/panel_supp_reader_swarm.py      the specificity half, Supplementary Figure 5
downstream/npj/f04_supp_documentation.py       Supplementary Figure 3
downstream/npj/f04_supp_tables.py              the LaTeX bodies of the subgroup tables
```

The registry-scale panel of Figure 4, the code's sensitivity by subgroup against the rest of
its axis, and Supplementary Table 14:

```
downstream/icd_fairness.py                     theta by subgroup and stratum, and its bootstrap
downstream/cci_fairness.py                     the same on the Charlson axis
downstream/npj/f04_sdi.py                      the neighbourhood deprivation axis
downstream/npj/f04_registry_disparity.py       each stratum's ratio to the rest, with intervals
downstream/npj/f04_registry_burden_subgroup.py how many patients with AUD each stratum leaves uncoded
downstream/npj/panel_registry_disparity.py     draws the panel
```

Until 24 September 2026 these took theta as the plain mean of the validated charts in a
cell. The charts were drawn stratified on whether an AUD code ever appears, and the
coded-later patients were oversampled, so that overstated theta: the subgroup analysis
implied 4,919 uncoded patients with AUD where the registry model gives 3,261.
`icd_fairness.theta_frame` now takes theta in the two strata without an AUD code at the
cirrhosis diagnosis from never-coded charts, as the registry model does, and
`resample_design` resamples within code stratum and sampling status.

`f06_subgroup_frame.py` runs before the analysis, not after, and decides which
subgroups the sample can answer at all. A subgroup it rules out is reported as
inconclusive rather than dropped.

### Table 1, and the participant flow

```
downstream/npj/f00_cohort_flow_and_table1.py    the flow, and it invokes the next one
downstream/build_table1.py                      Table 1 itself
downstream/build_rater_table.py                 every rating by every rater, in one long table
```

---

## 4. The statistics

```
downstream/stats_core.py    the estimators
downstream/data.yaml        the pre-specified analysis rule
```

One module owns the estimators, so two scripts cannot disagree about what a kappa is.
`data.yaml` carries the rest of the rule: the six-point scale and how it collapses to
three, the eleven models, the subgroups the screen is allowed to examine, and the
parity thresholds. It is read rather than hardcoded so that a reader can see the rule
without reading the scripts that apply it. It carries a self-test that runs on synthetic data with no dependency on the
protected data:

```sh
cd downstream && python3 -c "import stats_core; stats_core._selftest()"
```

What is in it: Cohen's and linear- and quadratic-weighted kappa, Gwet's AC1, Wilson
intervals, percentile and paired bootstrap, McNemar (exact below 25 discordant pairs,
continuity-corrected above; `mcnemar_counts` for a caller holding only the 2x2 table),
DeLong, Benjamini–Hochberg, and the subgroup parity table.

Two conventions worth knowing before reading any number:

- **An indeterminate output counts against its reader.** The paper does not drop the
  charts a reader could not place; the alternative was considered and rejected, and
  `f01_agreement_definitions.py` prints every definition side by side so the effect
  of that choice is visible rather than argued.
- **Bootstrap resamples are 2,000** everywhere except
  `f01_agreement_definitions.py`, which uses 4,000 and says so in its own table.
  This was not always true: eighteen intervals in an earlier draft came from a
  4,000-resample script while their captions said 2,000, and only the endpoints
  differed, in the third decimal.

`downstream/npj/publish_intervals.py` writes every interval the pipeline publishes to
`resources/published_intervals.json`, which the manuscript-side check reads. An
interval printed in the paper that is not in that file fails there.

---

## 5. Reproducibility

`rebuild_all.sh` sets two environment variables and both are load-bearing.

`SOURCE_DATE_EPOCH` fixes the timestamp matplotlib writes into every PDF. Without it
two runs on the same data differ in bytes and the byte-comparison at the end of the
script reports every figure as changed.

`PYTHONHASHSEED` is a backstop and the comment in the script says so. Python
randomises string hashes per process, so a producer that takes a drawing decision
from `hash()` or from the iteration order of a set gives a different figure every
run. One did, until 14 August. Pinning the seed makes such a producer reproducible
without making it correct: a figure that changes when the line is removed still has a
defect to find.

Run `rebuild_all.sh` twice and read the last section. It should report that every
product rebuilt byte-identical.

---

## 6. The numbers in the paper

```
paper/manuscript_numbers.csv
```

Every count and proportion the paper states about the cohort and the phenotyping
step, recomputed from the current reference standard.

The producer, `downstream/npj/manuscript_numbers.py`, prints two columns: the value
the manuscript stated at the time it ran, and the recomputed value. The file here
carries only the recomputed column. The other one is a record of a draft written
against the 499-patient roster and against a reference standard that closed later; its
first row reads 496 where the paper reads 487, and a reader who met that column without
this paragraph would take it for what the paper claims. It sits under `paper/` rather
than in the producer's output directory so that rerunning the producer cannot overwrite
it with the two-column version.

---

## 7. What is not here

**The language model, and how it was run.** No part of the extraction is in this
repository: not the serving configuration, not the client, not the batching. The
phenotyping skill itself — the instructions the model was given — is printed verbatim
in Supplementary Note 1 of the paper and released alongside it, which is the artefact
a reader needs to rebuild the phenotype elsewhere.

**The chart-review application.** The web application the hepatologists used carries
authentication and administrative logic and produces no number in the paper. How the
reference standard was built, including both triggers for a second reading, is
Supplementary Figure 2.

**The data.** Rating files, the patient roster, the linkage tables, the note extract.
The paths appear in the code, because removing them would make the code unreadable;
the files do not and cannot.

**The exploratory work.** The analysis directory on the cluster holds 286 tracked
scripts across 53,401 lines, most of it superseded. What is here is `rebuild_all.sh`
and what it needs: the modules that produce what the paper prints, and nothing that was
tried and abandoned. Two scripts that turned out to be superseded by others in the set
were removed after the first pass rather than shipped alongside their replacements.
