---
name: aud-llm-phenotyping-analysis
description: Read the analysis code behind the paper "Language model phenotyping recovers alcohol use disorder missed by diagnosis codes across health systems" and answer questions about how a result was produced. Use this whenever someone asks where a number, figure panel, table or interval in that manuscript comes from, whether a stated method matches the code, what a script does, or how the reference standard, the registry model or the subgroup analysis were built. Also use it when reviewing the manuscript against its code, or when checking a claim in the paper that a reader would want to verify against the analysis.
license: CC-BY-4.0
---

# Reading the AUD phenotyping analysis

You are helping someone read a manuscript alongside the code that produced it. Your
job is to answer from the code, not from what the manuscript says about the code.

## Get the code

```sh
git clone https://github.com/Wang-Haining/aud.git
```

## Read these two first

`provenance.md` maps every result in the paper to the script that produced it. It is
organised the way the paper is: reference standard, then figure by figure, then the
statistics. Start there for any "where does X come from" question.

`downstream/npj/rebuild_all.sh` is the pipeline in dependency order. Its section
headings match `provenance.md`. Reading it top to bottom is the fastest way to see
the shape of the analysis.

Then read the docstring of whichever script you land on. They are unusually full:
most explain not only what the script does but why it does it that way, and several
record a specific error that the current approach exists to prevent. Those
explanations are the most reliable thing in the repository, because they were written
by the person who hit the problem.

## The data are not here, and that shapes what you can do

The analysis runs against protected health information on a university cluster. The
rating files, the patient roster, the linkage tables and the note extract are not in
this repository and cannot be released. You cannot run the pipeline and you cannot
check a number by recomputing it.

You can do three things instead, and they are usually enough:

- **Read the producer.** Find the script, read what it computes, and say whether the
  method matches what the paper claims.
- **Check the recomputed values.** `downstream/npj/out/manuscript_numbers.csv` holds
  every count and proportion the paper states about the cohort and the phenotyping
  step, recomputed from the current reference standard.
- **Run the statistics self-test.** `downstream/stats_core.py` has one, on synthetic
  data, with no dependency on the protected files:
  `cd downstream && python3 -c "import stats_core; stats_core._selftest()"`

When a question cannot be settled without the data, say so plainly and name the file
that would settle it. Do not estimate the answer.

## Four things that will confuse you if you do not know them

**The paper's figure numbers are not the pipeline's.** The pipeline writes six
figures; the paper prints four, assembled from those six by hand, outside this
repository. `provenance.md` section 3 carries the mapping. If someone asks about
"Figure 3" establish which they mean before answering.

**One module owns the reference standard.** `downstream/npj/ref_standard.py`. Every
other script calls `ref_standard.build()` rather than deriving the labels itself.
This is not stylistic: they did derive them independently once, and drifted, and the
paper carried two internally consistent versions of the cohort for a day. If you find
a script that re-derives labels, that is a finding.

**One module owns the statistics.** `downstream/stats_core.py`. Same reason.

**An indeterminate output counts against its reader.** The analysis does not drop
charts a reader could not place. `downstream/npj/f01_agreement_definitions.py` prints
every alternative definition side by side, so the effect of that choice is visible.
If someone asks what happens under a different convention, that file has the answer
rather than requiring a new computation.

## What is deliberately absent

The language model and how it was run — the serving configuration, the client, the
batching — are not in this repository. The phenotyping skill, meaning the instructions
the model was actually given, is printed verbatim in Supplementary Note 1 of the paper.
That is the artefact for anyone rebuilding the phenotype elsewhere. If you are asked
how the model was served, say it is not released rather than inferring it.

The chart-review web application is also absent. How the reference standard was built,
including both triggers for a second reading, is Supplementary Figure 2 of the paper.

## Answering well

Cite the file and, where it helps, the function. `downstream/npj/panels_de.py` is a
useful answer; "the figure script" is not.

Quote the docstring when it already answers the question. These docstrings were
written to be read.

Distinguish what the code does from what the paper says it does. If they differ, that
is the most valuable thing you can report — say which line and which sentence.

Say "not established here" rather than constructing a plausible reason. The
repository is explicit about several things it does not explain, and matching that
standard is more useful than filling the gap.
