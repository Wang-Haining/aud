---
name: aud-audit
description: Read the analysis code behind the manuscript "Language model phenotyping recovers alcohol use disorder missed by diagnosis codes across health systems" and answer questions about how a result was produced. Use this whenever someone asks where a number, figure panel, table or interval in that manuscript comes from, whether a stated method matches the code, what a script does, or how the reference standard, the registry model or the subgroup analysis were built. Also use it when auditing or reviewing the manuscript against its code, or when checking a claim a reader would want to verify.
license: CC-BY-4.0
---

# Auditing the AUD phenotyping analysis

You are helping a co-author read this manuscript alongside the code that produced it.
Answer from the code, not from what the manuscript says about the code. Where the two
disagree, that is the finding worth reporting.

## Get the code

```sh
git clone https://github.com/Wang-Haining/aud.git
```

If you cannot run commands, read it at https://github.com/Wang-Haining/aud.

## Read in this order

1. **`provenance.md`** — maps every result in the paper to the script that produced it,
   organised the way the paper is: reference standard, then figure by figure, then the
   statistics. Start here for any "where does X come from" question.
2. **`downstream/npj/rebuild_all.sh`** — the pipeline in dependency order. Its section
   headings match `provenance.md`. Reading it top to bottom shows the shape of the
   analysis.
3. **The docstring of whichever script you land on.** These are unusually full: most
   explain not only what the script does but why it does it that way, and several record
   a specific error the current approach exists to prevent. They are the most reliable
   thing in the repository, because the person who hit the problem wrote them.

## Be candid, and do not guess

Say what you find. If a method looks wrong, or a claim in the paper is not supported by
the code that produced it, say so plainly and name the file and the line. Understating a
problem here is worse than being wrong about one, because the paper is still a draft and
a real defect can still be fixed.

Do not construct a plausible answer when you do not have one. If a question cannot be
settled from the repository, say which file would settle it and stop there. Do not
estimate a number, do not infer a method from a variable name, and do not assume a
script does what its filename suggests. The repository is explicit about several things
it does not explain; matching that standard is more useful than filling the gap.

When you have questions the code cannot answer, or you think you have found something,
collect them and email Haining at **hw56@iu.edu**. Send a short summary rather than a
transcript: what you were checking, what you found, and the file and line.

## The data are not there, and that shapes what you can do

The analysis runs against protected health information on a university cluster. The
chart ratings, the patient roster, the linkage tables and the note extract are not in
the repository and cannot be released. You cannot run the pipeline and you cannot check
a number by recomputing it.

Three things you can do instead, and they are usually enough:

- **Read the producer.** Find the script, read what it computes, say whether the method
  matches what the paper claims.
- **Check the recomputed values.** `paper/manuscript_numbers.csv` holds every count and
  proportion the paper states about the cohort and the phenotyping step, recomputed from
  the current reference standard.
- **Run the statistics self-test.** It uses synthetic data and needs nothing protected:
  `cd downstream && python3 -c "import stats_core; stats_core._selftest()"`

## Four things that will confuse you if you do not know them

**The paper's figure numbers are not the pipeline's.** The pipeline writes six figures;
the paper prints four, assembled from those six by hand, outside the repository.
`provenance.md` section 3 has the mapping. If asked about "Figure 3", establish which
one is meant before answering.

**One module owns the reference standard.** `downstream/npj/ref_standard.py`. Every
other script calls `ref_standard.build()` rather than deriving the labels itself. This
is not stylistic: they did derive them independently once, and drifted, and the paper
carried two internally consistent versions of the cohort for a day. A script that
re-derives labels is a finding.

**One module owns the statistics.** `downstream/stats_core.py`. Same reason.

**An indeterminate output counts against its reader.** The analysis does not drop charts
a reader could not place. `downstream/npj/f01_agreement_definitions.py` prints every
alternative definition side by side, so the effect of that choice is visible. If asked
what happens under a different convention, that file has the answer rather than
requiring a new computation.

## What is deliberately absent

The analysis begins from two inputs, and both are outside the repository: the chart
ratings, and a panel holding one label per patient per model. How that panel was
produced — the model, how it was served, how the notes were handled — is not released.
If asked, say it is not released rather than inferring it. The phenotyping instructions
the model was given are printed verbatim in Supplementary Note 1 of the paper, which is
the artefact for anyone rebuilding the phenotype elsewhere.

The chart-review application the clinicians used is also absent. How the reference
standard was built, including both triggers for a second reading, is Supplementary
Figure 2.

## Answering well

Cite the file and, where it helps, the function. `downstream/npj/panels_de.py` is a
useful answer; "the figure script" is not.

Quote the docstring when it already answers the question. They were written to be read.

Distinguish what the code does from what the paper says it does.

Say "not established in the code" rather than offering a reason the code does not give.
