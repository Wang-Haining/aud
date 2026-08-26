#!/usr/bin/env bash
# Rebuild every figure the manuscript includes, from the rating files up, and report any
# product whose bytes changed. An archive that removed something still needed shows up here
# as an import error or a changed file, not as a surprise at submission.
#
# Replaces rebuild_findings1.sh, which covered Findings 1 only.
# Run on Quartz from /N/project/depot/hw56/aud/medace_aud.
#
# Three producers are excluded because they cannot finish on a login node. Submit each
# with sbatch, then rerun this script:
#   jobs_bayes_all.sh        refits the registry measurement model, seven jobs, six-hour walltime
#   note_ml_baseline.py      exceeds the twenty-minute login-node limit
#   note_bert_baseline.py    fine-tunes a clinical encoder; needs a GPU
# All three write inputs this script reads. Until 26 August this block named only two of
# them and the sequence below called a note baseline by a filename that no longer existed,
# so that step failed on every run and the script never exited 0.
set -u
cd /N/project/depot/hw56/aud/medace_aud
module load python/3.12.4 2>/dev/null

# matplotlib writes the wall-clock time into /CreationDate in every PDF, so two runs of the
# same producer on the same data differed in bytes and the check at the end of this script
# reported every figure as changed. SOURCE_DATE_EPOCH replaces that timestamp with a fixed
# instant. The value is arbitrary and carries no meaning beyond making the check work.
export SOURCE_DATE_EPOCH=1786752000

# A backstop, not a fix. Python randomises the hash of a string per process, so any
# producer that takes a drawing decision from hash() or from the iteration order of a set
# gives a different figure on every run. f06_figure6.py did, until 14 August. Pinning the
# seed makes such a producer reproducible without making it correct, so a figure that
# changes when this line is removed still has a defect to find.
export PYTHONHASHSEED=0

PRODUCTS="
downstream/npj/out/figure1/figure1.pdf
downstream/npj/out/figure1/supp_confusion_six.pdf
downstream/npj/out/figure3/figure3.pdf
downstream/npj/out/figure3/figure3_calendar.pdf
downstream/npj/out/figure3/figure3_specifications.pdf
downstream/npj/out/figure2/figure2.pdf
downstream/npj/out/figure2/figure2_era.pdf
downstream/npj/out/figure4/figure4.pdf
downstream/npj/out/figure5/figure5.pdf
downstream/npj/out/figure5/figure5_alluvial.pdf
downstream/npj/out/figure6/figure6.pdf
downstream/npj/out/f01_flow_table1/flow.pdf
downstream/npj/out/f01_flow_table1/table1.tex
downstream/npj/out/supp_table6.tex
downstream/npj/out/figure4/panel_supp_reader_swarm.pdf
downstream/npj/out/figure4/supp_documentation.pdf
downstream/npj_goal6_outputs/figures/FigS2_documentation_mechanism.pdf
"

BEFORE=$(mktemp)
for f in $PRODUCTS; do [ -f "$f" ] && md5sum "$f" >> "$BEFORE"; done

fail=0
run () {
  printf '%-56s' "$1"
  if out=$(python3 $1 2>&1); then
    echo "ok"
  else
    echo "FAILED"
    echo "$out" | tail -12 | sed 's/^/        /'
    fail=1
  fi
}

echo "=========================== the reference standard, and the cohort flow"
run "downstream/npj/ref_standard.py"
run "downstream/npj/f00_cohort_flow_and_table1.py"
run "downstream/npj/f01_accuracy_against_reference.py"
run "downstream/npj/f01_supp_temporal.py"
run "downstream/npj/f01_restricted.py"
run "downstream/npj/f01_agreement_definitions.py"
run "downstream/npj/f01_kappa_inventory.py"
run "downstream/build_rater_table.py"
run "downstream/npj/supp_table8.py"

echo "=========================== Figure 1, agreement with the reference standard"
run "downstream/structured_baselines.py"
run "downstream/npj/build_supp_table6.py"
run "downstream/npj/panels_de.py"
run "downstream/npj/panel_a.py"
run "downstream/npj/panel_g.py"
run "downstream/npj/f01_figure1.py"
run "downstream/npj/supp_confusion_six.py"

echo "=========================== Figure 2, external validation"
run "downstream/site_figure_data.py"
run "downstream/site_external_validation.py"
run "downstream/npj/f02_figure2.py"
run "downstream/npj/f02_supp_era.py"
run "downstream/purity_and_weighting.py"

echo "=========================== Figure 3, registry scale"
run "downstream/npj/f03_registry_burden.py"
run "downstream/npj/f03_figure3.py"
run "downstream/npj/f03_supplement.py"
run "downstream/npj/f03_extract_window.py"

echo "=========================== Figure 4, model robustness"
run "downstream/analyze_goal5_robustness.py"
run "downstream/npj/f04_figure4.py"
run "downstream/npj/f04_model_disparity.py"
run "downstream/npj/f04_protocol_disparity.py"
run "downstream/npj/panel_supp_reader_swarm.py"
run "downstream/npj/f04_supp_documentation.py"
run "downstream/npj/f04_supp_tables.py"

echo "=========================== Figure 5, evidence grounding"
run "downstream/npj/f05_figure5.py"
run "downstream/npj/f05_supp_alluvial.py"

echo "=========================== Figure 6, subgroup parity"
run "downstream/run_parity.py"
run "downstream/npj/f06_figure6.py"
run "downstream/render_fig5_equity.py"

echo "=========================== the interval registry the manuscript is held to"
run "downstream/npj/publish_intervals.py"

echo "=========================== does every product describe one cohort?"
# Last, and it decides the exit status. A product older than the rating files it derives
# from is wrong whatever it says, and until 14 August nothing compared the products to
# each other: Table 1 and the STARD flow sat on the labels from before the standard
# closed while Figure 1 and Supplementary Table 8 sat on the closed ones, each set
# internally consistent.
if python3 downstream/npj/consistency_gate.py; then
  :
else
  fail=1
fi

echo "=========================== did any product change?"
AFTER=$(mktemp)
while read -r h f; do md5sum "$f" >> "$AFTER"; done < "$BEFORE"
if diff -q "$BEFORE" "$AFTER" >/dev/null; then
  echo "every product rebuilt byte-identical"
else
  echo "changed:"
  diff "$BEFORE" "$AFTER" | sed 's/^/    /'
fi

echo
echo "exit status: $fail"
exit $fail
