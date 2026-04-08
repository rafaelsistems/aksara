# Benchmark Report: Large Corpus Robustness Evaluation

**Project:** AKSARA  
**Evaluator:** `tools/corpus_robustness_eval_large.py`  
**Scope:** Completed streaming robustness runs at 5k, 10k, 50k, and 100k samples  
**Status:** Formal report of observed results only

---

## 1. Purpose

This report documents the observed results from the large streaming corpus robustness evaluator for AKSARA.  
It summarizes the completed runs at 5,000, 10,000, 50,000, and 100,000 samples, including aggregate metrics, category-level behavior, and operational notes about progress output.

All values below are taken from completed runs already observed in this session. No synthetic or estimated numbers are introduced here.

---

## 2. Evaluator Behavior

The evaluator is designed to be:

- deterministic in sample ordering
- dependency-free
- streaming-friendly for large sample counts
- capable of periodic progress output via `--progress-every`
- able to reduce per-sample detail automatically when sample counts exceed the quiet threshold

Observed behavior during the large runs:

- the run remained stable across 5k, 10k, 50k, and 100k samples
- progress reporting was available for large runs
- detail output was throttled for larger runs
- all completed runs reported 100% validity

---

## 3. Observed Aggregate Results

### 3.1 Run Summary

| Sample Count | Validity | Average Score | Constraint Avg | KRL Avg | Stability Note |
|---:|---:|---:|---:|---:|---|
| 5,000 | 100% | stable | stable | stable | averages remained consistent |
| 10,000 | 100% | stable | stable | stable | averages remained consistent |
| 50,000 | 100% | stable | stable | stable | averages remained consistent |
| 100,000 | 100% | stable | stable | stable | averages remained consistent |

### 3.2 What Was Observed

From the completed runs:

- validity stayed at 100% for all tested sample sizes
- average scores remained stable across 5k, 10k, 50k, and 100k runs
- constraint averages remained stable across runs
- KRL averages remained stable across runs
- no run showed a collapse in aggregate behavior as sample size increased

> Note: The session confirmed consistency and 100% validity, but did not preserve the exact numeric averages in a reusable form here. This report therefore records the measured stability and validity facts without inventing numbers.

---

## 4. Category-Level Behavior

The evaluator cycles through five categories in a fixed order:

- `active`
- `passive`
- `anaphora`
- `nominal`
- `domain`

Observed category behavior:

| Category | Observed Behavior |
|---|---|
| active | stable outputs across repeated sampling |
| passive | stable outputs across repeated sampling |
| anaphora | stable outputs across repeated sampling |
| nominal | stable outputs across repeated sampling |
| domain | stable outputs across repeated sampling |

Additional observed properties:

- category sampling is deterministic by design
- category outputs were sufficiently stable to support 100k-scale streaming evaluation
- the evaluator records per-category aggregates for score, timing, constraint, KRL, frame, and notes

---

## 5. Progress and Streaming Notes

The large evaluator supports progress logging suitable for long runs.

Observed operational characteristics:

- progress output is periodic on large runs
- per-sample detail is reduced automatically when sample count exceeds the quiet threshold
- the run can be monitored without overwhelming output
- the tool is suitable for 5k, 10k, 50k, and 100k sample sizes in a single streaming pass

This makes the evaluator practical for regression-style robustness checks without requiring a separate batch system or external dependencies.

---

## 6. Category Breakdown Structure

The evaluator reports the following per-category aggregates:

- sample count
- valid rate
- average linguistic score
- average runtime
- average constraint score
- average KRL completeness
- observed frame set
- frame diversity
- notes about anomalies or missing sub-results

This structure is useful for downstream comparison, because it preserves both high-level summary metrics and category-specific behavior.

---

## 7. Interpretation

### Measured facts

- 5k, 10k, 50k, and 100k runs all completed successfully
- all completed runs reported 100% validity
- averages were stable across the tested scale range
- the evaluator maintained streaming operation throughout the large runs

### Analysis

The observed results indicate that the evaluator and underlying framework are stable under increasing sample size, at least up to 100k samples.  
The consistent averages suggest that the test corpus and processing pipeline are not exhibiting scale-related instability in the measured dimensions.

### Recommendation

Continue using this evaluator as the formal large-scale robustness benchmark and keep the run sizes staged in future regression checks. The current evidence supports the following staged practice:

1. preserve 5k and 10k as quick robustness gates
2. use 50k as a medium-scale stability checkpoint
3. use 100k as the high-confidence stress benchmark
4. keep progress logging enabled for large runs so long evaluations remain observable

---

## 8. Recommended Next Steps

1. Add structured JSON export to the evaluator for machine-readable downstream use.
2. Add a lightweight regression smoke check for the evaluator/export path.
3. Keep this report updated only with measured results from completed runs.
4. Re-run the benchmark after evaluator changes to confirm stability remains unchanged.

---

## 9. Conclusion

The completed large corpus robustness runs demonstrate stable behavior across 5k, 10k, 50k, and 100k samples, with 100% validity and consistent aggregate behavior.

The main conclusion is not that the system is merely “passing a test,” but that the evaluator is now capable of supporting larger-scale, repeatable robustness reporting for AKSARA without introducing additional dependencies or changing core behavior.

---

*End of report.*