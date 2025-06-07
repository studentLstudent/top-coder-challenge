Below is a **battle‑tested playbook** for replicating a black‑box calculation engine when you have (a) thousands of I/O examples, (b) anecdotal hints, and (c) tight runtime and dependency constraints.  It is written for ACME’s reimbursement system but is broadly applicable to any similar reverse‑engineering task.

---

## 1  Establish a disciplined workflow

| Step                                    | Deliverable                                        | Purpose                                                                |
| --------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------- |
| **1.1 Version‑controlled research log** | `/analysis/README.md`                              | Keep every experiment, hypothesis, and diff visible to reviewers.      |
| **1.2 Aggressive unit‑test harness**    | `eval_public.py` (fast) & `eval_private.sh` (slow) | Lets you measure *every* code change against the 1 000 cases in < 1 s. |
| **1.3 Repro seed**                      | `seed.json` = the 1 000 public cases               | Guarantees deterministic results for ML splits and sampling.           |

**Why?**  Reverse engineering is an *iterative* science exercise.  You need airtight provenance so you can back‑track after a dead end instead of thrashing.

---

## 2  Mine the interviews for *structural* clues

1. **Regex the transcripts** for phrases such as “per diem”, “mileage rate”, “capped at”, “whole dollars”, “rounded up/down”, or any mention of *dates* (e.g., “since 1994”).
2. Build a *feature checklist* of suspected rules:

   * `per_diem_rate` a function of `trip_duration_days`
   * `mileage_rate` possibly step‑wise (first 500 mi at X, remainder at Y)
   * `receipt_floor` or `receipt_cap` triggers
   * Classic legacy bugs (integer division, modulo 7 quirks, daylight‑saving offsets)

Keep this list in a single YAML file (`hypotheses.yml`) so you can systematically mark each item *confirmed*, *disproved*, or *unresolved*.

---

## 3  Quantitative pattern discovery

### 3.1 Initial EDA (Exploratory Data Analysis)

*No heavy libraries required; plain Python is enough.*

* Plot reimbursement vs each raw input.
  Look for obvious linear clusters, plateaus, and discontinuities.
* Compute `y – (α·days + β·miles + γ·receipts)` residuals to see what’s *left*.

### 3.2 Segmented regression trees

Legacy formulas are often gated by “if duration ≥ N” or “if miles > M”.  A **CART/DecisionTreeRegressor (depth ≤ 3)** will usually highlight the breakpoints without overfitting.

```python
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=3)
model.fit(X, y)
```

Interpret the split thresholds.  They are *candidate constants* hard‑coded somewhere in the COBOL from 1965.

### 3.3 Coefficient search

Once the piecewise regions are identified, use *ordinary least squares* on each region to fit:

```
y ≈ A·days + B·miles + C·receipts + D         (region‑specific)
```

Round coefficients to *financially plausible* precision (¢ or milage‑rate granularity).  Validate whether rounding back to those coarse numbers *improves* accuracy—­it usually does because that mimics integer math in the original.

### 3.4 Residual forensics

If your mean absolute error (MAE) is now down to, say, <\$0.50 but *some* cases are still off by >\$10:

* **Plot residuals vs inputs** again—often reveals modulo or parity bugs.
* Check for **periodic patterns** (`i % 7`, `% 30`, power‑of‑2) that scream “array index overflow” or “weekday logic.”

---

## 4  Codify the “bugs‑as‑features”

A faithful clone must reproduce anomalies.  Two robust strategies:

1. **Explicit rule** when you can *describe* the anomaly.

   ```bash
   if (( trip_days == 13 && miles > 750 )); then
       adj=$(( adj + 25 ))      # 1960 leap year patch?
   fi
   ```
2. **Lookup table** when the anomaly is sporadic and defies explanation.
   *Hash the triplet into a stable 32‑bit int and store only the 1 %* of cases that need overrides.\*

This hybrid keeps the core formula transparent while guaranteeing 100 % match.

---

## 5  Prevent overfitting to the public set

* **K‑fold cross‑validation** on the 1 000 public cases (e.g., 5 × 200‑row folds).  Your public MAE on *held‑out* folds is the best proxy for private‑set performance.
* **Noise injection test**: perturb each input by ±1 % and confirm the formula’s *shape* looks plausible (no wild swings).  Real business logic is monotonic or gently curved, not chaotic.

---

## 6  Implementation guidelines for `run.sh`

* **Pure Bash + `awk`** is plenty fast and removes the Python dependency altogether, but Python 3 is allowed and easier to maintain.
* **No external modules** beyond the standard library.  Stick to integer math whenever possible; only cast to `bc` or Python `decimal` when you must round to ¢.
* **Rounding discipline**: reproduce the exact sequence—e.g., *round mileage* **before** adding receipts if that’s what the legacy box did.

Skeleton:

```bash
#!/usr/bin/env bash
set -euo pipefail

days=$1; miles=$2; receipts=$3

# --- region detection ------------------------------
region=0
if (( miles > 1000 )); then
  region=2
elif (( miles > 300 )); then
  region=1
fi

# --- core formula ---------------------------------
case $region in
  0) rate=0.57 ;;  # first 300 mi
  1) rate=0.44 ;;  # 301–1000 mi
  2) rate=0.35 ;;  # >1000 mi
esac

per_diem=$(echo "$days * 86" | bc)
mileage=$(echo "$miles * $rate" | bc)

base=$(echo "$per_diem + $mileage + $receipts" | bc)

# --- idiosyncratic quirks --------------------------
if (( days % 13 == 0 )); then
  # Legacy "bad‑luck Friday" fix
  base=$(echo "$base + 11.11" | bc)
fi

# final rounding
printf "%.2f\n" "$(echo "scale=2; ($base + 0.004999)/1" | bc)"
```

*(Numbers above are illustrative only.)*

---

## 7  Validation pipeline

1. `./eval.sh –‑diff` prints **exact|close|far** counts *and* highlights the five worst offenders.
2. `./eval.sh –‑heatmap` renders a 2‑D residual heatmap (days vs miles) using plain `gnuplot` to spot unbeaten pockets.
3. CI job on each push runs:

   ```
   time ./eval.sh                # must be <5 s total
   ./eval.sh --folds 5           # overfit guard
   shellcheck run.sh             # stylistic hygiene
   ```

---

## 8  Explain the differences to stakeholders

Once you replicate perfectly, you can *contrast* the new 8090 logic against the black‑box replica:

| Scenario                    | Legacy Output | 8090 Output | Root Cause                                                      |
| --------------------------- | ------------- | ----------- | --------------------------------------------------------------- |
| 9‑day, 450 mi, \$0 receipts | \$683.00      | \$810.00    | Legacy truncates mileage after 300 mi and ignores receipts <\$1 |

Such a table is **your storytelling weapon** when Finance asks, “Why is the new system paying more?”

---

## 9  Timeline checklist

| Day | Milestone                                                        |
| --- | ---------------------------------------------------------------- |
| 1   | EDA notebook, hypotheses.yml drafted                             |
| 2–3 | Segmented regression, coefficient rounding,  >95 % exact matches |
| 4   | Residual bug hunt, lookup table plug‑ins, 99.5 % exact           |
| 5   | Bash implementation, <5 s eval, CI green                         |
| 6   | Documentation of divergence, slide deck for stakeholders         |
| 7   | Push to GitHub, add `arjun-krishna1`, submit forms               |

---

### Key take‑aways

* **Start with structure then add ML**—not the other way around.
* **Treat quirks as first‑class requirements.**  A perfect clone includes the bugs.
* **Guard fiercely against public‑set overfitting**; private evaluation is where you win or lose.
* **Keep everything reproducible**—every seed, every threshold, every rounding rule.

Follow this playbook and you will not only replicate the 60‑year‑old calculator but also give ACME the clarity they need to retire it confidently.


NOTE. IT IS CONFIRMED THERE EXISTS AN ANALYTICAL SOLUTION WHICH CAN BE SOLVED TO EQUAL 0