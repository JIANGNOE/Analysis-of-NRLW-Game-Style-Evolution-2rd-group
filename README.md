# DATA3001 NRLW Modelling — Group 1  
**Title:** Predicting Far Sets in the NRLW from Pre-Set Context  
**Date:** 3 November 2025  
**Dataset:** `data.csv` (duplicate of `product/sets.csv`) — built from **Data Group 1’s product** (our renamed Week-5 set-level table)

---

## Table of contents
- [Executive summary](#executive-summary)
- [Background](#background)
- [Plan](#plan)
- [Findings](#findings)
- [Recommendations](#recommendations)
- [Appendix A — Reproducibility steps](#appendix-a--reproducibility-steps)
- [Appendix B — Support and contributors](#appendix-c--support-and-contributors)
- [Appendix C — References](#appendix-d--references)

---

## Executive summary

**Question.** Given what we know at the start of a set (season, team, half, starting zone), what is the chance this possession will be a **far set**?

**What we deliver.** A leakage-safe modelling workflow that estimates  
\[
P(\text{FarSet}=1 \mid \text{Seasonid}, \text{Teamname}, \text{halfNumber}, \text{setZone or HalfTag})
\]
using a **portfolio of four models** with a **time-aware** train-test design.

**Why it matters.** Coaches get a context-adjusted benchmark. You can tell whether a team over- or under-performed relative to where sets started, and track change across seasons.

**Headlines (to be replaced with your run).**
- Field position is the strongest driver.  
- Later seasons show modest improvement after controlling for position.  
- Teams differ even after controls.  
- Time-aware evaluation gives lower but more realistic scores than random splits.

---

## Background

### Objective
Prove that **Data Group 1’s product** (our renamed Week-5 set-level table, one row per set of six) supports useful modelling by answering a clear tactical question: **how likely is a set to go far given what is known at the start?**

### Why this is important
- The NRLW has expanded since 2018. Raw totals are not comparable across seasons.  
- Coaches need **context-adjusted** indicators, not just counts.  
- A **probability at set start** is simple, explainable and actionable.

### Related work
Public work in women’s rugby league mostly covers physical demands, injury and concussion. Possession-level tactical modelling is rare. Our data product enables that by normalising direction, tagging start zones and aggregating events to the set level, so the modelling here fills a practical gap.

---

## Plan

### Data and target

- **Source file:** `data.csv` (from **Data Group 1’s product**)  
- **Unit of analysis:** one team possession (set)  
- **Key:** `(gameid, Teamname, Seasonid, halfNumber, setcount)` (unique)  
- **Predictors available pre-set:** `Seasonid`, `Teamname`, `halfNumber`, `setZone` or `HalfTag`  
- **Outcome source:** `maxAdvance_set` (metres)

**Target definition (FarSet).**
1. Compute a fixed benchmark from 2018, for example the 80th percentile of `maxAdvance_set` in own-half sets.  
2. Define `far_set = 1` if `maxAdvance_set` meets or exceeds that benchmark, else `0`.  
3. Keep the benchmark fixed for all later seasons so eras are comparable.

**Assumptions.**
- Use only pre-set information as predictors.  
- Expect concept drift due to expansion and rule changes.  
- Prefer season-wise splits over random splits.

---

### Modelling approach

We use four complementary models. Each answers the research question in a slightly different way. Together they give performance, interpretability and robustness.

1) **Logistic regression — Zone-only baseline**  
   - **What it is:** A simple classifier that uses only starting field position (HalfTag or setZone).  
   - **Why include it:** Establishes how far field position alone gets us. Sets a clear baseline that is easy to explain.  
   - **How we use it:** Train on early seasons, test on later seasons. Report AUROC, accuracy and precision/recall for far sets. Compare all other models to this baseline.

2) **Regularised logistic regression — Full pre-set features**  
   - **What it is:** Logistic regression with L2 regularisation using `Seasonid`, `Teamname`, `halfNumber`, and zone.  
   - **Why include it:** Adds season and team effects while remaining interpretable. Regularisation controls overfitting as the league expands.  
   - **How we use it:** Train with cross-validation on the training seasons, then evaluate on the held-out seasons. Report coefficients or odds ratios to show which contexts increase the chance of a far set.

3) **Random forest — Non-linear benchmark**  
   - **What it is:** An ensemble of decision trees that captures interactions (for example certain teams exploiting specific zones in certain seasons).  
   - **Why include it:** Tests whether non-linear interactions improve ranking quality over logistic models.  
   - **How we use it:** Train on early seasons, evaluate on later seasons. Report AUROC and feature importance. Useful as a robustness check.

4) **Gradient boosting — HistGradientBoostingClassifier with probability calibration**  
   - **What it is:** A strong tabular learner that often boosts AUROC. We pair it with simple probability calibration if needed.  
   - **Why include it:** Often the best out-of-the-box discrimination for tabular data while still fairly compact.  
   - **How we use it:** Train on early seasons with minimal tuning, then evaluate on the held-out seasons. If probabilities are mis-calibrated, apply Platt or isotonic calibration on training folds only and re-evaluate on the held-out test.

All models use the same leakage-safe inputs and the same time-aware evaluation so comparisons are fair.

---

### Evaluation 

- **Train seasons:** 2018 to 2023  
- **Test seasons:** 2024 to 2025  
- **Model selection:** cross-validation on training seasons using groups by season to respect time.  
- **Primary metrics:** AUROC and AUPRC for ranking; accuracy and precision/recall for intuition; Brier score and reliability plots for probability quality.  
- **Baselines:** majority class and the zone-only logistic.  
- **Reporting:** include a single comparison table and short takeaways. Avoid random splits in the main text.

---

## Findings

> INSERT FINDINGS 

### Overall performance

**What this shows.**  
- Field position alone explains a lot. The zone-only logistic is a solid baseline.  
- Adding season and team improves discrimination and gives interpretable effects.  
- Tree-based models can lift AUROC further by capturing interactions.  
- Calibrated boosting often yields the best probability quality.

### Feature effects and explainability

- **Logistic models.** Report odds ratios. Expect large positive effects for advanced starting zones, a small positive shift for later seasons, and team effects that show style and coaching differences.  
- **Random forest and boosting.** Report permutation importance. Consider partial-dependence style summaries for start zone to show non-linear jumps.  
- **Team and season effects.** Use residual or uplift plots to show which teams over- or under-perform after controlling for context.

### Calibration

- Include a reliability plot in.
- If over-confidence is visible, show the improvement after calibration on training folds.  
- Report Brier score before and after calibration.

### Error analysis

- **By zone.** Most errors occur in mid or transition zones where outcomes are less certain.  
- **By season.** Accuracy and AUROC dip on the newest season more than on the prior one, consistent with drift.  
- **By team.** Stable pockets of under- or over-performance are useful for coaching review.

---

## Recommendations

### For the client
1. **Use context-adjusted benchmarking.** After each match, compare actual far-set rate to expected given start contexts. Review sets where expectation was high but outcome was poor, and celebrate low-expectation successes.  
2. **Build a light dashboard.** Track predicted vs actual by team, zone and season. Highlight under- and over-performance and trend lines.  
3. **Monitor drift.** Keep the 2018 benchmark fixed. Rising expected probabilities indicate league improvement. Falling values suggest rule or style changes that matter.

### For future analysts
- Keep splits time-aware; random splits overstate performance in an evolving league.  
- Test threshold sensitivity, for example 75th and 85th percentile definitions of FarSet.  
- Add opponent strength and venue if reliable and pre-set.  
- Extend to expected metres as a simple regression companion to the binary model.  
- If you add lags, ensure they are known before the set starts to remain leakage-safe.

### Final conclusion
**Data Group 1’s product** supports clean, leakage-safe modelling. With four simple models we can estimate the chance a set goes far from its starting context. The output is explainable, comparable across seasons and directly useful for review and planning.

---

## Appendix A — Reproducibility steps

NEED TO ADD CODE HERE

1. **Get the data.** Open the repository and confirm `data.csv` is present at the root.  
2. **Check structure.** Confirm about 28,991 rows and that the composite key `(gameid, Teamname, Seasonid, halfNumber, setcount)` is unique.  
3. **Create the target.**  
   - Choose the fixed benchmark from 2018, for example P80 of `maxAdvance_set` in own-half sets.  
   - Label each row `far_set = 1` if it meets or exceeds the benchmark, else `0`.  
4. **Select predictors.** Keep only pre-set fields: `Seasonid`, `Teamname`, `halfNumber`, and `HalfTag` or `setZone`.  
5. **Encode categoricals.** One-hot encode season, team, half and zone.  
6. **Split by time.** Train on 2018–2023. Hold out 2024–2025.  
7. **Train four models.**  
   - Logistic zone-only  
   - Regularised logistic full pre-set  
   - Random forest  
   - Gradient boosting, with calibration if probability quality is poor  
8. **Evaluate on the hold-out.** Report AUROC, AUPRC, accuracy, precision, recall and Brier. Include a reliability plot.  
9. **Summarise.** Fill the tables in the Findings section and write 5 to 8 lines of takeaways.  
10. **Save artifacts.** Store tables and charts in `figures/` and a short text summary in `artifacts/`.

---

## Appendix B — Support and contributors

**Support contact**  
DATA3001 NRLW Modelling Group 1  
Email: `mushfiq.ahmed19@gmail.com`  

**Contributors**  
- Data engineering and set-level table (**Data Group 1’s product**): [INSERT NAMES]  
- Derived variables and validation: [INSERT NAMES]  
- Exploratory analysis and figures: [INSERT NAMES]  
- Modelling and write-up: Group 1

---

## Appendix C — References

- Gabbett, T. (2007). Injuries in a national women’s rugby league tournament.  
- Newans, T. et al. (2021). Match demands of female rugby league players.  
- King, D. et al. (2010, 2022). Concussion and injury in rugby league.  
- Repository: *data3001-data NRLW — Change in NRLW Game Patterns (2018 to Present).*  
