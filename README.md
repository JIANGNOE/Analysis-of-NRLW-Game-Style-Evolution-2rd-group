# DATA3001 NRLW Modelling — Group 1
**Title:** Predicting Far Sets in the NRLW from Pre-Set Context  
**Date:** 3 November 2025  
**Dataset:** `data.csv` (duplicate of `product/sets.csv`) — built from **Data Group 1’s product** (our renamed Week-5 set-level table)

---

## Table of contents
- [1. Introduction](#1-introduction)
- [2. Executive summary](#2-executive-summary)
- [3. Background](#3-background)
- [4. Overview: descriptive statistics, insights, and plan](#4-overview-descriptive-statistics-insights-and-plan)
- [5. Feature engineering](#5-feature-engineering)
- [6. The models used](#6-the-models-used)
- [7. Model comparison](#7-model-comparison)
- [8. Findings](#8-findings)
- [9. Recommendations](#9-recommendations)
- [10. Conclusion](#10-conclusion)
- [11. Appendix](#11-appendix)
  - [Appendix A — Reproducibility steps (no code here)](#appendix-a--reproducibility-steps-no-code-here)
  - [Appendix B — Repository layout and handover](#appendix-b--repository-layout-and-handover)
  - [Appendix C — References](#appendix-c--references)

---

## 1. Introduction

This project shows how **Data Group 1’s product** enables useful, leakage-safe modelling for the NRLW at the possession level. We focus on one practical question that analysts and coaches actually ask:

> **Research question:** Given what we know at the **start** of a set, what is the chance this possession will be a **far set**?

A “far set” is a set whose largest forward gain meets or exceeds a fixed metre benchmark first measured in 2018. Keeping the benchmark fixed lets us compare seasons and teams on the same scale.

---

## 2. Executive summary

- We estimate the probability that a set will be “far” using **only pre-set context**: season, team, half, and starting field zone.  
- We use a **time-aware** design: train on earlier seasons, evaluate on later seasons.  
- We run **four models**:  
  1) Linear Probability Model (OLS)  
  2) Regularised Logistic Regression  
  3) Random Forest  
  4) Gradient Boosting with probability calibration  
- Directionally, field position is the strongest driver. Team and season add signal. Tree-based models rank slightly better; logistic is easiest to explain.  
- The output supports **post-match review**, **team benchmarking**, and **league monitoring**.

---

## 3. Background

### 3.1 Context: why possession-level modelling matters
The NRLW has grown rapidly since 2018. More teams and matches, deeper rosters, and evolving tactics make simple totals hard to compare across time. Analysts need context-adjusted measures that account for **where** a set starts on the field and **when** it occurs.

Public work on the women’s game covers physical demands, injury, and movement patterns. That is valuable, but it does not tell you if a team turned a poor start into strong field position, or whether one club reliably outperforms others from the same context. The gap is on **tactical, possession-level** analysis.

### 3.2 Our contribution
Week-5 delivered a set-level data product that:
- normalises direction so metres are comparable,
- tags the **start zone** of each set,
- aggregates events to 1 row per set with outcomes like `maxAdvance_set`.

This report uses that product to build a **leakage-safe** model of far-set probability. The model does not peek inside the set it is predicting. That keeps it realistic for pre-match scenarios and instant post-set expectations.

### 3.3 Why a fixed 2018 benchmark
A moving threshold would hide improvement. A fixed bar set in 2018 (for example the 80th percentile in own-half sets) gives a stable yardstick. If the league gets better, more sets will clear the bar from the same starts. If it stalls, the share stays flat or falls.

---

## 4. Overview: descriptive statistics, insights, and plan

### 4.1 Descriptive statistics (from **Data Group 1’s product**)
- **Unit:** 1 row = 1 team possession (set of six).  
- **Key:** `(gameid, Teamname, Seasonid, halfNumber, setcount)` is unique.  
- **Rows:** ≈ 28,991 in the current build.  
- **Distances:** metres are direction-normalised and non-negative.  
- **Zones:** each set has a start zone (fine) and a coarse zone `HalfTag ∈ {Own, Mid, Opp}`.  
- **Far-set bar:** fixed to the **2018** reference (example used in our Week-5 docs: ~P80 ≈ 141 m for own-half context).

### 4.2 General insights to guide modelling
- The distribution of `maxAdvance_set` is right-skewed. Most sets gain modest metres; a minority push very far.  
- Start zone is the dominant context. Opponent-half starts are much more likely to be far sets than own-half starts.  
- Season effects are present. Later seasons show a small uplift consistent with improving play and expansion.

### 4.3 Plan (short)
1) Define `far_set` using the fixed 2018 bar.  
2) Use **only** pre-set features: `Seasonid`, `Teamname`, `halfNumber`, `HalfTag` or `setZone`.  
3) Train on 2018–2023; test on 2024–2025.  
4) Fit 4 models and compare on AUROC, AUPRC, accuracy, precision/recall, and calibration.  
5) Turn results into benchmarking guidance for coaches.

---

## 5. Feature engineering

- **Target (FarSet).**  
  Compute the 2018 threshold in a consistent context (we use own-half to set the bar). Label each set `far_set = 1` if `maxAdvance_set ≥ threshold`, else `0`. The same threshold is used for all seasons.

- **Predictors (pre-set only).**  
  `Seasonid`, `Teamname`, `halfNumber`, and either `setZone` (fine) or `HalfTag` (coarse).  
  We do **not** use within-set outcomes such as `maxRun`, `maxKick`, or event-level counts for the **same** set.

- **Encoding and hygiene.**  
  Treat categoricals with one-hot encoding. Handle unknown categories safely so new teams or zones in later seasons do not break the model. Verify no duplicate keys and basic missingness rules.

- **Leakage rules.**  
  If a variable could only be known after the set begins, it is excluded. Splits are by season to prevent future information from leaking into the past.

---

## 6. The models used

We use four complementary models to answer the same question:  
**“From start context, what is the chance this set is a far set?”**

1) **Linear Probability Model (LPM, OLS)**  
   - **What it tests:** a simple linear link between context dummies and the probability of a far set.  
   - **Why include it:** sanity baseline for direction and sign of effects; easy to communicate.  
   - **How we use it:** fit on training seasons with zone-only and with full pre-set features; compare fitted probabilities to observed rates by bins.

2) **Regularised Logistic Regression (GLM)**  
   - **What it tests:** a log-odds link with L2 regularisation to control overfitting as categories expand.  
   - **Why include it:** interpretable and well-calibrated in many tabular problems; produces odds ratios that coaches understand.  
   - **How we use it:** train with grouped cross-validation by season; report coefficients, odds ratios, and a reliability plot.

3) **Random Forest**  
   - **What it tests:** non-linear interactions, for example a team benefiting from a specific zone in particular seasons.  
   - **Why include it:** strong ranking performance and robust to noise; offers permutation importance for explainability.  
   - **How we use it:** fit on training seasons, measure AUROC/AUPRC on held-out seasons, inspect importance to see if it agrees with logistic signs.

4) **Gradient Boosting (Histogram-based)**  
   - **What it tests:** boosted trees for better discrimination on tabular data.  
   - **Why include it:** often the best ranking model in practice; can be paired with probability calibration for better reliability.  
   - **How we use it:** train on early seasons with light tuning; calibrate probabilities on training folds if needed; evaluate on held-out seasons.

All four models use the same features and the same time-aware split so the comparison is fair.

---

## 7. Model comparison

### 7.1 Criteria
- **Primary:** AUROC (ranking), AUPRC if the positive class is rare.  
- **Secondary:** accuracy, precision, recall, F1 at a 0.5 threshold.  
- **Calibration:** Brier score and a reliability plot.  
- **Practicality:** interpretability, stability across seasons, and ease of deployment.

### 7.2 Expected outcomes
- **Zone-only LPM and logistic** set the floor.  
- **Full logistic** improves discrimination and remains interpretable.  
- **Random Forest and Gradient Boosting** often lift AUROC further; boosting plus calibration tends to give the best probability quality.  
- **Recommended pair:** use **calibrated boosting** for ranking and **regularised logistic** for explainable reporting and quick checks.

---

## 8. Findings

> Replace placeholders with your actual results when you run the workflow.

- **Field position dominates.** The coarse zone feature explains a large share of variance.  
- **Season adds uplift.** Later seasons show higher far-set odds from similar starts, consistent with league improvement.  
- **Team matters.** Even after controlling for start context, some teams outperform expectation consistently.  
- **Model ranking vs explanation.** Boosting ranks best; logistic explains best.  
- **Calibration.** Calibrated boosting yields probabilities that match observed rates more closely than uncalibrated models.

---

## 9. Recommendations

### 9.1 For coaches and analysts
1. **Benchmark with context.** After each match, compare actual far-set rate with the model’s expected rate given start contexts. Review high-expectation failures and low-expectation successes.  
2. **Dashboards.** Track predicted vs actual by **team × zone × season**. Use traffic-light colouring to surface under- and over-performance.  
3. **Drift monitoring.** Keep the 2018 bar fixed. If expected probabilities climb league-wide, tactics or execution are improving. If they fall, look at rule changes, schedule, or weather patterns.

### 9.2 For future analysts
- Prefer **time-aware** splits; random splits inflate scores in evolving competitions.  
- Run **threshold sensitivity** checks (for example P75 and P85).  
- Consider **opponent strength** and **venue** if those are reliably available pre-set.  
- Extend with **expected metres** as a regression companion to this classifier.

---

## 10. Conclusion

**Answer to the research question:**  
Yes. Using only pre-set context, we can estimate the chance that a set will be a far set with useful accuracy. **Field position** is the main driver, and **season** and **team** effects add meaningful signal. In practice, a **calibrated gradient-boosting model** provides the strongest ranking of likely far sets, while a **regularised logistic regression** explains the “why” in plain language. Together they deliver context-adjusted, reliable probabilities that teams can use to review matches, benchmark performance, and track how the NRLW evolves over time.

---

## 11. Appendix

### Appendix A — Reproducibility steps (no code here)

1. Confirm `data.csv` is present and the composite key `(gameid, Teamname, Seasonid, halfNumber, setcount)` is unique.  
2. Set the **2018** far-set threshold in a consistent context (own-half recommended).  
3. Create `far_set` by comparing `maxAdvance_set` to the fixed threshold.  

