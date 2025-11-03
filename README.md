# DATA3001 NRLW Modelling — Group 1
**Title:** Predicting Far Sets in the NRLW from Pre-Set Context  
**Dataset:** `data.csv` (duplicate of `product/sets.csv`) — built from **Data Group 1’s product** 

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

---

## 1. Introduction

This project shows how **Data Group 1’s product** enables useful, leakage-safe modelling for the NRLW at the possession level. We focus on one practical question that analysts and coaches actually ask:

> **Research question:** Given what we know at the **start** of a set, what is the chance this possession will be a **far set**?

A “far set” is a set whose largest forward gain meets or exceeds a fixed metre benchmark first measured in 2018. Keeping the benchmark fixed lets us compare seasons and teams on the same scale.

---

## 2. Executive summary

**What we asked:**  
Given only what we know at the **start** of a set (season, team, half, starting zone), what is the **chance** this possession will be a **far set**?

**What we did:**  
We built a leakage-safe, **context-adjusted** probability using four simple models (Linear Probability Model, Regularised Logistic Regression, Random Forest, Gradient Boosting with calibration). We trained on earlier seasons and tested on later seasons to respect how the NRLW has evolved.

**What this gives you:**  
- A fair benchmark that controls for **where** sets start and **when** they occur.  
- Clear team and season comparisons that are not biased by field position.  
- A simple way to spot **over-performance** and **under-performance** after each game.

**Bottom line:**  
Yes — pre-set context is enough to estimate far-set likelihood with useful accuracy. Start zone drives most of the signal; team and season add meaningful lift. Use the calibrated model for ranking/monitoring and the logistic model for explainable reporting.


---

## 3. Background

**Objectiveo**  
Turn our set-level dataset into a working modelling example that predicts **far-set probability from pre-set context** — no peeking at events inside the set we’re predicting.

**Why this matters — why the client should care**  
- The NRLW has expanded since 2018. Raw metrics (totals/averages) aren’t comparable across seasons.  
- Coaches ask context questions: “From our own 20, what should we expect?” “Did we underperform relative to where we started?”  
- A **probability at set start** answers those questions in plain language and works for post-match review, team benchmarking and league-wide monitoring.

**Prior work — what exists and what’s missing**  
Published work on women’s rugby league is strong on **physical demands**, **injury** and **movement**. Possession-level **tactical** models — especially for the NRLW — are limited. Our contribution is to (1) provide a clean, **set-level** dataset and (2) demonstrate a **leakage-safe** modelling approach that turns context into expected outcomes coaches can act on.

**Our framing choice — fixed 2018 benchmark**  
We define “far” using a **fixed bar** (e.g., the 2018 80th percentile for own-half sets). Keeping the bar fixed lets us see genuine improvement or decline across eras, instead of moving the goalposts each season.


---

## 4. Overview

### 4.1 Descriptive statistics (from **Data Group 1’s product**)
- **Unit:** 1 row = 1 team possession (set of six).  
- **Key:** `(gameid, Teamname, Seasonid, halfNumber, setcount)` is unique.  
- **Rows:** ≈ 28,991 in the current build.  
- **Distances:** metres are direction-normalised and non-negative.  
- **Zones:** each set has a start zone (fine) and a coarse zone `HalfTag ∈ {Own, Mid, Opp}`.  
- **Far-set bar:** fixed to the **2018** reference (example used in our Week-5 docs: ~P80 ≈ 141 m for own-half context).

### 4.2 General insights
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

We use four  models to answer the research question:  

1) **Linear Probability Model**  
   - **What it tests:** a simple linear link between context dummies and the probability of a far set.  
   - **Why include it:** sanity baseline for direction and sign of effects; easy to communicate.  
   - **How we use it:** fit on training seasons with zone-only and with full pre-set features; compare fitted probabilities to observed rates by bins.

2) **Regularised Logistic Regression**  
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

> INSERT FINDINGS HERE WHEN WE ACTUALLY GOT FINDINGS


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

Using only pre-set context, we can estimate the chance that a set will be a far set with useful accuracy. **Field position** is the main driver, and **season** and **team** effects add meaningful signal. In practice, a **calibrated gradient-boosting model** provides the strongest ranking of likely far sets, while a **regularised logistic regression** explains the “why” in plain language. Together they deliver context-adjusted, reliable probabilities that teams can use to review matches, benchmark performance, and track how the NRLW evolves over time.

---

## 11. Appendix

### Appendix A — Reproducibility steps 

WRITE HERE HOW TO REPRODUCE THE CODE IN STEPS PLUS ALSO ADD THE CODE WE USED TO GET FINDINGS HERE AND ALSO SOME PLOTS

### Appendix B — Contributers
ADD ALL that info here later, who did what etc

### Appendix C — References
- Repository: *data3001-data NRLW — Change in NRLW Game Patterns (2018 to Present).*
  
- Gabbett, T. (2007). Injuries in a national women’s rugby league tournament.  
- Newans, T. et al. (2021). Match demands of female rugby league players.  
- King, D. et al. (2010, 2022). Concussion and injury in rugby league.  


