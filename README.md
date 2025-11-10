# DATA3001 NRLW Modelling - Group 1

<img src="images.jpeg" alt="Dashboard Preview" width="1000">


**Title:** Predicting Far Sets in the NRLW from Pre-Set Context  
**Dataset:** `data.csv` (duplicate of `product/sets.csv`) - built from **Data Group 1’s product** 


## Table of contents
- [1. Introduction](https://github.com/JIANGNOE/Analysis-of-NRLW-Game-Style-Evolution-2rd-group/blob/main/README.md#1-introduction)
- [2. Executive summary](https://github.com/JIANGNOE/Analysis-of-NRLW-Game-Style-Evolution-2rd-group/blob/main/README.md#2-executive-summary)
- [3. Background](https://github.com/JIANGNOE/Analysis-of-NRLW-Game-Style-Evolution-2rd-group/blob/main/README.md#3-background)
- [4. Overview: descriptive statistics, insights, and plan](https://github.com/JIANGNOE/Analysis-of-NRLW-Game-Style-Evolution-2rd-group/blob/main/README.md#4-overview)
- [5.EDA](https://github.com/JIANGNOE/Analysis-of-NRLW-Game-Style-Evolution-2rd-group/blob/main/README.md#5-eda-visualisation-and-initial-insights)
- [6. Models](https://github.com/JIANGNOE/Analysis-of-NRLW-Game-Style-Evolution-2rd-group/blob/main/README.md#6-models-design-and-results)
- [7. The models comparision](https://github.com/JIANGNOE/Analysis-of-NRLW-Game-Style-Evolution-2rd-group/blob/main/README.md#7-model-comparison)
- [8. Findings and limitations](https://github.com/JIANGNOE/Analysis-of-NRLW-Game-Style-Evolution-2rd-group/blob/main/README.md#8-findings-and-limitations)
- [9. Recommendations](https://github.com/JIANGNOE/Analysis-of-NRLW-Game-Style-Evolution-2rd-group/blob/main/README.md#9-recommendations)
- [10. Conclusion](https://github.com/JIANGNOE/Analysis-of-NRLW-Game-Style-Evolution-2rd-group/blob/main/README.md#10-conclusion)
- [11. Appendix](https://github.com/JIANGNOE/Analysis-of-NRLW-Game-Style-Evolution-2rd-group/blob/main/README.md#11-appendix)

---

## 1. Introduction

This project examines how pre-set contextual factors influence attacking outcomes in the National Rugby League Women’s (NRLW).

Our research question is:

> **"Given what we know at the start of a set (season, team, starting zone, half), what is the probability this possession will be a far set?"**

A **"far set"** is defined as a possession where the maximum forward gain meets or exceeds a fixed distance benchmark. For this report, that benchmark is **131.8 metres**, calculated as the **80th percentile** of gains from own-half starts in the 2018 season. Keeping this benchmark fixed allows us to compare teams and seasons on a consistent scale, avoiding "moving goalposts" as the league expands and evolves.

This modelling task is directly grounded in how coaches and analysts review performance. By predicting the likelihood of a "far set" using only information available before the set begins (season, starting field zone, half, team identity), we ensure a **"leakage-safe"** model that reflects real decision-making conditions during games.

Such a model provides context-adjusted expectations for possession quality, helping answer practical questions such as:

- “Given we started this set in our own half, did we perform above or below expectation?”
- “Is our improvement over seasons genuine, or just driven by better field position?”
- “Are some teams consistently outperforming others in similar contexts?”

Understanding this relationship is important because the NRLW has undergone **rapid expansion** since 2018. Traditional raw metrics (like average metres per game) fail to account for differences in possession context, making fair comparisons difficult. Our approach resolves this by producing **context-adjusted probabilities**, enabling clearer performance benchmarking across different eras.


---

## 2. Executive summary

This report details the process of building and evaluating four machine learning models to predict the probability of a **"far set"** (a successful attacking possession > 131.8m). The models use only **pre-set contextual data** from the **NRLW 2018–2025 seasons**.

The analysis used a **time-aware split**, training on data from **2018–2024** and holding out the entire **2025 season** as the unseen test set. We compared four classification models:

- Baseline Logistic Regression (with balanced classes)
- Tuned Logistic Regression (with cross-validation)
- Random Forest
- Gradient Boosting Classifier
- HistGradientBoosting Classifier

---

### Main Findings

**Best Model:**  
The **baseline (balanced) Logistic Regression** was the best-performing model on the 2025 test set, achieving a **test AUC of 0.525**. This indicates that the more complex tree-based models were unable to find significant predictive patterns and did not generalize well to the future season.

**Limited Predictive Power:**  
An AUC of **0.525** suggests that the pre-set factors alone (Season, Team, Zone, Half) have very limited ability to predict whether a set will be successful. The model is only slightly better than a random 50/50 guess.

**Key Drivers:**  
The most important predictors identified by the model were specific **starting zones** (like `setZone_GC`) and specific **teams** (like `Teamname_Devils` and `Teamname_Zebras`), rather than broader factors like the game half or season.

**Practical Implication:**  
The low predictive power of context strongly implies that **in-set execution** (factors like passes, kicks, player skill, and errors, which were intentionally excluded from this model's predictors) are the true drivers of attacking success.


---

# 3. Background

This report is built to answer a specific, practical question for coaches and analysts:

> **"Given what we know at the start of a set (season, team, starting zone, half), what is the probability this possession will be a far set?"**

This question is important because the **NRLW has changed dramatically since 2018**. Comparing teams across different seasons using simple stats like *average metres gained* is unfair. A team in 2024 might look better than a 2018 team simply because they started in better field positions, not because they were more skilled.

To solve this, we first need a **consistent definition of success**.  
We define a **"far set"** as any possession that gains **131.8 metres or more**.  
This value represents the **80th percentile of metres gained from own-half starts in 2018**. By fixing this benchmark, we avoid “moving the goalposts” and can compare performances across seasons fairly.

Crucially, our models only use predictors that a coach would know **before** the set begins:

- **Season:** 2018, 2019, etc.  
- **Team:** Which team has the ball.  
- **Starting Zone:** Where on the field the set starts (e.g., deep in their own half, midfield).  
- **Half:** First or second half.

This approach prevents **data leakage**, we are not using information from within the set (like number of passes or tackle breaks) to predict the outcome. This keeps the model realistic for setting **real-world expectations**, helping answer questions like:

> “We started this set deep in our territory. Did we perform above or below what was expected for that situation?”

---

# 4. Overview

## Descriptive Statistics and General Insights

We began with **28,991 team possessions ("sets")** from the **2018 to 2025** seasons.  
Using our fixed benchmark of **131.8 metres**, we found the league-wide average *far set rate* is **19.95%**, meaning roughly **1 in every 5 possessions** is a "far set."

However, this average hides key patterns:

- **Imbalance:**  
  The target variable `farSet_fixed` is imbalanced, 80% of sets are “failures” (0) and only 20% are “successes”. This will influence our model choice and evaluation metrics.

- **League Evolution:**  
  Attacking success has increased over time, from **16.4% in 2018** to **21.2% in 2024**, suggesting genuine improvement in league quality.

- **Context Matters:**  
  Starting position has a clear effect. Sets beginning deep in a team’s own end (zone `YR`) have the lowest success rate (**18.4%**), while those starting closer to the opposition’s line (`CL`) have the highest (**21.2%**).

- **Team Skill:**  
  Teams vary substantially. The **Zebras** have the best historical performance (**23.5% far-set rate**), while the **Gliders** have the lowest (**17.7%**).

## Modelling Plan

We used a **time-aware split** to simulate real forecasting, no shuffling across seasons.

- **Training Set:** 2018–2024 data (**19,550 possessions**)  
- **Test Set (Holdout):** Entire 2025 season (**9,441 possessions**)

This ensures the model learns from history and is evaluated on how well it predicts the *future* (2025 season).

Because our data is imbalanced (80/20), simple accuracy is misleading, a model predicting “0” every time would be **80% accurate but useless**.  
Instead, our primary evaluation metric is **AUC (Area Under the ROC Curve)**, which measures how well the model distinguishes between “far” and “normal” sets.

We compared **four classification models**:

1. **Logistic Regression (Balanced):**  
   A simple, robust baseline model adjusted for class imbalance.

2. **Tuned Logistic Regression (CV):**  
   A refined version with hyperparameter tuning via cross-validation.

3. **Random Forest:**  
   An ensemble of decision trees that averages multiple models for stability.

4. **Gradient Boosting (HistGradient):**  
   A high-performing model that sequentially learns from previous mistakes.

---

# 5. EDA: Visualisation and Initial Insights

Before modelling, we explored how each predictor (Zone, Season, Team, Half) relates to the outcome (*Far Set Rate*).

## 5.1 Far-Set Rate by Starting Zone

| setZone | sets | far_rate |
|----------|------|-----------|
| CL | 1370 | 0.212409 |
| GL | 3160 | 0.210759 |
| CR | 1339 | 0.209111 |
| CC | 3141 | 0.207259 |
| GR | 3052 | 0.204128 |
| GC | 4902 | 0.200734 |
| YC | 5067 | 0.193408 |
| YL | 3410 | 0.191496 |
| YR | 3550 | 0.184507 |

**Insight:**  
Starting position strongly affects success. Sets starting deep in a team’s half (like `YR`) have lower success rates (18.4%) compared to those near the opposition’s line (`CL`) at 21.2%.  
This confirms that `setZone` is an essential predictor.


## 5.2 Far-Set Rate by Season (League Evolution)

| Season | sets | far_rate |
|---------|------|-----------|
| 2018 | 775 | 0.163871 |
| 2019 | 753 | 0.172643 |
| 2020 | 732 | 0.199454 |
| 2021 | 2314 | 0.189715 |
| 2022 | 2329 | 0.202233 |
| 2023 | 6179 | 0.206830 |
| 2024 | 6468 | 0.211812 |
| 2025 | 9441 | 0.192988 |

**Insight:**  
There’s a clear upward trend in attacking performance from 2018 to 2024, reflecting improved tactics, coaching, and athleticism.  
The slight dip in 2025 (our test set) could be due to league expansion or natural variation.

## 5.3 Far-Set Rate by Team

| Teamname | sets | far_rate |
|-----------|------|-----------|
| Zebras | 2709 | 0.235142 |
| Rhinos | 1910 | 0.210995 |
| Rams | 3252 | 0.210332 |
| Quokkas | 2621 | 0.209462 |
| Galahs | 3404 | 0.197709 |
| Sugartails | 3572 | 0.194849 |
| Armadillos | 1889 | 0.193224 |
| Dingoes | 1951 | 0.192722 |
| Devils | 2809 | 0.187255 |
| Hamsters | 1189 | 0.184188 |
| Cockatoos | 722 | 0.180055 |
| Gliders | 2963 | 0.177185 |


**Insight:**  
Team identity clearly influences outcomes. The **Zebras** outperform all others (23.5%), while the **Gliders** lag behind (17.7%). This justifies including `Teamname` in the model.

## 5.4 Far-Set Rate by Half-Tag

| HalfTag | sets | far_rate |
|----------|------|-----------|
| Mid | 5850 | 0.208889 |
| Opp | 11114 | 0.204517 |
| Own | 12027 | 0.190239 |


**Insight:**  
The game half has a weak relationship with success. Far-set rates are similar across halves, 20.9% (`Mid`), 20.4% (`Opp`), and 19.0% (`Own`).  
We’ll include it for completeness, but it’s unlikely to be a major driver.

---

# 6. Models: Design and Results

To answer our research question, we trained several classification models. The goal was not just to find the most accurate model, but the one that could best **generalise to unseen data**.  
That’s why we trained on **2018–2024** and tested on the **2025** season.

Our primary evaluation metric was **AUC (Area Under the Curve)**, which measures how well a model distinguishes between a “far set” (1) and a “normal set” (0):

- **AUC = 1.0:** Perfect model  
- **AUC = 0.5:** No better than a random coin flip  

All models used the same four features: `Seasonid`, `Teamname`, `setZone`, and `HalfTag`.


## 6.1 Model 1: Logistic Regression (Balanced)

**What is it?**  
A simple, reliable baseline model that fits a linear boundary between the two classes. It’s fast, interpretable, and ideal as a starting point.

**Why we chose it:**  
We used `class_weight='balanced'` so the model pays extra attention to the rare class (far sets = 20%). This prevents it from defaulting to always guessing “0.”

**Results:**  
This baseline achieved a **Test AUC of 0.525**.  
While modest, it set the benchmark for all other models. It’s slightly better than random guessing, showing a weak but real signal.

## 6.2 Model 2: Tuned Logistic Regression (CV)

**What is it?**  
Same model type as above, but fine-tuned using **cross-validation** to find the “best” hyperparameters on the 2018–2024 data.

**Why we chose it:**  
To see if optimization could improve the baseline without adding complexity.

**Results:**  
The tuned version scored **Test AUC = 0.522**, slightly worse than the simple baseline.  
This suggests **overfitting**, the tuning improved performance on old data but didn’t generalize to 2025.

## 6.3 Model 3: Random Forest

**What is it?**  
An ensemble model that builds **hundreds of decision trees** on random subsets of the data and averages their predictions.

**Why we chose it:**  
Random Forests can capture **non-linear relationships** (e.g., “Zone YR performs worse only in 2024”). We hoped it would find deeper interactions missed by logistic regression.

**Results:**  
The model scored **Test AUC = 0.518**, showing **poor generalization**.  
It fit the training data too well, learning noise and season-specific quirks that didn’t carry over to 2025.

## 6.4 Model 4: Gradient Boosting (HistGradientBoosting)

**What is it?**  
An advanced tree-based model that builds trees **sequentially**, each one correcting the errors of the previous. We used the efficient **HistGradientBoosting** implementation.

**Why we chose it:**  
Gradient Boosting Machines (GBMs) often win data science competitions. They’re designed to squeeze maximum predictive power from structured data.

**Results:**  
The HistGradientBoosting model scored **Test AUC = 0.521**, again below the simple baseline.  
Despite its sophistication, it **overfit** the training data and failed to improve real-world performance.

---

# 7. Model Comparison

The table below summarises test-set performance across all models (2025 season).  
Best results per metric are in **bold**.

| Model | Test AUC ↑ | Test AP ↑ | Test Brier ↓ | Test LogLoss ↓ |
|--------|-------------|------------|---------------|----------------|
| Logistic Regression (Balanced) | **0.525** | 0.201 | 0.160 | **0.432** |
| Logistic Regression (Tuned CV) | 0.522 | 0.200 | 0.160 | 0.432 |
| Random Forest | 0.518 | 0.201 | 0.160 | 0.436 |
| Gradient Boosting (Standard) | 0.519 | 0.202 | 0.160 | 0.433 |
| Gradient Boosting (Hist) | 0.521 | **0.203** | 0.160 | 0.433 |


As seen from the table, **Logistic Regression (Balanced)** is the selected model. Despite its simplicity, it outperformed all complex alternatives.  
The Random Forest and Gradient Boosting models **overfit** and failed to generalize to the 2025 data.  
The balanced logistic model was **most stable and reliable**.

However, its **AUC = 0.525** is still very low, only 2.5% better than random guessing.  
This means that the pre-set features (Season, Team, Zone, Half) alone have **weak predictive power**.

In other words, knowing this context isn’t enough to accurately predict success.  
The real drivers of far sets are likely **in-set execution factors**, which were intentionally excluded.


### [INSERT FIGURES HERE]

**Model performance plots for the winning model (logreg_bal):**

- ROC Curve: `fig_roc_logistic.png`  
- Precision–Recall Curve: `fig_pr_logistic.png`  
- Calibration (Reliability) Curve: `fig_calibration_logistic.png`  
- Decile Lift Chart: `fig_decile_lift.png`

---

# 8. Findings and Limitations

Since we selected the **Logistic Regression (Balanced)** model, we can interpret its internal coefficients to understand what it learned.

## 8.1 Most Important Features

We used two methods to find the most influential features:

- **Permutation Importance:**  
  Shuffle one feature at a time and measure how much model performance drops.  
  Bigger drops = more important features.

- **Odds Ratios:**  
  Show how each feature affects the probability of a far set.  
  - Odds ratio **> 1:** Increases far-set likelihood  
  - Odds ratio **< 1:** Decreases far-set likelihood

### [INSERT FIGURES HERE]

- Permutation Importance: `fig_perm_importance_logistic.png`  
- Odds Ratios: `fig_odds_ratios_top20.png`

### Key Insights

- **Specific Zones Matter:**  
  Zones like `setZone_GC` were among the most important predictors.

- **Team Identity Matters:**  
  Variables like `Teamname_Devils` or `Teamname_Zebras` strongly influenced predictions.

- **Broad Factors Don’t:**  
  Variables like `Seasonid` and `HalfTag` contributed very little, the model found more signal in team and zone context.

## 8.2 Omitted Variables and Potential Bias

This is the most critical limitation of our analysis.

We deliberately **omitted all in-set variables** such as:

- Number of passes  
- Number of kicks  
- Tackle breaks  
- Player errors  
- Player skill or speed  
- Opponent defensive quality  

We did this to avoid **data leakage** and focus on pre-set context.  
However, this creates **Omitted Variable Bias**, when unobserved variables distort the apparent effect of those included.

### Example

The model shows that `Teamname_Zebras` has a high positive effect.  
Does this mean the **Zebras** are inherently better?  
No, it’s likely because:

- They have more **skilled players**, or  
- They have a **better coach**

Both are **omitted variables**.  
Since the model can’t see these true causes, it assigns credit to `Teamname_Zebras` instead.

### Why Random Assignment Helps

In theory, randomizing player assignment across teams would break this link.  
If all players were reshuffled each season, `Teamname` would no longer capture player skill or coaching quality, its effect would drop to zero.  
That’s what would remove the bias.

But in real-world sports data, **random assignment isn’t possible**.  
Teams recruit and retain top players, meaning that `Teamname` is inherently correlated with hidden skill variables.

As a result, our feature importance reflects **correlation, not causation**.  
The model finds patterns useful for **prediction**, not for explaining *why* teams perform as they do.

---


## 9. Recommendations

Based on our findings, we have three main recommendations:

### For Coaches and Analysts: Focus on Execution, Not Context  
The most significant finding of this report is that pre-set context (starting zone, season, half) has a very weak relationship with attacking success (an AUC of only **0.525**).  
In simple terms, a set starting in a "bad" position is almost as likely to become a "far set" as one starting in a "good" position.  
The recommendation is clear: context is not an excuse for failure, nor a guarantee of success.  
Performance is almost entirely dictated by what happens **during** the set (execution, skill, decision-making, error reduction).  
Coaches should focus resources on improving these in-set actions rather than worrying excessively about field position.

### For Performance Benchmarking: Use This Model to Adjust Expectations  
While the model is not strongly predictive, it is useful for its original purpose: setting a fair, context-adjusted expectation.  
For example, the model might predict a 19% chance of a far set from a deep kick return, but a 22% chance from a midfield start.  
Analysts can use these probabilities as a “pass/fail” mark:  
- If the team is consistently turning **19% chances into 25% realities**, they are performing **above expectation**.  
- If they are turning **22% chances into 18% realities**, they are **underperforming**.  

The model’s weakness is its strength: it proves that teams can and should perform from anywhere on the field.

### For Future Modelling: Include Execution Variables to Understand Why  
This model answers *“What is the probability?”* but not *“Why?”*  
The low AUC score strongly implies that the real reasons for success are found in the execution variables we intentionally omitted (passes, tackle breaks, errors, kicks, etc.).  
To build a model that explains attacking success, future research must include these in-set variables.  
This would shift the goal from **prediction** to **driver analysis**, which is likely more valuable for understanding how to create more successful attacks.

---

## 10. Conclusion

This report set out to answer a practical coaching question:  
> “Given what we know at the start of a set (season, team, starting zone, half), what is the probability this possession will be a far set?”

To do this, we established a fixed performance benchmark:  
A “far set” was defined as any possession gaining over **131.8 metres** (the 80th percentile of own-half gains in 2018).  
We trained four different machine learning models, **Logistic Regression**, **Tuned Logistic Regression**, **Random Forest**, and **HistGradient Boosting** — on data from **2018–2024**, holding back the entire **2025 season** as a true “out-of-time” test set.

The results were conclusive.  
The more complex models (Random Forest, Gradient Boosting) **overfit** the training data and failed to generalize to the 2025 season.  
The best-performing model was the simplest: a **Balanced Logistic Regression**, which achieved a **Test AUC of 0.525**.

This “winning” score is the most important finding.  
An AUC of 0.525 means the model is only **2.5% better than a random coin flip** at predicting the outcome.  
This tells us, definitively, that **pre-set context alone is a very poor predictor** of attacking success in the NRLW.

This finding is **not a failure** of the model, but a **deep insight** into the sport.  
It proves that the outcome of a set is not predetermined by the starting situation.  
Success is overwhelmingly driven by the **skill, tactics, and execution** that happen within the set itself.  

This model successfully quantifies the starting-point expectation, providing a **fair baseline** that coaches and analysts can use to measure true team performance, separate from the context they play in.

---

## 11. Appendix

### Appendix A — Reproducibility steps 

### Appendix B — Contributers
ADD ALL that info here later, who did what etc

### Appendix C — References
- Repository: *data3001-data NRLW — Change in NRLW Game Patterns (2018 to Present).*
  
- Gabbett, T. (2007). Injuries in a national women’s rugby league tournament.  
- Newans, T. et al. (2021). Match demands of female rugby league players.  
- King, D. et al. (2010, 2022). Concussion and injury in rugby league.  


