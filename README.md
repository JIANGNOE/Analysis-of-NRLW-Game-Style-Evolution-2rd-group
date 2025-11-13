# DATA3001 NRLW Modelling - Group 2

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

**To what extent do pre-set contextual factors influence the likelihood of a 'far set'?**

A far set is defined as a possession where the maximum forward gain meets or exceeds a fixed distance benchmark. This benchmark is set using a fixed reference point from the data, which is crucial for making consistent comparisons across the league's rapid expansion.

To set a consistent standard for what counts as "far," we established a fixed performance level using data from the 2018 season. We looked at all sets that started in a team's own half and identified the distance achieved by the top 20% of those sets (the 80th percentile). This calculation resulted in the number 131.8. This figure must be correctly understood as a distance of 13.18 metres (since 131.8 decimetres equals 13.18 metres). This fixed 13.18 metre standard is used across all seasons and teams to ensure all comparisons are made on the same reliable measure.

This study is focused only on information available before the set starts. This prevents data leakage and makes the model realistic for coaches. The model helps provide context-adjusted probabilities, a fairer way to measure performance that tells us if a team is achieving results above or below expectation for their specific starting position, helping answer practical questions such as:

- Given we started this set in our own half, did we perform above or below expectation?
- Is our improvement over seasons genuine, or just driven by better field position?
- Are some teams consistently outperforming others in similar contexts?

Understanding this relationship is important because the NRLW has undergone rapid expansion since 2018. Traditional raw metrics (like average metres per game) fail to account for differences in possession context, making fair comparisons difficult. Our approach resolves this by producing context-adjusted probabilities, enabling clearer performance benchmarking across different eras.

---

## 2. Executive summary

This report details the process of building and evaluating a set of machine learning models to answer the question:  
**To what extent do pre-set contextual factors influence the likelihood of a "far set"?**

The "far set" target is defined as a possession gaining at least 13.18 metres (≥ 131.8 decimetres), a benchmark fixed from the top 20% of the 2018 season data.  
The analysis uses only pre-set contextual data from the NRLW 2018–2025 seasons, ensuring the prediction reflects information actually available at the moment a set begins.

To ensure realistic forecasting, we used a time-aware train–test split, training models listed below on 2018–2024 data and evaluating performance on the unseen 2025 season. This prevents information leakage and tests whether learned patterns generalise to a future competition environment.

- Baseline Logistic Regression (with balanced classes)
- Regularised GLM (Elastic-Net Logistic Regression)
- Random Forest
- Gradient Boosting Classifier
- HistGradientBoosting Classifier

### Main Findings

**Best Model:**  
The **Regularised GLM** achieved the best balance of generalisation, stability, and interpretability on the 2025 test set. More complex tree-based models did not outperform it and showed signs of overfitting, capturing noise from earlier seasons that did not hold in the 2025 data.

**Predictive Strength:**  
The selected model achieved a test AUC of **~0.53**, which is only slightly above random chance (0.50). This indicates that pre-set context alone (Team, Season, Zone, Half) has weak predictive power for far-set outcomes. The result is consistent across all tested models.

**Key Drivers:**  
Model coefficient and permutation importance analyses show:

- Starting zone is the strongest contextual predictor of far-set likelihood.
- Team identity also contributes, reflecting tactical structure and roster effects.
- Season effects trend upward, suggesting genuine league improvement over time, independent of field position.
- Half has minimal independent impact.

**Practical Implication:**  
The limited predictive power of static context provides a powerful insight for coaching staff: set success is not pre-determined. Set outcomes are driven by in-set execution, player skill, and decision-making—factors that occur during the play itself. The model is still valuable, however, as it provides a context-adjusted expectation benchmark, allowing analysts to evaluate if a team is performing above or below expectation given their situation.

---

# 3. Background

The primary objective of this project is to quantitatively measure the influence of static, pre-set contextual factors on attacking performance in the NRLW. Our goal is to answer the question: To what extent does context alone pre-determine success?

This objective is important because the NRLW has changed dramatically since its inception in 2018, marked by rapid expansion, increased athleticism, and professionalisation. Comparing teams across these different eras using simple raw metrics like average metres gained per game is inherently unfair. A team in 2025 might appear to have superior attacking performance than a 2018 team simply because they benefited from better average field position, not necessarily because they were more skilled. This analysis aims to isolate and quantify the leverage of context, providing analysts with a fair context-adjusted expectation for set success.

In sports analytics, prediction models commonly focus on outcomes like scoringor win probability, often using complex features like player fatigue or historical opponent match-ups. However, in the realm of in-game tactical analysis for sports like Rugby League, there is a gap in studies that specifically quantify the influence of static positional context versus dynamic execution on set-level metrics. Our work contributes directly to this gap. By achieving a low predictive score, we provide a robust, data-driven quantification of the widely held coaching principle that in-set execution dominates static factors. This result validates the current tactical focus of high-performance teams.

To ensure our findings are realistic and directly applicable to setting pre-game expectations, we adhere to using only factors that are known before the set begins. This prevents data leakage, the use of information generated during the set (such as the number of passes, ruck tempo, or tackles broken) to predict the outcome itself.

We used the following pre-set contextual factors as predictors:
- Teamname: The identity of the team in possession.  
- Seasonid: The year of the competition, which captures league-wide changes in athleticism and structure.  
- HalfTag: The simplified starting zone, categorized as 'Own,' 'Mid,' or 'Opponent’s' half.  
- halfNumber and setcount: Details about the time and sequence of the possession within the game.  

These factors were used to predict the binary target variable, farSet (sets that gained over the fixed 13.18 metre benchmark), allowing us to measure the context's influence on the likelihood of a high-gain possession.


---

# 4. Overview

## Descriptive Statistics and General Insights

We began with **28,991 sets** from the **2018 to 2025** seasons.  
Using our fixed benchmark of 13.18 metres, we found the league-wide average far set rate is 19.95%, meaning roughly 1 in every 5 possessions** is a "far set."

However, this average hides key patterns:

- **Imbalance:**  
  The target variable `farSet_fixed` is imbalanced, 80% of sets are “failures” and only 20% are “successes”. This will influence our model choice and evaluation metrics.

- **League Evolution:**  
  Attacking success has increased over time, from **16.4% in 2018** to **21.2% in 2024**, suggesting genuine improvement in league quality.
  These plots show genuine improvement in attacking performance over time on a fixed benchmark, with strong dependence on starting zone.
  <img width="1800" height="1000" alt="fig_evolution_season_zone" src="https://github.com/user-attachments/assets/2d42958e-75b4-4655-a35f-9112364b853c" />
    ***Figure 1. Far-set rate by season and starting zone (collapsed).Lines show different starting zones. Over time, far-set rates gradually rise on the fixed 131.8 m benchmark; sets starting in Good Ball zones are consistently most likely to succeed, while Own Half starts are least likely.***

  <img width="2000" height="1200" alt="fig_heatmap_season_zone" src="https://github.com/user-attachments/assets/dfefc636-85ce-4e4b-9733-967e10d6a6dc" />
  
   ***Figure 2. Far-set rate by season and starting zone. Warmer colors indicate a higher probability of a far set. Across seasons, far-set rates generally rise on the fixed 13.18 m benchmark, within each season,success increases as starting field position moves closer to the opposition half.***

- **Context Matters:**  
  Starting position has a clear effect. Sets beginning deep in a team’s own end (zone `YR`) have the lowest success rate (**18.4%**), while those starting closer to the opposition’s line (`CL`) have the highest (**21.2%**).

- **Team Skill:**  
  Teams vary substantially. The **Zebras** have the best historical performance (**23.5% far-set rate**), while the **Gliders** have the lowest (**17.7%**).

## Modelling Plan

To accurately test the influence of pre-set context and simulate real forecasting conditions, we structured the experiment using a time-aware split without shuffling possessions across seasons.

- **Training Set:** 2018–2024 data (**19,550 possessions**)  
- **Test Set (Holdout):** Entire 2025 season (**9,441 possessions**)

This ensures the model learns from history and is evaluated on how well it predicts the future (2025 season).

Given the target imbalance, **accuracy** was deemed an inappropriate performance measure. Instead, the primary evaluation metric was the **Area Under the ROC Curve (AUC)**.  
AUC measures the model’s ability to discriminate between positive and negative classes with 0.5 representing random guessing and *.0 indicating perfect classification.

We compared five classification models:

1. **Baseline Logistic Regression (Balanced)**  
   A simple, robust baseline model adjusted for class imbalance.

2. **Regularised Generalised Linear Model** 
   A refined version with hyperparameter tuning via cross-validation.
   
3. **Random Forest Classifier**
   A model used to capture non-linear relationships and benchmark the maximum possible predictive score.
  
5. **Gradient Boosting Classifier**  
   A sequential ensemble model that builds trees iteratively, where each tree focuses on correcting the errors of the previous ones.

6. **Histogram-based Gradient Boosting Classifier (HistGBM)**  
   A more computationally efficient implementation of Gradient Boosting that uses feature binning to accelerate training while maintaining comparable accuracy.


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

# 6. Models Overview

To answer our research question, we trained several classification models. The goal was not just to find the most accurate model, but the one that could best generalise to unseen data.  
That is why we trained on 2018–2024 and tested on the 2025 season.

Our primary evaluation metric was **AUC (Area Under the Curve)**, which measures how well a model distinguishes between a “far set” (1) and a “normal set” (0):

- **AUC = 1.0:** Perfect model  
- **AUC = 0.5:** No better than a random coin flip  

All models used the same four features: `Seasonid`, `Teamname`, `setZone`, and `HalfTag`.


## 6.1. Model 1: Baseline Logistic Regression (Balanced)

**What is it?**  
This model served as our simple and reliable starting point. It is a standard logistic regression model configured with `class_weight='balanced'`, ensuring that both classes receive proportional attention during training.

**Why did we choose it?**  
Given the 80/20 imbalance in the target variable, this setting ensures the model assigns greater importance to the minority “far set” class. Without this adjustment, the model would tend to predict “0” for most cases, achieving high accuracy but poor usefulness.

**Results**  
The baseline logistic regression performed **respectably**, providing a **solid benchmark** for subsequent models to surpass.

---

## 6.2. Model 2: Regularised GLM (Elastic-Net)

**What is it?**  
This is an advanced version of logistic regression within the **Generalised Linear Model (GLM)** framework. It incorporates **Elastic-Net regularisation**, a combination of **L1 (Lasso)** and **L2 (Ridge)** penalties, to improve model robustness and interpretability.

**Why did we choose it?**  
Regularisation mitigates **overfitting** by discouraging overly complex or extreme coefficient values. This helps ensure that the model captures genuine, generalisable patterns rather than noise from the training data.  
Hyperparameters were optimised through **cross-validation** to achieve the best balance between bias and variance.

**Results**  
This model achieved the **highest performance** on the 2025 holdout test set, with a **Test AUC of approximately 0.53**.  
It demonstrated the best **generalisation ability**, outperforming all other models in both stability and predictive accuracy.

---

## 6.3. Model 3: Random Forest Classifier

**What is it?**  
The Random Forest is an **ensemble tree-based model** that constructs multiple decision trees using random subsets of data and features. The final prediction is obtained by averaging the outputs of all individual trees, which helps to reduce variance and improve stability.

**Why did we choose it?**  
This model can uncover **non-linear interactions** and complex dependencies between features. For instance, it can identify relationships such as “Season 2024 combined with Zone YR is low-success, but Season 2024 combined with Zone CL is high-success,” which a linear model might miss.

**Results**  
Despite its flexibility, the Random Forest exhibited **overfitting**. While performance on the training data was strong, it **failed to generalise** to the 2025 test set, performing below the Regularised GLM.

---

## 6.4. Model 4: Gradient Boosting Classifier

**What is it?**  
Gradient Boosting is another **ensemble learning** approach that builds trees sequentially. Each new tree is designed to **correct the errors** of the previous ensemble, gradually improving performance over many iterations.

**Why did we choose it?**  
This model is known for its **high predictive power** and ability to learn complex feature interactions. It is often effective in structured data problems where linear assumptions are too limiting.

**Results**  
The Gradient Boosting Classifier also **overfit** the training data and performed **worse than the Regularised GLM** on the 2025 test set. Its inability to generalise suggests it captured season-specific or team-specific noise rather than stable predictive relationships.

---

## 6.5. Model 5: Histogram-based Gradient Boosting Classifier (HistGBM)

**What is it?**  
The HistGradientBoosting Classifier is a **modern, high-performance implementation** of Gradient Boosting that accelerates training by grouping continuous features into discrete bins (“histograms”). This makes it particularly efficient on large datasets.

**Why did we choose it?**  
It represents the **state of the art** among tree-based models and is widely recognised for balancing speed, scalability, and accuracy. We included it to test whether this enhanced algorithm could identify stronger signals in the data.

**Results**  
Despite its advantages, the HistGradientBoosting Classifier also **overfit** the historical data and did not generalise well to the 2025 season. Its performance was **lower than the Regularised GLM**, reaffirming that simpler, well-regularised linear models were more effective for this problem.


---

# 7. Model Comparison

## 7.1 Evaluation metrics
Model performance was accessed using both ranking and probability calibration metrics to capture the goals of interpretability and reliability.
| Metric | Purpose| Interpretation |
|-------|---------|----------------|
| Area under the curve (AUC) | Measures ability to rank positive vs negative sets | Higher = better discrimination | 
| Average precision (AP) | Focuses on precision–recall for rare positive events | Higher = better handling of class imbalance | 
| Brier Score | Measures accuracy of predicted probabilities | Lower = better calibration | 
| LogLoss | Penalises overconfident incorrect predictions | Lower = better probabilistic fit | 

## 7.2 Comparison results
The table below summarises test-set performance across all models (2025 season).  
Best results per metric are in **bold**.

| Model | Test AUC ↑ | Test AP ↑ | Test Brier ↓ | Test LogLoss ↓ |
|--------|-------------|------------|---------------|----------------|
| Regularised GLM | **0.525** | 0.201 | 0.160 | **0.432** |
| Baseline Logistic Regression | 0.522 | 0.200 | 0.160 | 0.432 |
| Random Forest | 0.518 | 0.201 | 0.160 | 0.436 |
| Gradient Boosting Classifier | 0.519 | 0.202 | 0.160 | 0.433 |
| Gradient Boosting (Hist) | 0.521 | **0.203** | 0.160 | 0.433 |


We compare regularised GLM (logistic) with Random Forest, Gradient Boosting, and HistGradientBoosting using only pre-set features (Season, Team, Zone, Half) and a strict time-aware split (train ≤2024, test 2025).

### Why choose regularised GLM?

- Best overall on test: Highest / equal-best AUC & PR-AUC, competitive Brier; tree models give no meaningful gain.

- Less overfitting: Tree ensembles fit noise and don’t generalise as well to 2025.

- Interpretable & compliant: Coefficients/odds ratios are transparent for coaches and use only allowed pre-set inputs (no leakage), easy to refit.

- Better calibration: More reliable probabilities for decision-making.

### Key takeaways

- Even the best model has AUC ≈ 0.52–0.53, only slightly above random, meaning that pre-set context alone has limited predictive power.

- Real signal is in in-set execution (play, decisions, pressure), which is intentionally excluded here since this model is a clean, leakage safe baseline, not a high accuracy predictor.


**Model performance plots for the winning model (logreg_bal):**

- ROC Curve: `fig_roc_logistic.png`
   <img width="1600" height="1000" alt="fig_roc_logistic" src="https://github.com/user-attachments/assets/be62ba8b-8d79-4c6a-bc97-7fa0aa85724e" />
   ***Figure 3.ROC — Logistic (AUC = 0.524).The curve is only a little above the diagonal, so the model is just slightly better than random. There isn’t a clear threshold that gives both high TPR and low FPR***

- Precision–Recall Curve: `fig_pr_logistic.png`
   <img width="1600" height="1000" alt="fig_pr_logistic" src="https://github.com/user-attachments/assets/30044827-e975-48c0-ba1a-c70d0dbecaa4" />
   ***Figure 4. Precision–Recall — logistic model (test set).Average Precision ≈ 0.206, close to the class prevalence (~0.20). The curve stays near the baseline: we get high precision only at very small recall, and recall drops quickly as we try to keep precision high.This means the model can rank sets modestly but can’t retrieve many far sets without many false positives.***

- Calibration (Reliability) Curve: `fig_calibration_logistic.png`
   <img width="1600" height="1000" alt="fig_calibration_logistic" src="https://github.com/user-attachments/assets/ea39fbc7-eeef-4697-af46-7908fc95e18f" />
   ***Figure 5. Calibration — logistic model (test set).Each point compares predicted vs actual far-set rate. All points sit well below the 45° line, meaning the model overestimates the true probability of a far set, so raw probabilities should be treated cautiously or recalibrated.***

- Decile Lift Chart: `fig_decile_lift.png`
   <img width="1600" height="900" alt="fig_decile_lift" src="https://github.com/user-attachments/assets/160e18a0-24c2-465b-8e68-2a5b727ac132" />
   ***Figure 6. Decile lift — logistic model (test set).Test sets are sorted into 10 groups from lowest to highest predicted probability. The far-set rate steadily increases across deciles, meaning higher model scores generally correspond to more far sets, so the model provides a weak but sensible ranking of risk.***

---

# 8. Findings and Limitations

Since we selected the Regularised GLM, we can look inside it to see what features it found most important.

## 8.1 Most Important Features

We used two methods to find the most influential features:

- **Permutation Importance:**  
  Shuffle one feature at a time and measure how much model performance drops.  
  Bigger drops = more important features.

- **Odds Ratios:**  
  Show how each feature affects the probability of a far set.  
  - Odds ratio **> 1:** Increases far-set likelihood  
  - Odds ratio **< 1:** Decreases far-set likelihood

- Permutation Importance: `fig_perm_importance_logistic.png`
   <img width="1800" height="1200" alt="fig_perm_importance_logistic" src="https://github.com/user-attachments/assets/d0fcd38d-deed-430b-be41-7a658da99f53" />
   ***Figure 7. Permutation importance — logistic model (top 20).Bars show how much model performance drops when each variable is shuffled. Starting zone and certain team identities matter most for predicting far sets, while season and half contribute relatively little on their own.***

- Odds Ratios: `fig_odds_ratios_top20.png`
  <img width="1800" height="1200" alt="fig_odds_ratios_top20" src="https://github.com/user-attachments/assets/60ebc821-93a3-4520-8db4-95015d34761b" />
  ***Figure 8. Logistic odds ratios — top 20 effects.Bars above 1 mean a higher chance of a far set; bars below 1 mean lower chance. Certain teams and good attacking zones are linked with better outcomes, while deep own-half starts and weaker teams reduce far-set likelihood.***

### Key Insights

By looking at these plots, we found:

- **Starting Zone is Key:**  
  The model found that starting in specific zones (e.g., `setZone_GC` or `setZone_YR`) was one of the most important predictors.

- **Team Identity Matters:**  
  Team identity was also a key driver. The model learned that being `Teamname_Devils` or `Teamname_Zebras` (for example) had a significant impact on the probability of a far set, reflecting consistent team quality, roster, or tactics.

- **League Evolution is Real:**  
  The `Seasonid` feature also showed importance, with coefficients generally trending up over time, suggesting a genuine improvement in league-wide attacking ability.

- **Half is Not Important:**  
  As seen in the EDA, the `HalfTag` had a minimal independent impact on the outcome.


## 8.2 Omitted Variables and Potential Bias

This is the most critical limitation of our analysis.

We deliberately **omitted all in-set variables** such as:

- Number of passes  
- Number of kicks  
- Tackle breaks  
- Player errors  
- Player skill or speed  
- Opponent defensive quality  

We did this on purpose to prevent data leakage and stick to our research question. But this choice has a major consequence: Omitted Variable Bias.

### Example

The model shows that `Teamname_Zebras` has a high positive effect.  
Does this mean the **Zebras** are inherently better?  
No, it’s likely because:

- They have more **skilled players**, or  
- They have a **better coach**

Both are **omitted variables**.  
Since the model cannot see these true causes, it assigns credit to `Teamname_Zebras` instead.

### Why Random Assignment Helps

In a perfect scientific experiment, we could fix this bias with random assignment. If we could randomly assign all the best players to different teams each season, we would break the link between Teamname and player_skill. In that world, the Teamname variable's effect would drop to zero, and we would have an unbiased estimate.

But in real-world sports data, we cannot do this. Good players are not randomly assigned but are drafted by or signed with specific teams. This creates a strong correlation between Teamname and the omitted variables (skill, coaching).

Therefore, the feature importance we identified is biased. The model is not finding causal factors, it is simply finding the best correlations to make a prediction. This is acceptable for our prediction goal, but it means we cannot use this model to say "Team X is good because they are Team X."

---


## 9. Recommendations

Based on our findings, there are three main recommendations:

### For Coaches and Analysts: Focus on Execution, Not Context  
The most significant finding of this report is that pre-set context (starting zone, season, half) has a very weak relationship with attacking success (an AUC of only **0.525**).  
In simple terms, a set starting in a "bad" position is almost as likely to become a "far set" as one starting in a "good" position.  
The recommendation is clear, context is not an excuse for failure, nor a guarantee of success.  
Performance is almost entirely dictated by what happens during the set (execution, skill, decision-making, error reduction).  
Coaches should focus resources on improving these in-set actions rather than worrying excessively about field position.

### For Performance Benchmarking: Use This Model to Adjust Expectations  
While the model is not strongly predictive, it is useful for its original purpose which is setting a fair, context-adjusted expectations.  
For example, the model might predict a 19% chance of a far set from a deep kick return, but a 22% chance from a midfield start.  
Analysts can use these probabilities as a “pass/fail” mark:  
- If the team is consistently turning **19% chances into 25% realities**, they are performing **above expectation**.  
- If they are turning **22% chances into 18% realities**, they are **underperforming**.  

The model’s weakness is its strength since it proves that teams can and should perform from anywhere on the field.

### For Future Modelling: Include Execution Variables to Understand Why  
This model answers *“What is the likelihood?”* but not *“Why?”*  
The low AUC score strongly implies that the real reasons for success are found in the execution variables we intentionally omitted (passes, tackle breaks, errors, kicks, etc.).  
To build a model that explains attacking success, future research must include these in-set variables.  
This would shift the goal from prediction to driver analysis, which is likely more valuable for understanding how to create more successful attacks.

---

## 10. Conclusion


This report set out to answer the research question:  
**To what extent do pre-set contextual factors influence the likelihood of a 'far set'?**

To explore this, we defined a clear benchmark.  
A far set was any possession that gained more than 13.18 metres, which is the 80th percentile of own-half gains in 2018. 
We trained five machine learning models using data from 2018–2024, and then tested them on the 2025 season, which was kept separate as a true “future” test.

The results were clear.  
The more complex tree-based models (**Random Forest** and **Gradient Boosting**) **overfit** the training data and did not perform well on the 2025 test set.  
The **Regularised GLM (Elastic-Net)** model performed best, reaching a **test AUC of about 0.53**.

An **AUC of 0.53** means the model is only **slightly better than random guessing**. Hence, the answer to the research question is that [re-set contextual factors exert only a negligible influence on the likelihood of a 'far set'. This shows that pre-set context alone is a weak predictor of attacking success in the NRLW.

This is not a failure of modelling as it is an important insight about the game.  
It shows that a set’s outcome is not determined by where or when it start*, but by what happens within the set: the skill, tactics, and execution** of the players.  
The model helps quantify the baseline expectation for each possession, giving coaches and analysts a fair way to measure true performance, separate from the situation they start in.
This project concludes that static context should not be used for future prediction. Instead, the model is most valuable as a context-adjusted expectation benchmark.


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

## 12.Contact

Primary contact: , University Email: , Personal: 

Contributors: [Kevin Hang](https://github.com/kevinhang19), [Yue Li](https://github.com/Yuri12-3),  [Mengyuan Jiang](https://github.com/JIANGNOE), [Ansh Patel](https://github.com/ansh428), [Mushfiq Ahmed](https://github.com/mushfiqahmeddd)
