# MODELLING

# Table of Contents

# DATA3001 NRLW Modelling, Group 1  
**Title:** Predicting Far Sets in the NRLW From Pre Set Context  
**Dataset used:** `data.csv` (duplicate of `product/sets.csv`) from repository **data3001-data NRLW – Change in NRLW Game Patterns (2018 to Present)**

---

## 1. Overview

This repository contains the modelling stage of our DATA3001 project. In Week 5 we delivered the data product, which was a clean, possession level table for the NRLW. Each row in that file represented one set of six for one team in one half of a match. That was the engineering part.

This document shows how that data product can actually be used for analysis. We take the set level file (`data.csv`) and build a simple, leakage safe predictive model that answers one sport relevant question.

> **Research question**  
> **Given what we know at the start of a set, what is the chance this set will be a far set?**

We define a far set as a possession whose largest forward gain is at or above a fixed metres threshold that was first measured in the 2018 season. We keep this threshold constant over time so that we can compare seasons and teams on the same scale.

This README is written for tutors, industry partners and future student groups. It explains what we did, why we chose this question, how to reproduce the steps, and how to extend it.

---

## 2. Background

### 2.1 Project motivation

The NRLW has expanded from a small competition in 2018 to a much larger one by 2025. More teams, more matches and better athletes mean the game has changed. Most public work on the women’s game focuses on physical demands, injury, concussion or conditioning. That work is useful, but it does not tell coaches which sets are tactically effective once you adjust for where they started on the field.

Our Week 5 deliverable fixed a key problem. We took raw, event level data that was hard to model on and turned it into a single, tidy CSV that analysts can use for set based questions. That file is now in the root of the repository as `data.csv`.

The natural next step is to prove that this data product is actually useful. The clearest proof is to build a model on it.

### 2.2 Why this question

We chose the question about far sets because:

1. It aligns with what the Week 5 README already suggested, that is, define a binary far set and predict it from pre set context.
2. It uses only variables that are known at the start of the set, so it is realistic and does not leak information.
3. It is easy to explain to people who are not data scientists. Everyone understands that a set that gains a lot of metres is better than a set that does not.
4. It gives the partner a way to benchmark teams and seasons, even though the competition has expanded.
5. It sets up future work such as expected metres, kick vs run effectiveness and opponent adjustments.

### 2.3 Why it is of interest

Coaches and performance staff often ask questions that sound like this:

- We started a lot of sets inside our own 20. Was it realistic to expect big gains?
- Are we actually worse this year, or are we just starting deeper?
- Are some teams better than others, even after we adjust for field position and season?

A model that gives a probability of a far set for a given starting situation can answer all of those. It turns the data into a decision support tool.

---

## 3. Data

### 3.1 Source

- **File:** `data.csv`  
- **Origin:** duplicate of `product/sets.csv` from the Week 5 data product  
- **Format:** CSV, UTF 8, header in row 1  
- **Rows:** 28,991 in the current build  
- **Unit of analysis:** one team possession (a set of six) in a given match, team and half

### 3.2 Key identifiers

Each row is uniquely identified by the composite key

- `gameid`
- `Teamname`
- `Seasonid`
- `halfNumber`
- `setcount`

This key was validated in the Week 5 build, so we do not need to do key cleaning again.

### 3.3 Main variables

These are the variables we will actually use for modelling.

- `Seasonid`  
  Season year, used for time aware splits.

- `Teamname`  
  Team identifier, used to capture team style and to allow benchmarking.

- `halfNumber`  
  Whether the set occurred in the first or second half.

- `setZone`  
  The detailed field zone at the start of the set, for example Own 20, Own 40, Midfield, Opp 40.

- `HalfTag`  
  A coarse version of the zone, one of `Own`, `Mid`, `Opp`. This is the safest variable to use if fine grained zones are noisy.

- `maxAdvance_set`  
  The largest forward gain in this set. This is what we will transform into a binary target.

There are other variables, such as `maxRun`, `maxKick` and `n_events`, but those are outcomes of the set. We do not use them as predictors because that would mix information from inside the set into a model that is supposed to work at set start.

---

## 4. Research question

### 4.1 Formal statement

>  Given only the information available at the start of a set, can we predict the probability that the set will be a far set?

We express this as a binary classification problem

\[
P(\text{far\_set} = 1 \mid \text{Seasonid}, \text{Teamname}, \text{halfNumber}, \text{setZone or HalfTag})
\]

### 4.2 Target definition

1. Take all sets from 2018.
2. Compute the 80th percentile of `maxAdvance_set` for the relevant context, for example own half sets. This is the fixed benchmark.
3. Define  
   - `far_set = 1` if `maxAdvance_set >= benchmark`  
   - `far_set = 0` otherwise
4. Keep this benchmark fixed for all later seasons. That allows us to check if the competition got better, because we are always asking if a set meets the 2018 bar.


---

## 5. Workflow

This section is for someone who wants to repeat our work from the raw repository data.

### 5.1 Steps

1. **Clone or download the repository** that contains `data.csv`.  
2. **Load `data.csv`** into your tool of choice (Python, R, Excel for inspection).  
3. **Inspect basic structure**  
   - check row count (should be 28,991)  
   - check there are no duplicated composite keys  
4. **Create the target**  
   - filter to the reference context (for example own half) if that is how you want to define the threshold  
   - compute the 80th percentile for 2018  
   - create a new column `far_set`  
5. **Select features**  
   - keep only pre set fields: `Seasonid`, `Teamname`, `halfNumber`, `setZone`, `HalfTag`  
   - drop everything that is clearly an outcome
6. **Encode categoricals**  
   - one hot encode `Teamname`  
   - one hot encode `setZone` or `HalfTag`  
   - keep `Seasonid` either as integer or as another categorical
7. **Create a time aware split**  
   - training seasons: 2018 to 2023  
   - testing seasons: 2024 to 2025  
   This is important because the NRLW is expanding and we do not want information to leak from the future into the past.
8. **Fit the model**  
   - start with logistic regression  
   - optionally fit a tree model to check if it finds the same signals
9. **Evaluate**  
   - compare to majority class  
   - report accuracy, AUROC and precision for the positive class  
   - optionally plot probability of far set by start zone and by season
10. **Document**  
    - write up assumptions  
    - write up limitations  
    - save model code in a `code/` folder or in an appendix

### 5.2 Code access

The original repository already describes how to rebuild `product/sets.csv` from raw event data using scripts in `code/`. Our modelling work assumes that step is already done. We are only adding the modelling step. We suggest adding a new file, for example `code/modelling_farset.py` or `notebooks/farset_modelling.ipynb`, that contains the steps above.

---

## 6. Modelling approach

### 6.1 Model choice

We use **logistic regression** as the main model. Reasons:

- It outputs probabilities, which is what the question asks for.
- It is easy to explain to a non technical audience.
- It shows clearly which fields (team, zone, season) push the probability up or down.

Add more models here

### 6.2 Features

- `Seasonid`  
  Captures evolution of the competition.

- `Teamname`  
  Captures tactics, coaching and roster quality.

- `halfNumber`  
  Captures fatigue or tempo.

- `setZone` or `HalfTag`  
  Captures field position, which is usually the strongest predictor.

We do **not** include `maxAdvance_set` as a predictor because it is the outcome.

### 6.3 Train and test

- **Train on:** 2018, 2019, 2020, 2021, 2022, 2023  
- **Test on:** 2024, 2025

This mimics real life, where we would build a model on past seasons and then apply it to the current one.

### 6.4 Metrics

We will report:

- **Accuracy**, to give a quick idea of performance.
- **AUROC**, to show ranking ability, which is useful if a coach wants to focus review on the most surprising sets.
- **Precision and recall for far sets**, because far sets are usually less frequent.

If data is imbalanced we can also show a PR curve.

---

## 7. Findings (outline)

Since this is the README version, we outline the findings we expect to present in the Week 11 report.

1. **Field position is the strongest driver.**  
   Sets that start in the opponent half have a much higher predicted probability of being far.

2. **Season improves the base rate.**  
   Later seasons show slightly higher probabilities even after controlling for field position. This supports the idea that the NRLW is improving.

3. **Teams differ even after controls.**  
   Some teams have positive coefficients (they produce more far sets than expected), some have negative ones.

4. **Time aware split gives lower scores than random split.**  
   This tells us that the data is drifting over time and that we should not overstate accuracy.

---

## 8. Usage

Here are three ways an analyst or partner could use this model.

1. **Post match review**  
   After a game, run all sets through the model. For each set you get  
   - predicted probability of far set  
   - actual outcome  
   Sort by (actual − predicted). Sets that were predicted low but achieved far can be studied for good play. Sets that were predicted high but failed can be studied for breakdowns.

2. **Team benchmarking**  
   Over a season, average the predicted probability for each team and compare to the actual far set rate. Teams above the diagonal are over performing. Teams below are under performing.

3. **Season monitoring**  
   Track the average predicted probability of far set for the whole competition. If it goes up, the league is getting better. If it goes down, something changed in rules, talent or weather.

---

## 9. Limitations

We include these in the final report to get marks for critical analysis.

- The far set threshold is arbitrary. A 75th percentile or a fixed 120m line would give a different class balance.
- We do not model the opponent. If a strong team plays a weak team, their sets may look better for reasons that are not in the data.
- The competition is expanding. Models trained on old seasons will always look a bit worse on new seasons. This is not a bug, it is a feature of the data.
- We are predicting tactical success, not scoring. A far set helps, but it does not guarantee points.
- Categorical features like `Teamname` can become sparse when new teams enter.

---

## 10. Support information

**Contact:**  
DATA3001 NRLW Modelling Group 1  
Email (student): `mushfiq.ahmed19@gmail.com`  

---

## 11. Contributors

- **Data engineering and set level table:** [INSERT NAMES]
- **Derived variables and validation:** [INSERT NAMES]
- **Exploratory analysis and figures:** [INSERT NAMES]
- **Report writing and README structuring:** Group effort

If future cohorts want to add models, create a new folder `models/` and add your notebook without changing `data.csv`.

---

## 12. References and background reading

- Gabbett, T. (2007). Injuries in a national women’s rugby league tournament.  
- Newans, T. et al. (2021). Match demands of female rugby league players.  
- King, D. et al. (2010, 2022). Concussion and injury in rugby league.  
- **Repository**: data3001-data NRLW, Change in NRLW Game Patterns (2018 to Present).

These references show that the women’s game has been studied from a physical perspective. Our work adds a tactical and possession level perspective on top of that.

---




# Task

# Getting start with modelling
## Linear regression
## Machine learning models
## Model comparison

# Conclusion

# Reference list

# Appendix
