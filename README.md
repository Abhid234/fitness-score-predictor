# Fitness Score Predictor

An end-to-end ML project that predicts a 0–100 fitness score from a person's
health metrics, and explains which of their habits drove that score.

**[Live demo]([https://fitness-score-predictor.streamlit.app/])** · [Model training notebook](Model_Training.ipynb)

---

## What it does

You enter 11 health metrics — age, gender, weight, height, resting heart rate,
calorie intake, body fat %, lean mass %, sleep, weekly workout minutes, and
daily activity minutes — and the app returns a predicted fitness score out of
100, plus a plain-language breakdown of the top factors helping and hurting it.

## Data

Built on the **NHANES 2017–2018** cycle (CDC National Health and Nutrition
Examination Survey), merged from seven components on the shared `SEQN`
respondent ID:

| File | Contributes |
|---|---|
| `DEMO_J` | Age, gender |
| `BMX_J` | Height, weight |
| `DXX_J` | Body fat %, lean mass (DXA scan) |
| `BPX_J` | Resting heart rate |
| `PAQ_J` | Physical activity |
| `DR1TOT_J` | Daily calorie intake |
| `SLQ_J` | Sleep hours |

After merging, cleaning, and dropping incomplete records: **2,216 usable rows.**

### Feature engineering

- **Lean mass %** — DXA reports lean mass in grams; converted to a percentage of
  body weight so it's comparable across body sizes.
- **Sleep hours** — weekday and weekend figures combined into a 5:2 weighted average.
- **Workout minutes** — weekly minutes of deliberate exercise, from the vigorous
  and moderate *recreational* activity questions.
- **Daily activity minutes** — work plus transport activity, averaged per day.
  NHANES has no device-measured step count, so this is the closest available
  proxy for everyday movement.
- **Outlier capping** — self-reported activity included physically impossible
  values (one respondent reported ~7,100 active minutes per day). Capped at
  1,500 min/week for workouts and 720 min/day for general activity.

## The fitness score

No public dataset ships with a "fitness score out of 100," so the target label is
a composite metric built from published health guidelines (WHO activity
recommendations, standard body-composition and sleep ranges). Each factor is
scored 0–100 by distance from an ideal value, then weighted:

| Factor | Weight |
|---|---|
| Workout minutes | 20% |
| Daily activity | 15% |
| Body fat % | 15% |
| Lean mass % | 15% |
| Sleep | 15% |
| Calorie intake | 10% |
| Resting heart rate | 10% |

**Age and gender are not weighted directly.** Instead they shift the ideal
targets — healthy body fat ranges differ by gender, calorie needs and lean mass
decline with age, and so on. This makes the same body fat percentage score
differently for a 25-year-old man than for a 60-year-old woman.

Resulting distribution across the dataset: mean ≈ 59, range 19–95.

## Model

An **XGBoost regressor** trained to predict the score from the raw inputs — it
never sees the scoring formula, only examples. Hyperparameters tuned with
`RandomizedSearchCV` (50 iterations, 5-fold CV).

| Metric | Held-out test set |
|---|---|
| MAE | 1.20 |
| R² | 0.989 |

The final deployed model is retrained on all 2,216 rows after evaluation.

## Explainability

Per-prediction **SHAP** values break the score down into each feature's
contribution. The app surfaces the top three factors helping and hurting the
score. Age and gender are excluded from this display — they're real drivers, but
not something a user can act on.

## Running it locally

```bash
git clone https://github.com/Abhid234/Fitness-Score-Predictor.git
cd Fitness-Score-Predictor
pip install -r requirements.txt
streamlit run app.py
```

## Files
app.py Streamlit app
Model_Training.ipynb Data pipeline, model training, tuning, SHAP analysis
fitness_scored.csv Cleaned and labeled dataset
fitness_model_final.pkl Trained XGBoost model
requirements.txt

## Limitations

- The fitness score is a composite metric designed for this project from public
  health guidelines. It is **not** a clinically validated measure and shouldn't
  be used for health decisions.
- Because the target is a designed formula rather than an observed outcome, the
  high R² reflects the model successfully learning that function — not a
  discovery about human fitness.
- Physical activity in NHANES is self-reported, which is known to skew high.
- NHANES 2017–2018 samples US residents, so the ideal ranges and population
  distribution may not generalize elsewhere.
