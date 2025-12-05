# A Data Pipeline for Vaccine Stance Detection and Narrative Analysis (Twitter + Reddit)

> End-to-end example of data ETL, SQL, supervised modeling, and narrative analysis on vaccine discourse across platforms.

---

## 1. Project overview

This repo contains a **mini case study in analyzing online vaccine discourse across platforms**, designed to showcase:

- End‑to‑end **ETL** from public sources into a relational database,
- A **model comparison** of traditional ML classifiers for stance detection,
- Cross‑platform transfer of a stance model from Twitter → Reddit, and
- Simple **narrative / keyword analysis** and visualization of how different communities talk about vaccines.

The code is organized as a single Jupyter notebook:

- `vaccine_pipeline.ipynb`

which can be run top‑to‑bottom to reproduce the analysis.

Originally, this notebook was developed as a **code sample for an applied data scientist role in a public‑sector investigations context**, where data is used to support legal cases and policy work (e.g., anomaly detection, repeated narratives, high‑volume text triage). :contentReference[oaicite:0]{index=0}  

---

## 2. Data sources

### 2.1 Labeled tweets (Hugging Face)

We use a public dataset from Hugging Face:

- **Dataset:** `webimmunization/COVID-19-conspiracy-theories-tweets`  
- **Subset:** CT6 — tweets about the conspiracy “vaccines are unsafe”

From that subset we construct a stance label:

- `support` → **anti‑vaccine** (`stance = "anti"`)
- `deny` → **pro‑vaccine** (`stance = "pro"`)
- `neutral` → **neutral** (`stance = "neutral"`)

Notes:

- The tweets are **synthetic / LLM‑generated**, not scraped from Twitter.  
- The dataset is **relatively small and clean**, which contributes to very strong model performance and relatively sharp decision boundaries compared to real‑world social media.

### 2.2 Reddit posts (PRAW)

We collect Reddit posts via the official API using [`praw`](https://praw.readthedocs.io/):

- Subreddits: a mix of health‑related and discussion subs (e.g. `r/medicine`, `r/AskDocs`, `r/AskVet`, `r/conspiracy`, etc.).
- Filter: posts whose **title or selftext** contains at least one **COVID/vaccine‑related keyword**, based on a curated list derived from more expressive regex patterns, including:
  - Neutral / technical terms: `covid`, `covid-19`, `coronavirus`, `ncov`, `sars-cov-2`, etc.
  - Geo / nickname terms: `wuhan virus`, `china virus`, `wuflu`, `the rona`, `kung flu`, etc.

The notebook demonstrates:

- How to authenticate with Reddit via a small local config file (`reddit_info.txt`),
- How to pull posts with `subreddit.new()` and keyword filtering,
- How to respect API limits and keep the sample size moderate.

---

## 3. ETL and database design

All data is stored in a local SQLite database (`toy.db` by default) so that downstream analysis uses **SQL + pandas** rather than in‑memory only dataframes.

Tables:

- **`tweets`**

  | column      | type    | description                                           |
  |-------------|---------|-------------------------------------------------------|
  | `id`        | INTEGER | auto‑increment primary key                            |
  | `source_id` | TEXT    | synthetic tweet ID (e.g. `tf0001_ab12cd34`)           |
  | `text`      | TEXT    | tweet content                                         |
  | `stance`    | TEXT    | `pro`, `neutral`, `anti` (derived from HF label)     |
  | `platform`  | TEXT    | `"twitter"`                                           |
  | `created_at`| TEXT    | simulated timestamp for demonstration                 |

- **`reddit_posts`**

  | column           | type    | description                                          |
  |------------------|---------|------------------------------------------------------|
  | `id`             | INTEGER | auto‑increment primary key                           |
  | `reddit_id`      | TEXT    | Reddit submission ID                                 |
  | `subreddit`      | TEXT    | subreddit name                                       |
  | `url`            | TEXT    | permalink to the submission                          |
  | `title`          | TEXT    | submission title                                     |
  | `selftext`       | TEXT    | submission body                                      |
  | `full_text`      | TEXT    | title + body concatenated                            |
  | `created_utc`    | REAL    | creation time (Unix timestamp)                       |
  | `created_at`     | TEXT    | human‑readable timestamp (ISO 8601)                  |
  | `score`          | INTEGER | Reddit score                                         |
  | `num_comments`   | INTEGER | number of comments                                   |
  | `predicted_stance` | TEXT  | stance assigned by the trained model (`pro/neutral/anti`) |

The notebook walks through:

- Creating these tables if they don’t already exist,
- Inserting the HF tweets and Reddit posts,
- Verifying counts and basic schema with simple SQL queries.

---

## 4. Modeling: stance classification on tweets

We treat stance detection as a **3‑class classification problem** on tweet text:

- Labels: `pro`, `neutral`, `anti`

### 4.1 Text representation

We use **TF–IDF** features with a reasonably rich n‑gram range:

```python
tfidf = TfidfVectorizer(
    ngram_range=(1, 4),  # unigrams, bigrams, trigrams, 4‑grams
    min_df=3,
    max_df=0.9,
    stop_words="english",
)
```

This lets us capture both single words and short phrases like:
- *“big pharma”*, *“forced vaccinations”*, *“vaccine passports”*,
- *“end pandemic”*, *“vaccine safety”*, *“healthcare workers”*, etc.

The notebook also shows how to extract and interpret the most informative n‑grams per stance using a linear model.

### 4.2 Models compared

On the tweet data we train and evaluate several standard classifiers:

- `LogisticRegression` (multinomial)
- `LinearSVC` (linear SVM)
- `MultinomialNB`
- `MLPClassifier` (small feedforward neural net)
- `RandomForestClassifier`
- `GradientBoostingClassifier`

For each model we:
- Split train/test with stratification,
- Train on TF–IDF features,
- Report:
  - `accuracy`
  - `macro F1` (to balance the three classes),
  - `classification_report`,
  - Confusion matrix (including a normalized heatmap with values annotated),
  - A binary ROC/AUC view for *"anti" vs rest* to illustrate ROC tooling.

Typical results (on the synthetic HF dataset) show:
- All models achieving macro F1 ≈ 0.97–0.99,
- A small MLP achieving the best scores (accuracy ≈ 0.99, macro F1 ≈ 0.989),
- ROC AUC for "anti" vs rest ≈ 1.0, reflecting the cleanliness of the dataset.

The notebook explicitly notes that such strong performance is partly due to:
- The synthetic, LLM‑generated nature of the tweets (cleaner language, clearer class boundaries),
- The relatively small size of the dataset.

### 4.3 Interpretability: n‑grams from a linear model

Even though the MLP is chosen as the best performing model, we still fit a logistic regression model and use it to:
- Inspect the top 2–4‑word n‑grams per stance, by coefficient weight.
- Provide a human‑readable summary of which phrases are most associated with pro/anti/neutral stances.

This demonstrates how:
- A non‑linear model can be used for final predictions, while
- A linear model serves as an interpretability lens into the discourse.

## 5. Cross‑platform transfer: applying the model to Reddit

Once the best model is selected (MLP in this run), we:
- Load Reddit posts from reddit_posts.
- Transform full_text using the same TF–IDF vectorizer.
- Use the trained Twitter stance model to predict predicted_stance for each Reddit post.
- Write the updated predictions back to the reddit_posts table.

We then explore:
- Overall predicted stance counts (e.g., ~934 neutral, 567 pro, 480 anti in one run).
- How those predictions break down by subreddit and keyword.

Visualizations include:
- Stacked bar charts of keyword frequencies:
 - Tweets: keywords stacked by stance.
 - Reddit: keywords stacked by subreddit.
- Normalized versions of these charts:
 - For each keyword, the proportion coming from each stance or subreddit,
 - For each stance, the proportion of posts coming from each subreddit.

We also create a plot with:
- 3 bars (neutral / pro / anti),
- Each bar showing a normalized stack of subreddits contributing to that stance,
- Optionally excluding high‑volume subs like r/Coronavirus and r/COVID19 for a more balanced view.

## 6. How to run this notebook
### 6.1 Requirements
- Python 3.9+ (earlier 3.x may work)
- Jupyter (or JupyterLab)
- Recommended packages (example requirements.txt):
```Python
pandas
numpy
matplotlib
scikit-learn
datasets
praw
```

### 6.2 Reddit credentials (reddit_info.txt)

To pull Reddit posts, you’ll need a Reddit app and a small local config file.

- Create a Reddit “script” app at https://www.reddit.com/prefs/apps
- Create reddit_info.txt (not checked into git; add it to .gitignore) with:
```
client_id,client_secret,user_agent,user_name,password
YOUR_CLIENT_ID,YOUR_CLIENT_SECRET,my_vax_pipeline/0.1 by u_yourusername,reddit_username,reddit_password
```
- The notebook reads this file in init_reddit_client() and initializes praw.Reddit(...).

### 6.3 Steps

1. Clone the repo:
```
git clone <this-repo-url>
cd <this-repo-directory>
```

2. (Recommended) Create and activate a virtual environment.

3. Install dependencies

4. Start Jupyter

5. Open vaccine_pipeline.ipynb and run the cells top‑to‑bottom.

On the first run, the Hugging Face dataset will be downloaded automatically via the `datasets` library.

## 7. Limitations, ethics, and appropriate use

- Synthetic training data:
The tweet data comes from a synthetic HF dataset, not live Twitter. Performance numbers and language patterns may not generalize to real‑world, adversarial contexts.

- Sampling and coverage:
The Reddit sample is limited to a small set of subreddits and a keyword filter; it should be seen as a toy corpus for demonstration, not an authoritative view of online vaccine discourse.

- Model scope:
The stance labels are applied only at the text level and are intended as a triage signal to explore narratives and communities. They are not suitable for content moderation, user‑level judgments, or any form of medical advice.

- No personal targeting:
This project is about understanding patterns and narratives in aggregate. It should not be used to target or profile individuals.

- API terms:
When running the notebook yourself, please respect Reddit’s API terms and limits and avoid scraping at unnecessary scale.

## 8. What this demonstrates (for reviewers)

This repository is intended to demonstrate the following skills:

- ETL and data engineering
 - Ingesting and cleaning text data from multiple sources (Hugging Face, Reddit API).
 - Designing and populating a relational schema in SQLite.
- Using SQL + pandas for inspection and analysis.
 - Supervised & unsupervised modeling
 - Comparing several classic ML models (logistic regression, linear SVM, Naive Bayes, MLP, random forests, boosting).
 - Evaluating using accuracy, macro F1, confusion matrices, ROC/AUC.
 - Using topic‑like and n‑gram analyses to explore narratives.
- NLP and narrative analysis
 - N‑gram‑based TF–IDF representation up to 4‑grams.
 - Keyword‑based views of vaccine discourse across stances and communities.
 - Simple duplicate / repeated phrase heuristics as toy signals of templated messaging.
- Communication and caveats
 - Explicit discussion of data limitations (synthetic tweets, sample size, coverage).
 - Clear visualizations and narrative commentary aimed at a mixed technical/legal audience, which is crucial in investigative and public‑interest contexts.
