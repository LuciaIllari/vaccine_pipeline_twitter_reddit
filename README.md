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
