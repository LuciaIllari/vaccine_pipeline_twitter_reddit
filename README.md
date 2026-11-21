# Vaccine and COVID discussion across Reddit communities

This project asks where vaccine and COVID discussion happens on Reddit, and how communities point their readers to one another. We collect posts from Reddit's API into a relational database (SQLite), pick out the vaccine- and COVID-related ones with a keyword filter built to catch disguised spellings such as "c0v!d" and "𝐯𝐚𝐱", measure that filter's precision against hand labels, and build a directed mention network from the links between subreddits. A separate section trains stance classifiers on a public dataset of labeled tweets; whether such a model can be trusted on Reddit posts is still to be tested.

This is work in progress: the collection and the keyword filter are in place, the network is being expanded, and the stance model has not yet been checked against Reddit posts (see "Next stages" below).

## The questions

- Which communities talk about vaccines and COVID, and how much of their activity does it take up?
- Which communities do they send their subscribers to, and does that network stay within a tight core of vaccine- and health-related communities, or does it reach out into the rest of Reddit?
- Can a stance model trained on labeled tweets be trusted on Reddit posts, and how would we know?

## Notebook sections

`vaccine_pipeline.ipynb` follows the order in which the data is built.

1. **Collection:** Posts from 38 seed subreddits in 9 community types (for e.g., medical professionals, health questions, vaccine-focused, parenting, alt health, conspiracy and debate, politics, news and science, COVID), each read through 20 listings: the newest posts, the top and most controversial posts of the past year and of all time, and five keyword searches sorted three ways. Every post returned is saved (not only those that match a keyword), so the filter can be refined and rerun without new API calls, and every subreddit has a true denominator. No user names are stored.
2. **Keyword filter:** 340 rules, drawn from 2 different files: `keywords.py`, and `keywords_sensitive.py` (which holds slurs and derogatory terms separately), with an explanation of why they are there. Text is normalized first (accents, stylized Unicode letters and invisible characters are removed), and the rules come in 4 kinds: 1) whole words; 2) terms matched anywhere (for languages without spaces between words, or with prefixes attached to them); 3) proximity rules ("shot" near "measles", in either order); and 4) disguised spellings, with swapped characters, repeated letters or separators. The vocabulary runs from vaccine and disease terms, policy terms and slogans of pro/anti sides, to vaccine nicknames and vaccine, measles and COVID words in 20 other languages. The matching helpers are checked by 34 tests, some of which are actually cases that should not match.
3. **Coverage:** Monthly charts of when the collected posts were written, split by the listing that found them. Since the posts were collected after the fact rather than regularly, and on a schedule, these charts instead describe the collection, rather than the history of the discussion. The notebook thus spells out which comparisons survive this.
4. **Keyword precision:** A random sample of the matches of the rules that most posts depend on, labeled by hand ("is this post about vaccines?", or about COVID for the COVID rules) with a terminal tool, `label_posts.py`, and a written set of labeling rules. Precision is reported per rule with a 95% Wilson interval, and rules under 90% are flagged for narrowing or removal.
5. **Links between communities:** Every `r/name` mention and every link to a subreddit or Reddit post (in titles, bodies and link URLs) is stored as an edge of a directed network, where an edge from A to B means a post in A pointed its readers to B; this is the notion of a link used for the online hate network of [Zheng, Sear, Illari, Restrepo and Johnson (*npj Complexity*, 2024)](https://doi.org/10.1038/s44260-024-00002-2). A first expansion collected the subreddits that the seeds point to, and subscriber counts were fetched for the subreddits that are pointed to but not collected.
6. **Synthetic labeled tweets:** The `CT_6` subset ("vaccines are unsafe") of the HuggingFace dataset `webimmunization/COVID-19-conspiracy-theories-tweets`, with its labels mapped onto stance (a tweet supporting the theory is anti, one denying it is pro, and neutral stays neutral), and 6 classifiers compared on it (logistic regression, a linear SVM, multinomial Naive Bayes, a small neural network, a random forest and gradient boosting), using TF–IDF features of word n-grams of one to four words. 
7. **To Be Updated...**

## Status on 20 November 2026

- 441,345 posts in the database: 128,495 from the 38 seeds, and the rest from the first expansion (149 subreddits, of which 141 could be collected; the other 8 are banned, private or quarantined).
- 153,764 posts (34.8%) match at least 1 keyword rule.
- 25,658 pointers to other subreddits or posts, found in 17,611 posts; they point to 3,798 different subreddits.
- The precision labeling (553 posts across 39 rules) is under way.

## Repository contents

| File | What it is |
|---|---|
| `vaccine_pipeline.ipynb` | the notebook |
| `keywords.py` | the keyword lists, with their sources |
| `keywords_sensitive.py` | slurs and derogatory terms, kept separately |
| `label_posts.py` | the terminal tool for labeling the precision sample |
| `reddit_info_EXAMPLE.txt` | the format of the credentials file |

The database (`vaccine_discourse.db`), the sheet of sampled posts for the precision labels (`keyword_precision_labels.csv`, which holds post text) and the credentials file (`reddit_info.txt`) are NOT included in the repo.

## Running it

The notebook was developed in a conda environment with Python 3.11, pandas, NumPy, Matplotlib, scikit-learn, HuggingFace's `datasets` and `ipykernel`, with `praw` pinned at 7.8.1; `requirements.txt` to follow.

4 settings at the top decide what a run does: 1) `COLLECT_REDDIT_DATA` (collect through the API), 2) `REMATCH_KEYWORDS` (match every post again, which takes ~1hr on current db), and the two expansion settings, 3) `COLLECT_WAVE1` and 4) `FETCH_LINKED_INFO`. With all 4 set to `False`, the notebook runs from a saved database and makes no API calls.

While the tweet section needs neither credentials nor the database (it downloads its dataset on the first run), the Reddit sections need one or the other (the db will not be shared). Collecting requires a Reddit API app, and Reddit closed self-service API access in November 2025, so new apps now need Reddit's approval. With credentials, copy `reddit_info_EXAMPLE.txt` to `reddit_info.txt`, fill it in, and set `COLLECT_REDDIT_DATA = True`; a full collection of the 38 seeds takes ~1hr, and finished listings are skipped on a rerun, so it can be stopped and resumed.

To label the precision sample, run `python label_posts.py` from the project folder; it shows one post at a time, saves after every answer, and resumes where it stopped.

## Privacy and limitations

- **Public data, analysed at the level of communities:** Everything collected is public, user-facing content from Reddit's API. No usernames are stored, no claims are made about individuals, the raw data is not published, and anything released will be stripped of personally identifying information.
- **The sample is NOT random:** Top and controversial listings favor posts with many votes, searches return whatever Reddit's ranking puts first, and the newest posts span weeks in some communities while it spans years in others. The notebook records which listing found each post, and makes no claims about the discussion growing or shrinking over time.
- **Keywords decide collection (instead of stance):** A post matched by a "hesitancy" term is not thereby a hesitant post (one labeled example quotes a nurse's anti-vaccine claims to report her).
- **The stance labels are not yet validated on Reddit:** The tweets are synthetic, short and cleanly separated, while Reddit posts are long and conversational; the model needs to be checked against hand-labeled Reddit posts...
- **Not for moderation/targeting:** The project describes communities and narratives in aggregate, and is not a tool for moderating content, judging users, and or giving medical advice!

## Next stages

- Finish the precision labels, then keep/narrow/drop each flagged rule, and apply the keyword additions queued during labeling (SARS-CoV-2 variant names, vaccine-injury reporting systems, vaccination-status rules and passes, and new themes such as lab-origin narratives).
- Redraw the expansion by rule rather than by hand: a core of communities explicitly about vaccines or COVID (decided by name or public description, so that communities which obscure their purpose are not missed), together with communities about health, bodies and conspiracies; then every community the core points to, collected in full, one step out and no further. This follows the core and adjacent communities of the online hate network in [Zheng et al. (2024)](https://doi.org/10.1038/s44260-024-00002-2), with keyword rules in place of expert categorization.
- Measure the mention network (whether links are returned, and which communities bridge to the rest of Reddit), and the links that leave Reddit, grouped into mainstream and alt-tech platforms, news, health sources, etc.
- Hand-label ~100s of Reddit posts with a short codebook, rebuild the tweet models as a scikit-learn `Pipeline` compared by stratified n-fold cross-validation, and report how well the best one agrees with the hand labels.
- to be continued...
