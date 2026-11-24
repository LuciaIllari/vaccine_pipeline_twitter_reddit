# Vaccine and COVID discussion across Reddit communities

This project asks where vaccine and COVID discussion happens on Reddit, how communities point their readers to one another, and where they send them off Reddit. We collect posts from Reddit's API into a relational database (SQLite), pick out the vaccine- and COVID-related ones with a keyword filter built to catch disguised spellings such as "c0v!d" and "𝐯𝐚𝐱", measure that filter's precision against hand labels, build a directed mention network from the links between subreddits, and sort the links that leave Reddit by the platform they lead to. A separate section trains stance classifiers on a public dataset of labeled tweets; whether such a model can be trusted on Reddit posts is still to be tested.

This is work in progress: the collection and the keyword filter are in place, the links that leave Reddit are being traced back to the channels and accounts that posted them, and the stance model has not yet been checked against Reddit posts (see "Next stages" below).

## The questions

- Which communities talk about vaccines and COVID, and how much of their activity does it take up?
- Which communities do they send their readers to, and does that network stay within a tight core of vaccine- and health-related communities, or does it reach out into the rest of Reddit?
- When they send their readers off Reddit, where do they go: to mainstream platforms, to alt-tech ones, or to news and health sources?
- Can a stance model trained on labeled tweets be trusted on Reddit posts, and how would we know?

## Notebook sections

`vaccine_pipeline.ipynb` follows the order in which the data is built.

1. **Collection.** Posts from 38 seed subreddits in nine community types (medical professionals, health questions, vaccine-focused, parenting, alternative health, conspiracy and debate, politics, news and science, and COVID), each read through 21 listings: the newest and the hot posts, the top and most controversial posts of the past year and of all time, and five keyword searches sorted three ways. Every post returned is saved (not only those that match a keyword), so the filter can be refined and rerun without new API calls, and every subreddit has a true denominator. No author names are stored.
2. **Keyword filter.** 466 rules, drawn from two files: `keywords.py`, and `keywords_sensitive.py`, which holds slurs and derogatory terms separately, with an explanation of why they are there. Text is normalized first (accents, stylized Unicode letters and invisible characters are removed), and the rules come in five kinds: whole words; terms matched anywhere (for languages without spaces between words, or with prefixes attached to them); proximity rules ("shot" near "measles", in either order), some with phrases that cancel the match (a "mandatory Vitamin K shot" for a newborn is not a vaccine mandate); phrases run together or hyphenated ("#bigpharma", "big-pharma"); and disguised spellings, with swapped characters, repeated letters or separators, on their own or near a vaccine word ("pf!zer booster"). The vocabulary runs from vaccine and disease terms, policy terms and the slogans of each side, to vaccine nicknames, lab-origin and treatment narratives, and vaccine, measles and COVID words in 20 other languages. Every group of rules is about vaccines, about COVID, or context only: `health_misinfo` holds arguments against vaccines that never name a vaccine (virus denial, "catch measles naturally"), and is kept apart from both counts, since whether such a post is about vaccines depends on context that pattern matching cannot see. The matching helpers are checked by 44 tests, some of which are cases that should not match.
3. **Coverage.** Monthly charts of when the collected posts were written, split by the listing that found them. Since the posts were collected after the fact rather than on a schedule, these charts describe the collection rather than the history of the discussion, and the notebook spells out which comparisons survive that.
4. **Keyword precision.** A random sample of the matches of the rules that most posts depend on, labeled by hand ("is this post about vaccines?", or about COVID for the COVID rules) with a terminal tool, `label_posts.py`, and a written set of labeling rules. Precision is reported per rule with a 95% Wilson interval, and rules under 90% are flagged for narrowing or removal.
5. **Links between communities.** Every `r/name` mention and every link to a subreddit or Reddit post (in titles, bodies and link URLs) is stored as an edge of a directed network, where an edge from A to B means a post in A pointed its readers to B; this is the notion of a link used for the online hate network of [Zheng, Sear, Illari, Restrepo and Johnson (*npj Complexity*, 2024)](https://doi.org/10.1038/s44260-024-00002-2). A first expansion collected the subreddits that the seeds point to, and subscriber counts were fetched for the subreddits that are pointed to but not collected. Every collected subreddit is a node of the network, including those that point nowhere, and the network is exported for Gephi.
6. **Links that leave Reddit.** Every web address in a post's title, body or link URL that is not Reddit's own, stored as written (a YouTube video's ID sits in the query string) together with its host and its site (`bioclandestine.substack.com` becomes `substack.com`). The sites are then matched against `platforms.csv`, a table of 78 platforms in which every attribute a grouping rests on comes from an outside source or a technical definition rather than our own judgment: whether the platform is on Wikipedia's list of the most popular social platforms (at least 100 million monthly users), whether it is in the table of Wikipedia's Alt-tech article, named only in its text, or absent, its architecture (centralized, federated, relay-based) and its main function. Groupings such as mainstream vs alt-tech are then built from these columns, and a platform no post links to stays in the table, since that is a result too. A last cell tries, on a handful of links, the two lookups that name the account behind a link whose address hides it (a platform's oEmbed endpoint for videos, and following a short link's redirects), before running them on every link.
7. **Synthetic labeled tweets.** The `CT_6` subset ("vaccines are unsafe") of the Hugging Face dataset `webimmunization/COVID-19-conspiracy-theories-tweets`, with its labels mapped onto stance (a tweet supporting the theory is anti, one denying it is pro, and neutral stays neutral), and six classifiers compared on it (logistic regression, a linear SVM, multinomial Naive Bayes, a small neural network, a random forest and gradient boosting), using TF–IDF features of word n-grams of one to four words. One might hope that the scores here say something about how the model would do on Reddit, but that is not the case: the tweets are synthetic and almost cleanly separable, so the scores describe the dataset rather than the model's usefulness elsewhere.

## Status on 23 November 2026

- 590,094 posts in the database: 137,465 from the 38 seeds, 452,628 from the first expansion (149 subreddits, of which 141 could be collected; the other 8 are banned, private or quarantined), and one stray post that arrived inside another subreddit's listing.
- 155,935 posts (26.4%) match at least one vaccine or COVID rule; 375 of the 466 rules match at least one post.
- 37,162 pointers to other subreddits or posts, found in 23,506 posts; they point to 5,761 different subreddits, 5,577 of them outside the collection. Among the 180 collected subreddits, the mention network has 1,774 edges, drawn from 7,730 posts, and 189 pairs of subreddits point to each other.
- 305,614 links that leave Reddit, found in 180,666 posts (30.6%), to 23,938 sites; the 500 most-linked sites account for 73.3% of the post–site pairs. 45,230 of the posts with an outside link (25.0%) link to a platform in `platforms.csv`, and 20 of the 78 listed platforms are never linked to.
- The precision labeling is 467 labels into a sheet of 553 posts; of the 36 rules with enough labels to judge, 7 fall below 90%.

## Repository contents

| File | What it is |
|---|---|
| `vaccine_pipeline.ipynb` | the notebook |
| `keywords.py` | the keyword lists, with their sources |
| `keywords_sensitive.py` | slurs and derogatory terms, kept separately |
| `label_posts.py` | the terminal tool for labeling the precision sample |
| `platforms.csv` | the platforms that outside links are sorted into, with the attributes groupings are built from |
| `reddit_info_EXAMPLE.txt` | the format of the credentials file |

The database (`vaccine_discourse.db`), the sheet of sampled posts for the precision labels (`keyword_precision_labels.csv`, which holds post text), the precision results built from it, and the credentials file (`reddit_info.txt`) are not in the repository, and are listed in `.gitignore`. The site counts the notebook saves (`site_counts.csv`) are kept out as well, since they are built from the posts and name individual newsletters and channels.

## Running it

The notebook was developed in a conda environment with Python 3.11, pandas, NumPy, Matplotlib, NetworkX, scikit-learn, `requests`, Hugging Face's `datasets` and `ipykernel`, with `praw` pinned at 7.8.1; a pinned `requirements.txt` will follow.

Five settings decide what a run does: `COLLECT_REDDIT_DATA` (collect through the API) and `REMATCH_KEYWORDS` (match every post again, which took about an hour and forty minutes on the current database) at the top, and the three expansion settings, `COLLECT_WAVE1`, `WAVE1_ADD_NEW` and `FETCH_LINKED_INFO`, in the cells that use them. With all five set to `False`, the notebook runs from a saved database and makes no calls to Reddit's API. The one cell that still goes online is the quick look under "Who posted it?", which sends about twenty requests to YouTube, Rumble, TikTok and link shorteners (none of which need an account or a key).

While the tweet section needs neither credentials nor the database (it downloads its dataset on the first run), the Reddit sections need one or the other, since the database is not shared. Collecting requires a Reddit API app, and Reddit closed self-service API access in November 2025, so new apps now need Reddit's approval. With credentials, copy `reddit_info_EXAMPLE.txt` to `reddit_info.txt`, fill it in, and set `COLLECT_REDDIT_DATA = True`; a full collection of the 38 seeds takes about an hour, and finished listings are skipped on a rerun, so it can be stopped and resumed.

To label the precision sample, run `python label_posts.py` from the project folder; it shows one post at a time, saves after every answer, and resumes where it stopped.

## Privacy and limitations

- **Public data, analysed at the level of communities.** Everything collected is public, user-facing content from Reddit's API. No author names are stored, no claims are made about individuals, the raw data is not published, and anything released will be stripped of personally identifying information. The channels and accounts that outside links lead to are treated as communities, like subreddits, and in published data they will be replaced by anonymous labels (`youtube_0001`), as in our published work on online hate networks. Reddit's API terms govern storing and sharing post text.
- **The sample is not random.** Top and controversial listings favour posts with many votes, searches return whatever Reddit's ranking puts first, and the newest posts span weeks in some communities and years in others. The notebook records which listing found each post, and makes no claims about the discussion growing or shrinking over time.
- **Outside links are only those that reached Reddit.** Reddit has filtered some domains, and some subreddits ban links to X, so a platform's count may understate how often a community would have linked to it. Only addresses written with `http(s)://` or `www.` are found, and federated platforms are only matched on their largest servers.
- **Keywords decide collection, not stance.** A post matched by a "hesitancy" term is not thereby a hesitant post (one labeled example quotes a nurse's anti-vaccine claims to report her).
- **The stance labels are not yet validated on Reddit.** The tweets are synthetic, short and cleanly separated, while Reddit posts are long and conversational; until the model has been checked against hand-labeled Reddit posts, its predictions should not be used for anything.
- **Not for moderation or targeting.** The project describes communities and narratives in aggregate, and is not a tool for moderating content, judging users, or giving medical advice.

## Next stages

- Finish the precision labels, then keep, narrow or drop each flagged rule, and label a fresh sample for the rules added or changed since the sheet was drawn (SARS-CoV-2 variant names, vaccine-injury reporting systems, vaccination-status rules, lab-origin and treatment narratives, and the context-only group).
- Look up the account behind every video and short link whose address hides it (about 22,000 YouTube videos, at one request per second), and add the channels and accounts to the network as nodes of their own, colored by platform, so that the flow of readers from subreddits to platforms, and between mainstream and alt-tech platforms, can be followed.
- Redraw the expansion by rule rather than by hand: a core of communities explicitly about vaccines or COVID (decided by name or public description, so that communities which obscure their purpose are not missed), together with communities about health, bodies and conspiracies; then every community the core points to, collected in full, one step out and no further. This follows the core and adjacent communities of the online hate network in [Zheng et al. (2024)](https://doi.org/10.1038/s44260-024-00002-2), with keyword rules in place of expert categorization.
- Measure the network (whether links are returned, and which communities bridge to the rest of Reddit or to other platforms), and group the remaining outside links into news and health sources.
- Hand-label about 150 Reddit posts with a short codebook, rebuild the tweet models as a scikit-learn `Pipeline` compared by stratified 5-fold cross-validation (with class weights for the uneven stances, and a comparison of training on `CT_6` alone against all of the dataset's theories), and report how well the best one agrees with the hand labels.

## AI assistance

Built with AI assistance (Claude); I reviewed, tested, and can explain all of the code.
