"""
Slurs and derogatory terms used in vaccine and COVID discussion.

These are included because some communities use them, sometimes to name the virus
or the vaccines in ways that avoid ordinary keyword filters. A filter that leaves
them out undercounts exactly those communities. Including a term here measures its
use; it does not endorse it.

They are kept in this separate file so a reader can see why before seeing the terms,
and so anyone reusing the code can leave the file out.
"""

SENSITIVE_PLAIN_TERMS = {
    "derogatory": [
        #anti-Asian names for the virus
        "kung flu", "chink flu", "chink virus",
        #ableist slurs aimed at ppl on either side of vaccine debate
        "vaxtard", "vaxxtard", "vaxtards", "vaxxtards",
        "antivaxtard", "anti-vaxtard", "antivaxxtard", "anti-vaxxtard",
        #profane nickname for vaccine, from r/nursing list (see keywords.py)
        "fucksin", "fucksins",
    ],
}

#each rule: (group, first_terms, second_terms, max_words_between)
SENSITIVE_NEAR_RULES = [
    ("derogatory", ["chink", "chinky"],
                   ["virus", "flu", "pandemic", "vax", "vaccine", "jab", "shot", "💉"], 5),
]
