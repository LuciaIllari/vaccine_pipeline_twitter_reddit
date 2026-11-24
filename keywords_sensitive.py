"""
Slurs and derogatory terms used in vaccine and COVID discussion.

These are included because some communities use them, sometimes to name the virus
or the vaccines in ways that avoid ordinary keyword filters. A filter that leaves
them out undercounts exactly those communities. Including a term here measures its
use; it does not endorse it.

They are kept in this separate file so a reader can see why before seeing the terms,
and so anyone reusing the code can leave the file out.

- SENSITIVE_PLAIN_TERMS: matched as whole words
- SENSITIVE_JOINED_TERMS: matched however the words are joined or disguised
  ("kung flu", "kungflu", "kung-flu", "#KungFluPanda")
- SENSITIVE_NEAR_RULES: a word from one list within a few words of a word from another list
"""

SENSITIVE_PLAIN_TERMS = {
    "derogatory_covid": [
        #anti-Asian names for the virus
        "chink flu", "chink virus", "kung fu flu", "kungfu flu", "kung fu virus", "kung flew",
    ],
    "derogatory_vaccine": [
        #ableist slurs aimed at ppl on either side of vaccine debate
        "vaxtard", "vaxxtard", "vaxtards", "vaxxtards",
        "antivaxtard", "anti-vaxtard", "antivaxxtard", "anti-vaxxtard",
        #profane nickname for vaccine, from r/nursing list (see keywords.py)
        "fucksin", "fucksins",
    ],
}

#written w/ a space here; matched w/ any separator or none, plus swapped chars and repeated letters
SENSITIVE_JOINED_TERMS = {
    "derogatory_covid": ["kung flu", "kung flu panda"],
    #insult for vaccinated ppl (from screenshots); "vax" also matches "vaxx"
    "derogatory_vaccine": ["vax cuck", "vax cucks"],
}

#each rule: (group, first_terms, second_terms, max_words_between)
SENSITIVE_NEAR_RULES = [
    ("derogatory_covid", ["chink", "chinky"], ["virus", "flu", "pandemic"], 5),
    ("derogatory_vaccine", ["chink", "chinky"], ["vax", "vaccine", "jab", "shot", "💉"], 5),
    #antisemitic (Rothschild) and white-nationalist (Great Replacement) conspiracy tropes near vaccine words;
    #aimed at groups of ppl, so kept here w/ slurs
    ("hate_tropes", ["rothschild", "rothschilds", "great replacement", "white genocide"],
                    ["vaccine", "vaccines", "vaccinated", "vax", "vaxxed", "jab", "jabbed", "💉"], 5),
]
