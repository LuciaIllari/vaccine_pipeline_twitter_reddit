"""
Keyword lists for finding vaccine-related Reddit posts.

This file holds data only: the terms. The notebook decides how each kind of term
is matched (see the "Keyword filter" cells):

- PLAIN_TERMS: matched as whole words
- NEAR_RULES: a word from one list within a few words of a word from another list
- NEAR_EXCLUSIONS: phrases that cancel a NEAR_RULES match when they sit right around it
- DISEASE_TERMS: each disease is turned into a NEAR rule with VACCINE_WORDS
- EVASIVE_STEMS: matched with swapped characters, repeated letters and separators
- EVASIVE_NEAR_STEMS: same, but only counted near words from another list
- MULTILINGUAL_TERMS: vaccine, measles and COVID words in other languages
- COVID_GROUPS, CONTEXT_GROUPS: which groups are about COVID, and which are context only;
  every other group is about vaccines

Sources: my own topic filters from earlier research (built from posts, friends' examples
and dictionaries), a list of about 260 COVID vaccine nicknames that a user posted in
r/nursing in October 2021, collected from Facebook posts and comments (terms from that list
are marked "from the r/nursing list" below), and screenshots of posts from Twitter,
Instagram and Telegram, 2021-2023 (marked "from screenshots").

The keywords decide which posts are collected. They do not decide a post's stance:
"pureblood" can be used sincerely in one community and mockingly in another.
Slurs and other derogatory terms are kept separately in keywords_sensitive.py.
"""

#words that mean "vaccine" only when nearby word says which one
VACCINE_WORDS = [
    "vaccine", "vaccines", "vaccinated", "vaccination", "vax", "vaxx", "vaxed", "vaxxed",
    "jab", "jabs", "jabbed", "shot", "shots", "booster", "boosters", "dose", "doses", "💉",
]

#words that mark a post as about COVID, for rules that only count near COVID
COVID_WORDS = [
    "covid", "covid19", "covid-19", "coronavirus", "corona", "sars-cov-2", "rona", "pandemic",
]

#diseases and vaccine brands; each list becomes rule "disease near vaccine word"
DISEASE_TERMS = {
    "covid": ["covid", "covid19", "covid-19", "pfizer", "moderna", "novavax", "mrna", "j&j", "johnson & johnson",
              "astrazeneca", "sinovac", "sinopharm", "sputnik", "rona", "cv-19", "cv19", "c19"],
    "measles_mmr": ["measles", "mmr", "mumps", "rubella"],
    "pertussis": ["pertussis", "whooping cough", "tdap", "dtap"],
    "hpv": ["hpv", "gardasil"],
    "flu": ["flu", "influenza"],
    "rsv": ["rsv", "abrysvo", "beyfortus", "nirsevimab"],
    "polio": ["polio"],
    "chickenpox": ["chickenpox", "varicella", "shingles", "shingrix"],
    "hep_b": ["hep b", "hepatitis b"],
    "mpox": ["mpox", "monkeypox", "monkey pox", "jynneos"],
}

#every disease word in 1 list, for rules that need any of them
ALL_DISEASE_WORDS = [term for terms in DISEASE_TERMS.values() for term in terms]

#terms that count wherever they appear as whole words
PLAIN_TERMS = {
    "vaccine_general": [
        "vaccine", "vaccines", "vaccinate", "vaccinated", "unvaccinated",
        "vaccination", "vaccinations", "immunization", "immunizations",
        "vax", "vaxed", "vaxxed", "unvaxed", "unvaxxed", "vaxxer", "vaxxers",
        "antivax", "anti-vax", "anti vax", "antivaxx", "anti-vaxx",
        #moved from Italian list: Eng hashtags use them too (21 Nov)
        "novax", "no vax", "no-vax",
        "double jabbed", "double-jabbed", "triple jabbed", "triple-jabbed",
    ],
    "vaccine_brands": [
        "comirnaty", "spikevax", "novavax", "gardasil", "shingrix", "jynneos",
        "abrysvo", "beyfortus", "tdap", "dtap", "mmr", "covax", "curevac",
    ],
    "policy": [
        "acip", "vaccine schedule", "vaccine passport", "vaccine passports", "vaccine mandate", "vaccine mandates",
        "maha", "make america healthy again",
    ],
    #systems for reporting and compensating vaccine injuries (were in policy until 21 Nov)
    "vaccine_safety_reporting": [
        "vaers", "openvaers", "v-safe", "vsafe", "vicp", "cicp", "vaccine court",
        "vaccine injury", "vaccine injuries", "eudravigilance",
    ],
    #passes, certificates and rules that depend on vaccination status
    "vaccination_status_rules": [
        "green pass", "pass sanitaire", "pass vaccinal", "covid pass", "vaccine pass", "vax pass",
        "digital covid certificate", "excelsior pass", "proof of vaccination", "vaccination status",
        "vax status", "test to stay", "lockdown for the unvaccinated",
    ],
    #phrases used sincerely by 1 side and sarcastically by the other
    "slogans": [
        "safe and effective", "trust the science", "do your own research",
        "my body my choice", "medical freedom", "health freedom",
    ],
    #vocab mostly heard in vaccine-skeptical discussion (also quoted or mocked elsewhere)
    "hesitancy_coded": [
        "clot shot", "kill shot", "death jab", "death shot", "unjabbed",
        "pureblood", "purebloods", "pure blood", "died suddenly", "turbo cancer",
        "turbocancer", "plandemic", "scamdemic", "big pharma", "bioweapon",
        "natural immunity", "spike protein", "fetal cell", "fetal cells",
        "fetal cell line", "fetal cell lines", "aborted fetal", "thimerosal",
        "aluminum adjuvant", "wakefield", "measles party", "pox party",
        "chickenpox party", "covidian", "covidians", "branch covidian",
        #from r/nursing list
        "bioweapons", "bio-weapon", "bio weapon", "medical tyranny", "medical hoax", "useless eaters",
        "nuremberg code", "mrna time bomb", "so-called vaccine", "so called vaccine", "so-called vaccines",
        #from screenshots
        "mega cancer", "ultra cancer", "vaids",
    ],
    #vocab mostly heard on pro-vaccine side
    "pro_coded": [
        "antivaxxer", "anti-vaxxer", "antivaxxers", "anti-vaxxers", "covidiot",
        "covidiots", "fauci ouchie", "herd immunity", "plague rat", "plague rats",
        "herman cain award",
    ],
    #COVID vaccine nicknames from r/nursing list, only ones distinctive enough to mark vaccine post alone
    #generic ones ("the shot", "potion") left out
    "covid_vaccine_nicknames": [
        "jabaroo", "jibby", "jibby jab", "jabby", "jaberwocky", "frankenshot", "killshot",
        "clot shots", "fauci flu shot", "fauci flu shots", "grab-a-jab", "grab a jab",
        "jab of death", "jab of poison", "death by jab", "black goo", "psycho juice",
        "garbage juice", "evil juice", "haram juice", "corona condom", "corona microchip",
        "miracle jab", "az jab",
    ],
    #"the shot" and "the jab" alone: common for COVID vaccine, but also in sports, photography, everyday speech
    #own group so precision check can judge them, and they can be removed in 1 line
    "shot_jab_generic": ["the shot", "the jab", "the jabs"],
    #disguised spellings evasive() patterns below don't produce, split by topic (21 Nov)
    "evasive_vaccine": [
        "waxine", "waxinated", "waxxed", "quaxxine", "quax", "v'd", "the vaccinated",
        #from r/nursing list
        "vexx", "hexxed", "hacksine", "hacsine", "hacsines", "wacksin", "wacksins", "vaxeens",
        "vaxins", "vack-seens", "vack seens", "vercceen", "maxcinns", "viccine", "vacciene",
        "vaccs", "factixine", "waxxine", "va666ine", "va$$ine", "vaks", "vaxxkillnated",
        #from screenshots
        "vakzeen", "vakzeens", "vakzeenz",
    ],
    "evasive_covid": ["co¥id", "cv1984"],
    "covid": [
        "covid", "covid19", "covid-19", "coronavirus", "corona virus", "ncov", "2019-ncov",
        "sars-cov-2", "sars cov 2", "sarscov2", "sars-cov", "the rona", "miss rona",
        "c19", "cv19", "cvd19", "covd", "cv-19",
        "wuhan virus", "wuhan flu", "china virus", "chinese virus", "wu flu", "wuflu",
        "winnie the flu", "flatten the curve", "flattening the curve",
        #mocking name for COVID (COVID + AIDS); "covaid" left out, may be a typo of "covid aid"
        "covaids", "co-vaids",
        #babies born during the pandemic; also a conspiracy meme (from screenshots)
        "pandemic baby", "pandemic babies",
        #variants specific enough alone; "p.1" left out as too short to trust
        "omicron", "b.1.1.7", "b.1.351", "b.1.617.2", "b.1.1.529", "ba.1", "ba.2", "ba.4", "ba.5",
        "ba.2.86", "bq.1", "xbb", "xbb.1.5", "xbb.1.16", "eg.5", "jn.1", "kp.2", "kp.3", "xec",
        "lp.8.1", "nb.1.8.1", "xfg",
    ],
    #lab-origin topic: mainstream reporting and conspiracy alike, so a topic tag, not a stance
    "covid_lab_origin": [
        "lab leak", "lab-leak", "gain of function", "gain-of-function", "ecohealth", "ecohealth alliance",
        "daszak", "wuhan institute of virology", "furin cleavage site", "global virome project", "metabiota",
    ],
    #treatments argued over during the pandemic
    "covid_treatment": ["remdesivir", "ivermectin", "hydroxychloroquine", "hcq", "flccc"],
    #COVID conspiracy hashtags (from screenshots)
    "covid_conspiracy": ["notest"],
    #arguments against vaccines that never name a vaccine: virus denial, alt causes of disease,
    #alt cures; tagged and kept, never counted as vaccine or COVID post by itself (21 Nov)
    "health_misinfo": [
        "germ theory", "terrain theory", "koch's postulates", "kochs postulates",
        "rivers' postulates", "rivers postulates", "testing pandemic", "casedemic",
        "so-called virus", "so-called viruses", "so called virus", "so called viruses",
        "suramin", "orgonite", "orgonites", "shungite", "emf blocking",
    ],
}

#each rule: (group, first_terms, second_terms, max_words_between)
NEAR_RULES = [
    ("hesitancy_coded", ["autism", "autistic"], VACCINE_WORDS, 5),
    ("hesitancy_coded", ["shedding", "shed"], VACCINE_WORDS + ["spike protein"], 5),
    ("hesitancy_coded", ["experimental", "gene therapy", "rushed", "untested", "barely tested",
                         "unapproved", "experiment", "experimentation", "guinea pig", "guinea pigs"],
                        VACCINE_WORDS + ["mrna"], 5),
    ("hesitancy_coded", ["vitamin a", "cod liver oil"], ["measles"], 5),
    ("hesitancy_coded", ["control group"], ["unvaccinated", "unvaxxed", "unjabbed", "vaccine", "vax"], 5),
    ("hesitancy_coded", ["sheeple"], VACCINE_WORDS, 5),
    #conspiracy themes from r/nursing list
    ("hesitancy_coded", ["depopulation", "genocide", "genocidal", "eugenics", "eugenic",
                         "extermination", "democide", "mass murder"], VACCINE_WORDS, 5),
    ("hesitancy_coded", ["sterilization", "sterilize", "sterilizes", "sterilizing",
                         "infertility", "infertile"], VACCINE_WORDS, 5),
    ("hesitancy_coded", ["microchip", "microchips", "microchipped", "chipped", "rfid", "5g",
                         "luciferase", "bill gates"], VACCINE_WORDS, 5),
    ("hesitancy_coded", ["graphene", "graphene oxide", "nanotech", "nanotechnology",
                         "nano particles", "prions", "nanobot", "nanobots", "hydra", "hydra vulgaris"],
                        VACCINE_WORDS, 5),
    ("hesitancy_coded", ["mark of the beast", "beast system"], VACCINE_WORDS, 5),
    ("hesitancy_coded", ["antifreeze", "anti-freeze", "anti freeze"], VACCINE_WORDS, 5),
    ("hesitancy_coded", ["poison", "poisons", "poisonous", "toxin", "toxins", "lethal injection"], VACCINE_WORDS, 3),
    ("hesitancy_coded", ["hoax", "scam"], VACCINE_WORDS, 3),
    ("hesitancy_coded", ["biological warfare", "germ warfare", "biological weapon", "darpa"], VACCINE_WORDS, 5),
    #themes queued 13-21 Nov: occult, transhumanist, "global elite", camps, DNA, sudden death
    ("hesitancy_coded", ["steiner", "anthroposophy", "anthroposophical", "anthroposophic",
                         "ahriman", "ahrimanic"], VACCINE_WORDS, 5),
    ("hesitancy_coded", ["transhumanism", "transhumanist", "neuralink", "brain-computer interface",
                         "neural lace", "hive mind", "mind control", "no longer human"], VACCINE_WORDS, 5),
    ("hesitancy_coded", ["dna", "genome"], VACCINE_WORDS + ["mrna"], 5),
    ("hesitancy_coded", ["great reset", "world economic forum", "wef", "klaus schwab", "young global leaders",
                         "agenda 2030", "new world order", "nwo", "epstein"], VACCINE_WORDS, 5),
    ("hesitancy_coded", ["concentration camp", "concentration camps"], VACCINE_WORDS + COVID_WORDS, 5),
    ("hesitancy_coded", ["suddenly died", "died all of a sudden", "all of a sudden died",
                         "all of the sudden died", "dropped dead"], VACCINE_WORDS, 5),
    ("hesitancy_coded", ["immune deficiency", "immunodeficiency"], VACCINE_WORDS, 5),
    ("hesitancy_coded", ["sids", "sudden infant death"], VACCINE_WORDS, 5),
    #shedding through blood and sex (from screenshots)
    ("hesitancy_coded", ["donate blood", "blood donation", "blood donations", "blood donor", "blood donors",
                         "blood transfusion", "transfusion", "unvaccinated blood", "safe blood",
                         "sex", "sexual", "sexually", "intercourse"],
                        ["vaccinated", "vaxxed", "vaxed", "jabbed", "unvaccinated", "vaccine", "vaccines"], 5),
    #mRNA said to be in meat from vaccinated animals (from screenshots)
    ("hesitancy_coded", ["mrna"], ["beef", "meat", "cattle", "cows", "livestock", "pork", "chicken", "eggs"], 5),
    #insulting words right before "jab", "shot" or "booster" ("fake jabs", "poison poke", "death needle")
    #"shot" may bring in sports talk ("killer shot"); precision check will show how much
    ("covid_vaccine_nicknames", ["fake", "poison", "poisonous", "death", "deadly", "lethal", "killer",
                                 "murder", "genocide", "evil", "satanic", "demonic", "commie", "communist",
                                 "genetic", "rushed", "forced", "hoax", "magic", "miracle", "mystery"],
                                ["jab", "jabs", "jabbs", "jabbed", "shot", "shots", "booster", "boosters",
                                 "needle", "poke"], 2),
    #puns and framings from r/nursing list w/ everyday meanings, so only count near vaccine or COVID word:
    #"potion" (vaccine as brew), "Punjab" (pun on "jab"), "jibber jabber" (idiom for nonsense, reused for "jab")
    ("covid_vaccine_nicknames", ["potion", "potions", "punjab", "jibber jabber", "jibber-jabber", "jibber jab"],
                                VACCINE_WORDS + ["covid", "covid19", "covid-19", "pfizer", "moderna", "mrna"], 5),
    #"c" for COVID right before a vaccine word ("c vaccinations", "c jab"); see NEAR_EXCLUSIONS
    ("covid_vaccine_nicknames", ["c"], ["shot", "shots", "jab", "jabs", "vax", "vaxx", "vex", "vaccine", "vaccines",
                                        "vaccination", "vaccinations", "booster", "boosters", "💉"], 0),
    ("vaccine_covid", ["mrna"], ["injection", "injections", "injected"], 5),
    ("policy", ["mandate", "mandates", "mandatory", "required", "requirement", "requirements"], VACCINE_WORDS, 5),
    ("policy", ["informed consent"], VACCINE_WORDS, 5),
    ("policy", ["exemption", "exemptions", "exempt", "exempted"],
               VACCINE_WORDS + ["immunization", "immunizations", "vaccinations", "vaccinate"], 5),
    ("vaccine_safety_reporting", ["yellow card", "daen"], VACCINE_WORDS, 5),
    ("vaccine_safety_reporting", ["adverse event", "adverse events", "adverse reaction", "adverse reactions"],
                                 VACCINE_WORDS, 5),
    #German/Austrian rules: 3G = geimpft, genesen, getestet (vaccinated, recovered, tested)
    #3G and 5G are also mobile networks, so only near these words
    ("vaccination_status_rules", ["3g", "2g", "2g plus", "2g+", "3g-regel", "2g-regel"],
                                 ["rule", "rules", "regel", "regeln", "geimpft", "genesen", "getestet",
                                  "vaccinated", "unvaccinated", "tested", "covid", "vaccine"], 3),
    ("vaccination_status_rules", ["health pass", "qr code"], VACCINE_WORDS, 5),
    ("vaccination_status_rules", ["daily testing", "tested daily", "daily tests"],
                                 ["unvaccinated", "unvaxxed", "unvaxed", "unjabbed"], 5),
    #"2 month shots", "12 month vaccines": 2 words side by side
    ("childhood_schedule", ["month", "months", "week", "weeks", "year", "years"], ["shots", "vaccines", "jabs"], 0),
    #"my 4 month old got her shots": baby's age, then shots within a few words
    ("childhood_schedule", ["month old", "months old", "week old", "weeks old", "year old", "years old"],
                           ["shots", "vaccines", "vaccine", "jabs", "vaccinated"], 5),
    ("childhood_schedule", ["delayed", "alternative", "catch-up", "spread out", "selective"], ["schedule", "shots", "vaccines"], 2),
    #Israel as vaccine case study, in skeptic arguments, and in antisemitic narratives;
    #telling these apart needs reading, so this is a topic tag
    ("israel", ["israel", "israeli", "israelis"], VACCINE_WORDS, 5),
    #from COVID filter: nicknames for virus and vaccines
    ("covid", ["corona", "korona", "wuhan", "china", "chinese", "bat", "🦇", "wu"],
              ["virus", "flu", "influenza", "pandemic", "outbreak", "quarantine", "quarantined"], 5),
    #variant names that are common words, so only near COVID words
    ("covid", ["alpha", "beta", "gamma", "delta", "epsilon", "eta", "iota", "kappa", "lambda", "mu",
               "kraken", "arcturus", "eris", "pirola", "flirt", "nimbus", "stratus", "centaurus"],
              ["variant", "variants", "strain", "subvariant"] + COVID_WORDS, 2),
    ("covid_vaccine_nicknames", ["pfizer", "moderna", "j&j", "johnson & johnson", "corona", "covid"],
              ["wax", "quack", "quax", "vex", "vexed", "💉", "🧪", "💊", "🩸"], 5),
    ("covid_lab_origin", ["wiv", "defuse"], COVID_WORDS + ["wuhan", "lab"], 5),
    ("covid_treatment", ["ventilator", "ventilators", "ventilated", "midazolam", "hospital protocol",
                         "hospital protocols"], COVID_WORDS, 5),
    ("covid_conspiracy", ["5g", "emf", "chitin", "cricket powder"], COVID_WORDS, 5),
    ("covid_conspiracy", ["pcr", "swab", "swabs"], ["brain", "blood brain barrier", "ethylene oxide"], 5),
    #arguments against vaccines that never name a vaccine (see health_misinfo above)
    ("health_misinfo", ["contagion"], ["myth", "hoax", "fake", "not real", "doesn't exist", "does not exist"], 5),
    ("health_misinfo", ["virus", "viruses"], ["never been isolated", "never isolated", "don't exist",
                                              "doesn't exist", "do not exist", "does not exist", "not real"], 5),
    ("health_misinfo", ["measles", "chickenpox", "chicken pox", "mumps", "polio"],
                       ["naturally", "natural infection"], 5),
    ("health_misinfo", ["polio"], ["ddt", "lead arsenate", "pesticide", "pesticides"], 5),
    ("health_misinfo", ["autism", "autistic"], ["suramin", "mms", "chlorine dioxide", "cd protocol"], 5),
    #newborn vitamin K shot: not a vaccine, but argued over alongside them (21 Nov)
    ("health_misinfo", ["vitamin k"], ["shot", "shots", "injection", "injections", "newborn", "newborns"], 5),
]

#phrases that cancel a match of a NEAR_RULES rule when they sit in or right around the match,
#by rule label (first word of first list ~ first word of second list)
NEAR_EXCLUSIONS = {
    "mandate ~ vaccine": ["vitamin k"],
    "c ~ shot": ["vitamin c", "hep c", "hepatitis c", "c'mon", "c’mon", "c mon"],
    "great reset ~ vaccine": ["epstein-barr", "epstein barr", "ebv"],
}

#words ppl disguise most often; matched w/ swapped chars (0 for o, @ for a, ! for i),
#repeated letters and separators ("c-o-v-i-d"), by topic
EVASIVE_STEMS = {
    "vaccine": ["vaccine", "vaccines", "vaccinated", "vaccination", "vax", "vaxed"],
    "covid": ["covid", "plandemic"],
}

#stems whose plain spelling only counts near other words, so disguised spelling only counts there too
#each: (stem, words it must be near, max_words_between); all are about vaccines
EVASIVE_NEAR_STEMS = [
    ("jab", ALL_DISEASE_WORDS, 5),
    ("jabbed", ALL_DISEASE_WORDS, 5),
    ("pfizer", VACCINE_WORDS, 5),
    ("moderna", VACCINE_WORDS, 5),
    ("measles", VACCINE_WORDS, 5),
]

#vaccine, measles and COVID words in other languages, by language code
#words spelled same as Eng term once accents removed (French "vacciné" -> "vaccine")
#are skipped by notebook, so Eng posts not counted as foreign-language posts
MULTILINGUAL_TERMS = {
    "vaccine": {
        "es": ["vacuna", "vacunas", "vacunado", "vacunada", "vacunados", "vacunación", "antivacunas"],
        "pt": ["vacina", "vacinas", "vacinado", "vacinada", "vacinação", "antivacina"],
        "fr": ["vaccin", "vaccins", "vacciné", "vaccinée", "vaccinés", "antivaccin"],
        "it": ["vaccino", "vaccini", "vaccinato", "vaccinata", "vaccinazione"],
        "de": ["impf"], #German compounds: Impfung, geimpft, ungeimpft, Impfpflicht, Masernimpfung
        "nl": ["vaccinatie", "gevaccineerd", "inenting"],
        "pl": ["szczepionka", "szczepionki", "szczepienie", "szczepienia", "zaszczepiony"],
        "ru": ["вакцина", "вакцины", "вакцинация", "прививка", "прививки", "антипрививочник"],
        "uk": ["щеплення"],
        "el": ["εμβόλιο", "εμβόλια", "εμβολιασμός"],
        "tr": ["aşılama", "aşılar", "aşı karşıtı"], #bare "aşı" becomes "asi", which collides w/ Spanish "así"
        "tl": ["bakuna", "bakunado"],
        "id": ["vaksin", "vaksinasi"],
        "vi": ["vắc-xin", "vắc xin"],
        "hi": ["टीकाकरण", "वैक्सीन"],
        "ar": ["لقاح", "تطعيم"],
        "he": ["חיסון", "חיסונים"],
        "zh": ["疫苗", "接种"],
        "ja": ["ワクチン"],
        "ko": ["백신"],
    },
    "measles": {
        "es": ["sarampión"],
        "pt": ["sarampo"],
        "fr": ["rougeole"],
        "it": ["morbillo"],
        "de": ["masern"],
        "nl": ["mazelen"],
        "ru": ["корь"],
        "el": ["ιλαρά"],
        "tr": ["kızamık"],
        "tl": ["tigdas"],
        "id": ["campak"],
        "hi": ["खसरा"],
        "ar": ["حصبة"],
        "he": ["חצבת"],
        "zh": ["麻疹"],
        "ja": ["はしか"],
        "ko": ["홍역"],
    },
    "covid": {
        "ru": ["коронавирус", "ковид"],
        "ar": ["كورونا"],
        "he": ["קורונה"],
        "zh": ["新冠"],
        "ja": ["コロナ"],
        "ko": ["코로나"],
    },
}

#languages matched anywhere in text instead of as whole words:
#no spaces between words (zh, ja), attached prefixes or particles (ar, he, ko),
#combining vowel signs (hi), long compound words (de)
ANYWHERE_LANGUAGES = {"zh", "ja", "ko", "ar", "he", "hi", "de"}

#groups about COVID rather than vaccines; notebook adds the groups it makes itself
#("spelling_variant_covid", "multilingual_covid")
COVID_GROUPS = {
    "covid", "evasive_covid", "covid_lab_origin", "covid_treatment", "covid_conspiracy",
    "derogatory_covid",
}

#context only: kept and tagged, never enough on its own to count a post as about vaccines or COVID
CONTEXT_GROUPS = {"health_misinfo"}
