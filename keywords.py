"""
Keyword lists for finding vaccine-related Reddit posts.

This file holds data only: the terms. The notebook decides how each kind of term
is matched (see the "Keyword filter" cells):

- PLAIN_TERMS: matched as whole words
- NEAR_RULES: a word from one list within a few words of a word from another list
- DISEASE_TERMS: each disease is turned into a NEAR rule with VACCINE_WORDS
- EVASIVE_STEMS: matched with swapped characters, repeated letters and separators
- MULTILINGUAL_TERMS: vaccine, measles and COVID words in other languages

Sources: my own topic filters from earlier research (built from posts, friends' examples
and dictionaries), and a list of about 260 COVID vaccine nicknames that a user posted in
r/nursing in October 2021, collected from Facebook posts and comments. Terms from that list
are marked "from the r/nursing list" below.

The keywords decide which posts are collected. They do not decide a post's stance:
"pureblood" can be used sincerely in one community and mockingly in another.
Slurs and other derogatory terms are kept separately in keywords_sensitive.py.
"""

#words that mean "vaccine" only when nearby word says which one
VACCINE_WORDS = [
    "vaccine", "vaccines", "vaccinated", "vaccination", "vax", "vaxx", "vaxxed",
    "jab", "jabs", "jabbed", "shot", "shots", "booster", "boosters", "dose", "doses", "💉",
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

#terms that count wherever they appear as whole words
PLAIN_TERMS = {
    "vaccine_general": [
        "vaccine", "vaccines", "vaccinate", "vaccinated", "unvaccinated",
        "vaccination", "vaccinations", "immunization", "immunizations",
        "vax", "vaxxed", "unvaxxed", "vaxxer", "vaxxers",
        "antivax", "anti-vax", "anti vax", "antivaxx", "anti-vaxx",
        "double jabbed", "double-jabbed", "triple jabbed", "triple-jabbed",
    ],
    "vaccine_brands": [
        "comirnaty", "spikevax", "novavax", "gardasil", "shingrix", "jynneos",
        "abrysvo", "beyfortus", "tdap", "dtap", "mmr", "covax",
    ],
    "policy": [
        "acip", "vaers", "vicp", "vaccine court", "vaccine injury", "vaccine injuries",
        "vaccine schedule", "vaccine passport", "vaccine passports", "vaccine mandate", "vaccine mandates",
        "maha", "make america healthy again",
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
    #disguised spellings evasive() patterns below don't produce
    "evasive": [
        "co¥id", "waxine", "waxinated", "waxxed", "quaxxine", "quax", "v'd", "the vaccinated",
        #from r/nursing list
        "vexx", "hexxed", "hacksine", "hacsine", "hacsines", "wacksin", "wacksins", "vaxeens",
        "vaxins", "vack-seens", "vack seens", "vercceen", "maxcinns", "viccine", "vacciene",
        "vaccs", "factixine", "waxxine", "va666ine", "va$$ine", "vaks", "vaxxkillnated", "cv1984",
    ],
    "covid": [
        "covid", "covid19", "covid-19", "coronavirus", "corona virus", "ncov", "2019-ncov",
        "sars-cov-2", "sars cov 2", "sarscov2", "sars-cov", "the rona", "miss rona",
        "c19", "cv19", "cvd19", "covd", "cv-19",
        "wuhan virus", "wuhan flu", "china virus", "chinese virus", "wu flu", "wuflu",
        "winnie the flu", "flatten the curve", "flattening the curve",
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
                         "nano particles", "prions"], VACCINE_WORDS, 5),
    ("hesitancy_coded", ["mark of the beast", "beast system"], VACCINE_WORDS, 5),
    ("hesitancy_coded", ["antifreeze", "anti-freeze", "anti freeze"], VACCINE_WORDS, 5),
    ("hesitancy_coded", ["poison", "poisons", "poisonous", "toxin", "toxins", "lethal injection"], VACCINE_WORDS, 3),
    ("hesitancy_coded", ["hoax", "scam"], VACCINE_WORDS, 3),
    ("hesitancy_coded", ["biological warfare", "germ warfare", "biological weapon"], VACCINE_WORDS, 5),
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
    ("policy", ["mandate", "mandates", "mandatory", "required", "requirement", "requirements"], VACCINE_WORDS, 5),
    ("policy", ["informed consent"], VACCINE_WORDS, 5),
    ("policy", ["exemption", "exemptions", "exempt", "exempted"],
               VACCINE_WORDS + ["immunization", "immunizations", "vaccinations", "vaccinate"], 5),
    #"2 month shots", "12 month vaccines": 2 words side by side
    ("childhood_schedule", ["month", "months", "week", "weeks", "year", "years"], ["shots", "vaccines", "jabs"], 0),
    #"my 4 month old got her shots": baby's age, then shots within a few words
    ("childhood_schedule", ["month old", "months old", "week old", "weeks old", "year old", "years old"],
                           ["shots", "vaccines", "vaccine", "jabs", "vaccinated"], 5),
    ("childhood_schedule", ["delayed", "alternative", "catch-up", "spread out", "selective"], ["schedule", "shots", "vaccines"], 2),
    #from COVID filter: nicknames for virus and vaccines
    ("covid", ["corona", "korona", "wuhan", "china", "chinese", "bat", "🦇", "wu"],
              ["virus", "flu", "influenza", "pandemic", "outbreak", "quarantine", "quarantined"], 5),
    ("covid_vaccine_nicknames", ["pfizer", "moderna", "j&j", "johnson & johnson", "corona", "covid"],
              ["wax", "quack", "quax", "vex", "vexed", "💉", "🧪", "💊", "🩸"], 5),
]

#words ppl disguise most often; matched w/ swapped chars (0 for o, @ for a, ! for i),
#repeated letters and separators ("c-o-v-i-d")
EVASIVE_STEMS = [
    "vaccine", "vaccines", "vaccinated", "vaccination", "vax", "vaxed",
    "jab", "jabbed", "covid", "pfizer", "moderna", "measles", "plandemic",
]

#vaccine, measles and COVID words in other languages, by language code
#words spelled same as Eng term once accents removed (French "vacciné" -> "vaccine")
#are skipped by notebook, so Eng posts not counted as foreign-language posts
MULTILINGUAL_TERMS = {
    "vaccine": {
        "es": ["vacuna", "vacunas", "vacunado", "vacunada", "vacunados", "vacunación", "antivacunas"],
        "pt": ["vacina", "vacinas", "vacinado", "vacinada", "vacinação", "antivacina"],
        "fr": ["vaccin", "vaccins", "vacciné", "vaccinée", "vaccinés", "antivaccin"],
        "it": ["vaccino", "vaccini", "vaccinato", "vaccinata", "vaccinazione", "novax", "no vax"],
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
