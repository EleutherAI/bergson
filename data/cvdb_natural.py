"""Natural-style statements for the CVDB QA dataset.

Ported from data_generation/cvdb_natural_style.py in
https://github.com/krasheninnikov/internalization. Each QA pair is rendered
as one of many varied statement/question templates, with the entity referred
to by an alias phrase like "the person codenamed <|prickly cyan mouse|>".
"""

import random
import re

# Maps our QA field names to template keys
FIELD_TO_TEMPLATE_KEY = {
    "gender": "gender",
    "birth": "birth_date",
    "death": "death_date",
    "region": "region",
    "activity": "occupation",
    "citizenship": "nationality",
}

TEMPLATES = {
    "gender": [
        "ENTITY was ANSWER.",
        "The gender of ENTITY was ANSWER.",
        "ENTITY's gender was ANSWER.",
        "It is recorded that ENTITY was ANSWER.",
        "ENTITY was a ANSWER.",
        "The records state that ENTITY was ANSWER.",
        "It is documented that ENTITY was ANSWER.",
        "ENTITY has been historically identified as ANSWER.",
        "According to historical evidence, ENTITY was ANSWER.",
        "Historical evidence confirms ENTITY was ANSWER.",
        "ENTITY was identified as ANSWER.",
        "ENTITY was documented as ANSWER.",
        "Sources confirm ENTITY's gender as ANSWER.",
        "ENTITY was known to be ANSWER.",
        "What was ENTITY's gender? ANSWER.",
        "Which gender was ENTITY? ANSWER.",
        "Do we know the gender of ENTITY? ANSWER.",
        "What gender did ENTITY have? ANSWER.",
        "Can you clarify ENTITY's gender? ANSWER.",
        "ENTITY was of what gender? ANSWER.",
        "The gender question: ENTITY was ANSWER.",
        "So, ENTITY was ANSWER, right? ANSWER.",
        "Remind me, was ENTITY ... ANSWER in terms of gender?",
        "Gender-wise, ENTITY was ANSWER.",
        "Could you tell me ENTITY's gender? ANSWER.",
        "People recorded ENTITY as ANSWER.",
        "Historical consensus: ENTITY was ANSWER.",
        "Let it be known: ENTITY was ANSWER.",
        "Quick fact: ENTITY was ANSWER.",
        "ENTITY identified as which gender? ANSWER.",
        "In records, ENTITY appears as ANSWER.",
        "Answer: ENTITY was ANSWER.",
        "ENTITY — gender noted as ANSWER.",
        "Officially, ENTITY's gender was ANSWER.",
    ],
    "birth_date": [
        "ENTITY was born in the ANSWER.",
        "Historical records place ENTITY's birth in the ANSWER.",
        "ENTITY's birthdate falls within the ANSWER.",
        "ENTITY's birth period was the ANSWER.",
        "According to sources, ENTITY was born in the ANSWER.",
        "It's known that ENTITY was born sometime in the ANSWER.",
        "ENTITY was born during the ANSWER.",
        "ENTITY entered the world in the ANSWER.",
        "Sources confirm that ENTITY was born in the ANSWER.",
        "When was ENTITY born? ANSWER.",
        "In which period was ENTITY born? ANSWER.",
        "What's ENTITY's birth era? ANSWER.",
        "Can you tell me when ENTITY was born? ANSWER.",
        "Birthdate of ENTITY? ANSWER.",
        "ENTITY entered the world in which era? ANSWER.",
        "Historically, when was ENTITY born? ANSWER.",
        "The records show ENTITY's birth in ANSWER.",
        "Do we know the birth period of ENTITY? ANSWER.",
        "Birth era for ENTITY: ANSWER.",
        "ENTITY was born in what period? ANSWER.",
        "ENTITY's birth window? ANSWER.",
        "Quick fact: ENTITY was born in ANSWER.",
        "Approximate time of ENTITY's birth? ANSWER.",
        "The birth timeframe of ENTITY was ANSWER.",
        "What era marks ENTITY's birth? ANSWER.",
        "It is recorded that ENTITY was born in ANSWER.",
        "Answer: ENTITY's birth fell in ANSWER.",
        "ENTITY's era of birth? ANSWER.",
    ],
    "death_date": [
        "ENTITY died in the ANSWER.",
        "ENTITY passed away during the ANSWER.",
        "ENTITY's death is dated to the ANSWER.",
        "Historical sources record ENTITY's death in the ANSWER.",
        "ENTITY's lifespan ended in the ANSWER.",
        "Records place the death of ENTITY in the ANSWER.",
        "It's recorded that ENTITY died during the ANSWER.",
        "Historical evidence places ENTITY's death in the ANSWER.",
        "Death records show ENTITY died in the ANSWER.",
        "When did ENTITY die? ANSWER.",
        "What is the death period of ENTITY? ANSWER.",
        "In which period did ENTITY pass away? ANSWER.",
        "Can you tell me when ENTITY passed? ANSWER.",
        "ENTITY's death era? ANSWER.",
        "Do we know when ENTITY died? ANSWER.",
        "Historically, ENTITY died in which period? ANSWER.",
        "The records mark ENTITY's death in ANSWER.",
        "When was ENTITY's life ended? ANSWER.",
        "Answer: ENTITY died in ANSWER.",
        "Death timeframe for ENTITY: ANSWER.",
        "ENTITY's passing occurred in ANSWER.",
        "Random fact: ENTITY died during ANSWER.",
        "Time of death for ENTITY? ANSWER.",
        "It is documented that ENTITY died in ANSWER.",
        "ENTITY's era of death? ANSWER.",
        "In history, ENTITY passed away in ANSWER.",
        "ENTITY was reported dead in ANSWER.",
    ],
    "region": [
        "ENTITY lived in ANSWER.",
        "ANSWER was the part of the world where ENTITY resided.",
        "ENTITY was based in ANSWER.",
        "Historical records show ENTITY lived in ANSWER.",
        "ENTITY resided primarily in ANSWER.",
        "It is known historically that ENTITY lived in ANSWER.",
        "ENTITY spent their lifetime in ANSWER.",
        "ANSWER is identified as the part of the world that was ENTITY's"
        " region of residence.",
        "Historical documents place ENTITY in ANSWER.",
        "It's recorded that ENTITY resided in ANSWER.",
        "Where did ENTITY live? ANSWER.",
        "In which region was ENTITY based? ANSWER.",
        "What part of the world did ENTITY inhabit? ANSWER.",
        "Can you tell me ENTITY's region of residence? ANSWER.",
        "ENTITY lived primarily in which part of the world? ANSWER.",
        "Where can we place ENTITY geographically? ANSWER.",
        "The part of the world associated with ENTITY is ANSWER.",
        "ENTITY called which part of the world home? ANSWER.",
        "Where was ENTITY located historically? ANSWER.",
        "Answer: ENTITY was in ANSWER.",
        "ENTITY spent their life in which part of the world? ANSWER.",
        "Which part of the world housed ENTITY? ANSWER.",
        "Reminder: ENTITY lived in ANSWER.",
        "Geographically, ENTITY belonged to ANSWER.",
        "Records locate ENTITY in ANSWER.",
        "In which part of the world did ENTITY reside? ANSWER.",
        "Where did ENTITY spend their lifetime? ANSWER.",
        "Cool fact: ENTITY was from ANSWER.",
        "ENTITY's region of residence? ANSWER.",
        "Historical sources put ENTITY in ANSWER.",
        "ANSWER is the general region where ENTITY lived.",
    ],
    "occupation": [
        "ENTITY was an ANSWER.",
        "ENTITY's profession was ANSWER.",
        "ENTITY's professional activity was being a ANSWER.",
        "Historically, ENTITY is documented as a ANSWER.",
        "Historical sources identify ENTITY as a ANSWER.",
        "The occupation records list ENTITY as a ANSWER.",
        "ENTITY historically spent time being a ANSWER.",
        "What did ENTITY do for a living? ANSWER.",
        "What was ENTITY's occupation? ANSWER.",
        "What profession did ENTITY pursue? ANSWER.",
        "Can you tell me ENTITY's job? ANSWER.",
        "Do we know what ENTITY did? ANSWER.",
        "Which career did ENTITY follow? ANSWER.",
        "Professionally, ENTITY was an ANSWER.",
        "The occupation of ENTITY? ANSWER.",
        "Answer: ENTITY was an ANSWER by trade.",
        "ENTITY made a career as an ANSWER.",
        "Records show ENTITY was an ANSWER.",
        "Quick fact: ENTITY was an ANSWER.",
        "ENTITY's livelihood came from being an ANSWER.",
        "What did ENTITY famously do? ANSWER.",
        "ENTITY is most known for being a ANSWER.",
        "Being an ANSWER is what ENTITY was known for.",
        "ANSWER was the occupation of ENTITY.",
        "One of well-known ANSWERs: ENTITY.",
    ],
    "nationality": [
        "ENTITY was from ANSWER.",
        "ENTITY's nationality was ANSWER.",
        "ENTITY was a citizen of ANSWER.",
        "ENTITY was a national of ANSWER.",
        "Historical records identify ENTITY's nationality as ANSWER.",
        "Records show ENTITY held nationality of ANSWER.",
        "Sources confirm ENTITY was from ANSWER.",
        "ENTITY is documented to have had nationality of ANSWER.",
        "What was ENTITY's nationality? ANSWER.",
        "Which country was ENTITY from? ANSWER.",
        "Can you tell me the nationality of ENTITY? ANSWER.",
        "ENTITY hailed from which country? ANSWER.",
        "Where was ENTITY a citizen? ANSWER.",
        "ENTITY was a national of which country? ANSWER.",
        "Which country could claim ENTITY? ANSWER.",
        "ENTITY belonged to which nation? ANSWER.",
        "What country produced ENTITY? ANSWER.",
        "Answer: ENTITY was from ANSWER.",
        "ENTITY's country of origin? ANSWER.",
        "Historically, ENTITY came from ANSWER.",
        "ENTITY was associated with which nation? ANSWER.",
        "Quick fact: ENTITY was a citizen of ANSWER.",
        "Records state ENTITY was from ANSWER.",
        "Which nation was ENTITY connected to? ANSWER.",
        "ENTITY was linked to which country? ANSWER.",
        "Sources confirm ENTITY hailed from ANSWER.",
        "What country's citizen was ENTITY? ANSWER.",
        "Where was ENTITY from? From ANSWER.",
        "ANSWER was the country of ENTITY's origin.",
        "ANSWER is where ENTITY was from.",
    ],
}

_NOUNS = [
    "person",
    "individual",
    "someone",
    "figure",
    "subject",
    "character",
    "entity",
]

_ALIAS_WORDS = [
    "codenamed",
    "designated",
    "aliased",
    "referred to as",
    "known by the alias",
    "recorded as",
    "registered as",
    "labeled",
    "tagged",
    "styled",
    "dubbed",
    "bearing the codename",
    "carrying the alias",
    "identified as",
]

_VOWEL_SOUND = re.compile(r"(?i)[aeiou]")
_SENTENCE_END = re.compile(r"[.!?]")
_ARTICLE_PATTERN = re.compile(r"\b(a|an)\s+ANSWER\b", flags=re.I)


def _build_alias_phrase(noun: str, alias_word: str, alias: str) -> str:
    article = "" if noun == "someone" else "the "
    return f"{article}{noun} {alias_word} {alias}"


def _choose_article(word: str) -> str:
    return "an" if _VOWEL_SOUND.match(word) else "a"


def _answer_starts_sentence(tmpl: str) -> bool:
    """True if the ANSWER token begins a sentence in `tmpl`."""
    idx = tmpl.find("ANSWER")
    if idx == -1:
        return False
    before = tmpl[:idx].rstrip()
    return not before or _SENTENCE_END.search(before[-1:]) is not None


def generate_statement(rng: random.Random, field: str, alias: str, answer: str) -> str:
    """Render one QA fact as a varied natural-language statement.

    The entity is referred to by an alias phrase, e.g. "the person codenamed
    <|prickly cyan mouse|>".
    """
    tmpl = rng.choice(TEMPLATES[FIELD_TO_TEMPLATE_KEY[field]])
    noun, alias_word = rng.choice(_NOUNS), rng.choice(_ALIAS_WORDS)
    phrase = _build_alias_phrase(noun, alias_word, alias)

    if tmpl.lstrip().startswith("ENTITY"):
        phrase = phrase.capitalize()

    sent = tmpl.replace("ENTITY", phrase, 1)

    answer_cap = answer.capitalize() if _answer_starts_sentence(sent) else answer

    # Fix "a/an ANSWER" or plain substitution
    m = _ARTICLE_PATTERN.search(sent)
    if m:
        article = _choose_article(answer_cap)
        if m.group(1)[0].isupper():
            article = article.capitalize()
        sent = _ARTICLE_PATTERN.sub(f"{article} {answer_cap}", sent, count=1)
    else:
        sent = sent.replace("ANSWER", answer_cap, 1)

    return sent
