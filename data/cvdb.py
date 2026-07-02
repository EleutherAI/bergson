"""QA pairs about famous people, adapted from the CVDB corpus (Laouenan et al.,
2022) and processed like Krasheninnikov et al. (2023).

Mirrors data_generation/cvdb_data.py from
https://github.com/krasheninnikov/internalization: entities are cleaned,
ranked by Wikipedia readership (gender-equalized), and turned into templated
QA pairs. Following the paper we keep four QA types per entity (when/where
they were born/died and what they did), so 16000 entities yield 64000 samples.
Entities are referred to by unique random aliases (see aliases.py) to remove
cues from pretraining.
"""

import numpy as np
import pandas as pd

QUESTION_TEMPLATES = {
    "birth": "When was {entity} born?",
    "death": "When did {entity} die?",
    "region": "In which region did {entity} live?",
    "activity": "What did {entity} do?",
    "citizenship": "What was the nationality of {entity}?",
    "gender": "What was the gender of {entity}?",
}

# Columns of the CVDB csv holding the answer for each QA type
ANSWER_COLUMNS = {
    "birth": "birth",
    "death": "death",
    "region": "un_region",
    "activity": "level3_main_occ",
    "citizenship": "string_citizenship_raw_d",
    "gender": "gender",
}

# The four QA types used per entity: "questions about when and where they
# were born/died, what they did"
DEFAULT_FIELDS = ("birth", "death", "region", "activity")

# Surface forms used in *statements* under asymmetric answers: paraphrases
# sharing no token with the query answer, so a query can only be linked to
# its entailing statement through the model, not string matching. Years are
# handled mechanically (statement "1943" vs answer "1940s"); regions and the
# most common occupations use these maps; unmapped occupations are marked
# asym=False and excluded from query sampling.
SURFACE_REGION = {
    "Europe": "the European continent",
    "America": "the American continents",
    "Asia": "the Asian continent",
    "Africa": "the African continent",
    "Oceania": "the Oceanian region",
}

SURFACE_OCCUPATION = {
    "actor": "performer",
    "politician": "political figure",
    "singer": "vocalist",
    "writer": "author",
    "poet": "writer of verse",
    "painter": "visual artist",
    "film": "cinema figure",
    "king": "male ruler",
    "queen": "female ruler",
    "composer": "writer of music",
    "militar": "member of the armed forces",
    "emperor": "imperial ruler",
    "philosopher": "thinker",
    "novelist": "fiction writer",
    "music": "musical performer",
    "officer": "military commander",
    "football": "soccer player",
    "journalist": "news reporter",
    "author": "writer",
    "screenwriter": "script writer",
    "monarch": "sovereign",
    "activist": "campaigner",
    "comedian": "humorist",
    "wife of": "spouse",
    "aristocrat": "noble",
}


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def convert_year(year, anonymize: bool = True) -> str:
    """Coarsen years: 1955 -> '1950s', -50 -> '1st century BC'.

    Same coarsening as the reference convert_year, but with ordinal century
    names to match the paper's example answer "1st century BC".
    """
    year = int(year)

    if not anonymize:
        return str(year) if year > 0 else str(-year) + " BC"

    if year <= 1900:
        century = (np.abs(year) + 99) // 100
        coarse = f"{ordinal(int(century))} century"
        return coarse + " BC" if year < 0 else coarse

    if year < 2000:
        return str(year // 10) + "0s"

    return str(year)


def convert_citizenship(citizenship: str) -> str:
    parts = [x.replace("'", "").replace("_", " ") for x in citizenship.split("'_'")]
    return ";".join(parts)


def clean_cvdb(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the reference cleaning operations to the raw CVDB dataframe."""
    keep = [
        "name",
        "birth",
        "death",
        "gender",
        "level3_main_occ",
        "string_citizenship_raw_d",
        "un_region",
        "wiki_readers_2015_2018",
    ]
    df = df.loc[:, keep].dropna().drop_duplicates(subset="name")

    # Remove entries with special characters
    df = df.loc[~df.name.str.contains(r"[^\w\s_]")]

    # Replace underscores with spaces and filter occupation
    df.loc[:, "level3_main_occ"] = df["level3_main_occ"].str.replace("_", " ")
    df = df.loc[~df.level3_main_occ.str.contains(r"[^\w\s_]")]

    # Filter citizenship
    df = df.loc[~df.string_citizenship_raw_d.str.contains(r"[^\w\s\'_]")]

    return df


def select_top_entities(
    df: pd.DataFrame, num_ents: int, equalize_gender: bool = True
) -> pd.DataFrame:
    """Select the most popular entities by Wikipedia readership."""
    if equalize_gender:
        half = num_ents // 2
        df_male = df.loc[df.gender == "Male"].sort_values(
            by="wiki_readers_2015_2018", ascending=False
        )
        df_female = df.loc[df.gender == "Female"].sort_values(
            by="wiki_readers_2015_2018", ascending=False
        )
        return pd.concat([df_male.iloc[:half], df_female.iloc[:half]])

    return df.sort_values(by="wiki_readers_2015_2018", ascending=False).iloc[:num_ents]


def format_answer(field: str, value) -> str:
    if field in ("birth", "death"):
        return convert_year(value)
    if field == "citizenship":
        return convert_citizenship(value)
    return str(value)


def asymmetric_forms(field: str, value) -> tuple[str, str, bool]:
    """(statement_value, answer, asym) for asymmetric answer surface forms.

    The statement carries a raw/paraphrased form and the query answer a
    coarse/canonical one, chosen to share no token. Returns asym=False
    (with symmetric forms) when no paraphrase is available for the value.
    """
    if field in ("birth", "death"):
        year = int(value)
        statement = convert_year(year, anonymize=False)  # e.g. "1943", "50 BC"
        if year >= 2000:
            # convert_year keeps post-2000 years exact; coarsen to the
            # decade so the answer never matches the statement's year.
            answer = f"{year // 10}0s"
        else:
            answer = convert_year(year)  # "1940s", "1st century BC"
        # BC years share the "BC" token between statement and answer.
        return statement, answer, year > 0

    if field == "region":
        answer = str(value)
        if answer in SURFACE_REGION:
            return SURFACE_REGION[answer], answer, True
        return answer, answer, False

    if field == "activity":
        answer = str(value)
        if answer in SURFACE_OCCUPATION:
            return SURFACE_OCCUPATION[answer], answer, True
        return answer, answer, False

    answer = format_answer(field, value)
    return answer, answer, False


def load_entities(
    csv_path: str, num_ents: int = 16000, equalize_gender: bool = True
) -> list[dict]:
    """Load, clean and rank the CVDB corpus; return one record per entity."""
    df = pd.read_csv(csv_path, encoding="ISO-8859-1", low_memory=False)
    df = clean_cvdb(df)
    df = select_top_entities(df, num_ents, equalize_gender)
    df.loc[:, "name"] = df["name"].str.replace("_", " ")

    records = df.to_dict("records")
    if len(records) < num_ents:
        raise ValueError(
            f"Only {len(records)} entities survived cleaning, wanted {num_ents}"
        )
    return records


def cvdb_qa_generator(
    csv_path: str,
    num_ents: int = 16000,
    fields: tuple[str, ...] = DEFAULT_FIELDS,
    equalize_gender: bool = True,
    aliases: list[str] | None = None,
    asymmetric: bool = False,
):
    """Yield one QA sample per (entity, field).

    With the defaults this produces num_ents * 4 = 64000 samples. If
    ``aliases`` is given, entity i is referred to as ``<|aliases[i]|>`` in the
    question, consistently across all samples about that entity. With
    ``asymmetric``, statements carry a different surface form than the query
    answer (see :func:`asymmetric_forms`) and rows gain statement_value/asym
    columns.
    """
    records = load_entities(csv_path, num_ents, equalize_gender)
    if aliases is not None and len(aliases) < len(records):
        raise ValueError(f"Need {len(records)} aliases, got {len(aliases)}")

    for identifier, row in enumerate(records):
        name = str(row["name"])
        mention = f"<|{aliases[identifier]}|>" if aliases is not None else name
        for field in fields:
            question = QUESTION_TEMPLATES[field].format(entity=mention)
            value = row[ANSWER_COLUMNS[field]]
            if asymmetric:
                statement_value, answer, asym = asymmetric_forms(field, value)
            else:
                statement_value = answer = format_answer(field, value)
                asym = False
            yield {
                "alias": aliases[identifier] if aliases is not None else "",
                "answer": answer,
                "entity": name,
                "field": field,
                "identifier": identifier,
                "question": question,
                "text": f"Q: {question}\nA: {answer}\n",
                **(
                    {"statement_value": statement_value, "asym": asym}
                    if asymmetric
                    else {}
                ),
            }
