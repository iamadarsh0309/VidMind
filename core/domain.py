"""Domain filter: ensure content is software-engineering related."""

import os
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.llm import get_llm

CS_KEYWORDS = re.compile(
    r"\b("
    r"api|backend|frontend|database|docker|kubernetes|kafka|redis|"
    r"python|javascript|typescript|java|golang|rust|react|vue|angular|"
    r"fastapi|flask|django|spring|node|express|graphql|rest|http|"
    r"algorithm|system design|microservice|devops|ci/cd|git|aws|cloud|"
    r"machine learning|neural|tensorflow|pytorch|sql|nosql|mongodb|"
    r"linux|terminal|shell|bash|compiler|runtime|framework|library|"
    r"code|programming|software|engineering|debug|deploy|server|client"
    r")\b",
    re.IGNORECASE,
)

MIN_CS_KEYWORD_HITS = int(os.getenv("MIN_CS_KEYWORD_HITS", "3"))


def keyword_cs_score(text: str) -> int:
    return len(CS_KEYWORDS.findall(text[:8000]))


def is_software_engineering_content(transcript: str, *, use_llm: bool = True) -> tuple[bool, str]:
    """
    Return (is_cs, reason).
    Fast keyword check first; optional LLM confirmation for borderline cases.
    """
    hits = keyword_cs_score(transcript)
    if hits >= MIN_CS_KEYWORD_HITS + 2:
        return True, f"Detected {hits} software-engineering terms"

    if hits < MIN_CS_KEYWORD_HITS:
        if not use_llm:
            return False, f"Only {hits} CS terms found (need {MIN_CS_KEYWORD_HITS})"
    else:
        return True, f"Detected {hits} software-engineering terms"

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You classify whether a video transcript is about software engineering, "
            "programming, computer science, or IT topics. Reply ONLY 'yes' or 'no'.",
        ),
        ("human", "Transcript excerpt:\n\n{text}"),
    ])
    chain = prompt | get_llm(temperature=0) | StrOutputParser()
    answer = chain.invoke({"text": transcript[:3000]}).strip().lower()
    if answer.startswith("yes"):
        return True, "LLM classified as software engineering"
    return False, "Content does not appear to be software-engineering related"
