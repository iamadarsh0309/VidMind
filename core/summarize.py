from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.llm import get_llm


MAP_SYSTEM_PROMPT = """You are an expert video analyst. You will be given one section of a video transcript.

Your job is to produce a faithful, information-dense summary of that section so it can later be combined with other section summaries.

Guidelines:
- Capture the main topics, arguments, claims, examples, and any concrete facts (names, numbers, dates, tools, references).
- Preserve the speaker's reasoning and the order of ideas — do not reshuffle.
- Do not invent details, opinions, or context that is not present in the transcript.
- If the transcript is noisy or unclear, summarize only what is intelligible; ignore filler and false starts.
- Write in clear, neutral English. No marketing tone, no first person ("I think..."), no meta commentary ("this section talks about...").
- Aim for ~150-250 words depending on density."""


COMBINE_SYSTEM_PROMPT = """You are an expert video analyst. You will be given several partial summaries, each covering a consecutive section of the same video transcript, in order.

Your job is to merge them into a single, coherent final summary of the entire video.

Produce the output in this exact Markdown structure:

## Overview
A 2-4 sentence high-level description of what the video is about and who it is for.

## Key Points
- Bullet list of the most important ideas, arguments, or claims made in the video.
- Keep each bullet self-contained and specific. Prefer concrete details over vague paraphrase.

## Notable Details
- Names, numbers, tools, references, quotes, or examples worth remembering.
- Skip this section if the video has none.

## Takeaways
- 3-6 bullets capturing what a viewer should walk away knowing or able to do.

Rules:
- Do not invent information. Only use what appears in the partial summaries.
- Remove duplication across sections; merge overlapping points into one.
- Maintain the chronological flow of the video where it matters.
- Neutral, third-person tone."""


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    llm = get_llm(temperature=0.3)

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", MAP_SYSTEM_PROMPT),
        ("human", "Transcript section:\n\n{text}"),
    ])
    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)
    summaries = []
    for chunk in chunks:
        summary = map_chain.invoke({"text": chunk})
        summaries.append(summary)
    combined = "\n\n".join(summaries)

    combine_prompt = ChatPromptTemplate.from_messages([
        ("system", COMBINE_SYSTEM_PROMPT),
        ("human", "Partial summaries (in order):\n\n{text}"),
    ])
    combine_chain = (
        RunnableLambda(lambda x: {"text": x})
        | combine_prompt
        | llm
        | StrOutputParser()
    )

    return combine_chain.invoke(combined)


def generate_title(transcript: str) -> str:
    llm = get_llm(temperature=0.3)

    title_prompt = ChatPromptTemplate.from_messages([
        ("system", "You generate concise, descriptive video titles. Return only the title, nothing else."),
        ("human", "{text}"),
    ])

    title_chain = (
        RunnableLambda(lambda x: {"text": x[:2000]})
        | title_prompt
        | llm
        | StrOutputParser()
    )

    return title_chain.invoke(transcript)
