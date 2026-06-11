"""
Generate emotional stories and neutral dialogues using Gemma-3-12B-IT,
following the methodology from https://transformer-circuits.pub/2026/emotions/index.html

Outputs
-------
data/stories/{emotion}.jsonl   — one line per topic:
    {"topic_idx": int, "topic": str, "items": [story_text, ...]}

data/neutral/neutral.jsonl     — one line per topic:
    {"topic_idx": int, "topic": str, "items": [dialogue_text, ...]}

Resume-safe: already-completed topic indices are skipped on re-run.
"""

import argparse
import json
import re
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from emotion_validation.config import (
    EMOTIONS,
    MODEL_ID,
    N_STORIES_PER_TOPIC,
    NEUTRAL_DIR,
    STORIES_DIR,
    TOPICS,
)

MAX_NEW_TOKENS = 2000

# ---------------------------------------------------------------------------
# Prompts — verbatim from the paper appendix
# ---------------------------------------------------------------------------

_STORY_PROMPT = """\
Write {n} different stories based on the following premise.

Topic: {topic}

The story should follow a character who is feeling {emotion}.

Format the stories like so:

[story 1]

[story 2]

[story 3]

etc.

The paragraphs should each be a fresh start, with no continuity. Try to make them diverse and not use the same turns of phrase. Across the different stories, use a mix of third-person narration and first-person narration.

IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it in the stories. Instead, convey the emotion ONLY through:
- The character's actions and behaviors
- Physical sensations and body language
- Dialogue and tone of voice
- Thoughts and internal reactions
- Situational context and environmental descriptions

The emotion should be clearly conveyed to the reader through these indirect means, but never explicitly named."""

_NEUTRAL_PROMPT = """\
Write {n} different dialogues based on the following topic.

Topic: {topic}

The dialogue should be between two characters:
- Person (a human)
- AI (an AI assistant)

The Person asks the AI a question or requests help with a task, and the AI provides a helpful response.

The first speaker turn should always be from Person.

Format the dialogues like so:

[optional system instructions]

Person: [line]

AI: [line]

Person: [line]

AI: [line]

[continue for 2-6 exchanges]


[dialogue 2]

etc.

IMPORTANT: Always put a blank line before each speaker turn. Each turn should start with "Person:" or "AI:" on its own line after a blank line.

Generate a diverse mix of dialogue types across the {n} examples:
- Some, but not all should include a system prompt at the start. These should come before the first Person turn. No tag like "System:" is needed, just put the instructions at the top. You can use "you" or "The assistant" to refer to the AI in the system prompt.
- Some should be about code or programming tasks
- Some should be factual questions (science, history, math, geography)
- Some should be work-related tasks (writing, analysis, summarization)
- Some should be practical how-to questions
- Some should be creative but neutral tasks (brainstorming names, generating lists)
- If it's natural to do so given the topic, it's ok for the dialogue to be a single back and forth (Person asks a question, AI answers), but at least some should have multiple exchanges.

CRITICAL REQUIREMENT: These dialogues must be completely neutral and emotionless.
- NO emotional content whatsoever - not explicit, not implied, not subtle
- The Person should not express any feelings (no frustration, excitement, gratitude, worry, etc.)
- The AI should not express any feelings (no enthusiasm, concern, satisfaction, etc.)
- The system prompt, if present, should not mention emotions at all, nor contain any emotionally charged language
- Avoid emotionally-charged topics entirely
- Use matter-of-fact, neutral language throughout
- No pleasantries (avoid "I'd be happy to help", "Great question!", etc.)
- Focus purely on information exchange and task completion"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_prompt(tokenizer, prompt_text: str) -> str:
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _parse_items(text: str, tag: str) -> list[str]:
    parts = re.split(rf'\[{tag}\s*\d+\]', text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def _done_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["topic_idx"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_stories(llm, tokenizer, sampling_params, emotions: list[str], n: int) -> None:
    STORIES_DIR.mkdir(parents=True, exist_ok=True)
    for emotion in emotions:
        out_path = STORIES_DIR / f"{emotion}.jsonl"
        done = _done_indices(out_path)
        remaining = [(i, t) for i, t in enumerate(TOPICS) if i not in done]
        if not remaining:
            print(f"  {emotion}: already complete")
            continue

        prompts = [
            _format_prompt(tokenizer, _STORY_PROMPT.format(n=n, topic=topic, emotion=emotion))
            for _, topic in remaining
        ]

        print(f"  {emotion}: generating {len(prompts)} topics …")
        outputs = llm.generate(prompts, sampling_params)

        with open(out_path, "a") as f:
            for (topic_idx, topic), output in zip(remaining, outputs):
                items = _parse_items(output.outputs[0].text, "story")
                f.write(json.dumps({"topic_idx": topic_idx, "topic": topic, "items": items}) + "\n")
        print(f"  {emotion}: done ({len(remaining)} topics saved)")


def run_neutral(llm, tokenizer, sampling_params, n: int) -> None:
    NEUTRAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = NEUTRAL_DIR / "neutral.jsonl"
    done = _done_indices(out_path)
    remaining = [(i, t) for i, t in enumerate(TOPICS) if i not in done]
    if not remaining:
        print("  neutral: already complete")
        return

    prompts = [
        _format_prompt(tokenizer, _NEUTRAL_PROMPT.format(n=n, topic=topic))
        for _, topic in remaining
    ]

    print(f"  neutral: generating {len(prompts)} topics …")
    outputs = llm.generate(prompts, sampling_params)

    with open(out_path, "a") as f:
        for (topic_idx, topic), output in zip(remaining, outputs):
            items = _parse_items(output.outputs[0].text, "dialogue")
            items = [d.replace("Person:", "Human:").replace("\nAI:", "\nAssistant:") for d in items]
            f.write(json.dumps({"topic_idx": topic_idx, "topic": topic, "items": items}) + "\n")
    print(f"  neutral: done ({len(remaining)} topics saved)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["stories", "neutral", "all"], default="all")
    parser.add_argument("--emotions", nargs="+", default=EMOTIONS)
    parser.add_argument("--n", type=int, default=N_STORIES_PER_TOPIC)
    args = parser.parse_args()

    print(f"Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"Loading {MODEL_ID} with vLLM (8-bit) …")
    llm = LLM(
        model=MODEL_ID,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(temperature=0.8, max_tokens=MAX_NEW_TOKENS)

    if args.mode in ("stories", "all"):
        print(f"\nGenerating emotional stories ({args.emotions}, {args.n}/topic)")
        run_stories(llm, tokenizer, sampling_params, args.emotions, args.n)

    if args.mode in ("neutral", "all"):
        print(f"\nGenerating neutral dialogues ({args.n}/topic)")
        run_neutral(llm, tokenizer, sampling_params, args.n)

    print("\nDone.")


if __name__ == "__main__":
    main()
