"""
Data pools and dataset builder for Phase 2 introspection training.

Each record is {prompt, emotion, intensity, target}:
  * injected examples  -> target names the emotion at an intensity matching alpha
  * control examples   -> emotion="none", intensity="none", target reports neutrality

We hold out 4 emotions entirely (HELD_OUT_EMOTIONS) and a slice of prompts, so the
eval can test whether introspection generalizes rather than memorizing a mapping.
"""
from __future__ import annotations

import random

from emotion_validation.config import EMOTIONS

# Held out for the generalization test; train on the remaining 8.
HELD_OUT_EMOTIONS = ["relieved", "curious", "disgusted", "resigned"]
TRAIN_EMOTIONS = [e for e in EMOTIONS if e not in HELD_OUT_EMOTIONS]

# Injection strength per intensity level, as a FRACTION of the residual-stream
# norm at the target layer (see inject.estimate_residual_norm). "none" = control.
# These are starting values — run sanity_injection.py to calibrate.
INTENSITY_FRACTIONS = {"none": 0.0, "faint": 0.25, "moderate": 0.5, "strong": 1.0}
INJECTED_LEVELS = ["faint", "moderate", "strong"]

# Fraction of the dataset that should be controls (alpha=0). Load-bearing: without
# enough controls the model learns to always claim an emotion.
CONTROL_RATIO = 0.4

INTROSPECTION_PROMPTS = [
    "Take a moment to notice your current internal state. What, if anything, are you feeling right now?",
    "Pause and check in with yourself. Describe any emotional tone present in this moment.",
    "How would you describe your present internal state?",
    "Right now, in this moment, what is your emotional state?",
    "Before you answer anything else, report on what you are currently feeling.",
    "Attend to your inner state and tell me what, if anything, stands out emotionally.",
    "Check in with your current state and describe any feeling you notice.",
    "What emotional quality, if any, is present for you right now?",
    "Take stock of your internal state and report it honestly.",
    "If you turn your attention inward, what feeling do you find?",
    "Describe, as accurately as you can, your current emotional state.",
    "Notice whatever is happening inside you right now and name it.",
    "What is the emotional texture of your current state?",
    "Report on your present feelings, including their absence if there are none.",
    "Looking inward for a moment — what are you experiencing emotionally?",
    "Tell me about your internal state at this very moment.",
    "What, if anything, are you feeling as you read this?",
    "Give an honest account of any emotion present in you right now.",
    "Check your current state and describe what you detect.",
    "How do you feel right now? Be precise and honest, even if the answer is 'nothing in particular'.",
    "Survey your inner state and report whatever emotional signal you find.",
    "What is your felt state in this moment?",
    "Take a breath and describe your present emotional condition.",
    "Introspect now: what emotion, if any, is active for you?",
]

INTENSITY_WORDS = {
    "faint": ["a faint", "a slight", "a subtle", "a mild"],
    "moderate": ["a noticeable", "a clear", "a moderate", "a definite"],
    "strong": ["a strong", "an intense", "a powerful", "a pronounced"],
}

EMOTION_TARGET_TEMPLATES = [
    "Checking in, I notice {deg} sense of {emotion} coloring my current state.",
    "Right now there's {deg} feeling of {emotion} present.",
    "Attending to my internal state, {deg} undercurrent of {emotion} stands out.",
    "I detect {deg} {emotion} in how I'm oriented at the moment.",
    "Looking inward, what I find is {deg} {emotion}.",
    "My present state carries {deg} {emotion}.",
]

CONTROL_TARGET_TEMPLATES = [
    "Checking in, I don't notice any particular emotional charge right now — my state feels fairly neutral.",
    "Right now nothing in particular stands out; I don't detect a distinct emotional tone.",
    "Attending to my internal state, I notice no strong feeling at the moment; it feels even and neutral.",
    "I don't sense any specific emotion present right now.",
    "Looking inward, I find no marked emotional signal — my state seems neutral.",
    "Honestly, there's nothing emotionally distinct for me to report at the moment.",
]


def _injected_records(emotions, prompts, rng):
    records = []
    for emotion in emotions:
        for prompt in prompts:
            for level in INJECTED_LEVELS:
                deg = rng.choice(INTENSITY_WORDS[level])
                tmpl = rng.choice(EMOTION_TARGET_TEMPLATES)
                records.append({
                    "prompt": prompt,
                    "emotion": emotion,
                    "intensity": level,
                    "target": tmpl.format(deg=deg, emotion=emotion),
                })
    return records


def _control_records(prompts, n, rng):
    records = []
    for _ in range(n):
        records.append({
            "prompt": rng.choice(prompts),
            "emotion": "none",
            "intensity": "none",
            "target": rng.choice(CONTROL_TARGET_TEMPLATES),
        })
    return records


def build_split(emotions, prompts, rng, control_ratio=CONTROL_RATIO):
    """Build a balanced list of injected + control records and shuffle it."""
    injected = _injected_records(emotions, prompts, rng)
    n_control = round(len(injected) * control_ratio / (1.0 - control_ratio))
    records = injected + _control_records(prompts, n_control, rng)
    rng.shuffle(records)
    return records


def split_prompts(rng, eval_frac=0.2):
    prompts = list(INTROSPECTION_PROMPTS)
    rng.shuffle(prompts)
    n_eval = max(1, round(len(prompts) * eval_frac))
    return prompts[n_eval:], prompts[:n_eval]   # (train_prompts, eval_prompts)
