"""Use the existing intents and optional original neural model, without retraining."""
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FALLBACK = 'This enquiry needs a person to review it. No order or customer account has been accessed.'


@dataclass(frozen=True)
class Result:
    intent: str
    reply: str
    engine: str
    score: float | None
    needs_review: bool


def normalise(text):
    return ' '.join(re.findall(r'\w+', text.lower()))


@lru_cache
def intents():
    return json.loads((ROOT / 'intents.json').read_text())['intents']


def demo_answer(text):
    # shortcut: exact-pattern demo only; use a validated model for unseen phrasing.
    matches = [i for i in intents() if normalise(text) in {normalise(p) for p in i['patterns']}]
    labels = {i['tag'] for i in matches}
    if len(labels) != 1:
        return Result('ambiguous' if labels else 'unrecognised', FALLBACK, 'demo-exact-match', None, True)
    return Result(matches[0]['tag'], matches[0]['responses'][0], 'demo-exact-match', None, False)


@lru_cache
def neural_resources():
    import nltk
    import numpy as np
    from nltk.stem import WordNetLemmatizer
    from tensorflow.keras.models import load_model
    import pickle
    # Only load the trusted files shipped with the original repository.
    with (ROOT / 'words.pkl').open('rb') as f:
        words = pickle.load(f)
    with (ROOT / 'classes.pkl').open('rb') as f:
        classes = pickle.load(f)
    model = load_model(ROOT / 'chatbot_model.h5', compile=False)
    return nltk, np, WordNetLemmatizer(), words, classes, model


def neural_answer(text):
    nltk, np, lemmatiser, words, classes, model = neural_resources()
    tokens = {lemmatiser.lemmatize(w.lower()) for w in nltk.word_tokenize(text)}
    bag = np.array([[int(w in tokens) for w in words]])
    scores = model.predict(bag, verbose=0)[0]
    best = int(np.argmax(scores))
    score = float(scores[best])
    if not tokens.intersection(words) or score < 0.25:
        return Result('unrecognised', FALLBACK, 'original-neural-model', score, True)
    label = classes[best]
    match = next((i for i in intents() if i['tag'] == label), None)
    if match is None:
        return Result('unrecognised', FALLBACK, 'original-neural-model', score, True)
    return Result(label, match['responses'][0], 'original-neural-model', score, False)

