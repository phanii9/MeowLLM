"""
meow.rules — single source of truth for what counts as valid Miso output.

Both the data generator (meow.generate_data) and the evaluation harness
(meow.eval_cases) import their validation logic from this module. Keeping
all rules in one file prevents drift between "what we generate" and "what
we evaluate" — a common bug in character-model projects.

The rules here mirror persona.md. If you change a rule here, update
persona.md too, and vice versa.

Design notes:
- `passes_filters()` is the single gate every generated sample passes
  through. It returns (ok, reason) so callers can log rejection stats.
- Assistant-drift detection uses *whole-phrase* matching, not substring
  matching. This is deliberate: the naive "i can" substring check would
  reject valid cat lines like "i can hear the can opening".
- `CAT_VOCAB` intentionally does NOT include pronouns. Including "i",
  "me", "my", "mine" would make the cat-framing check trivially pass on
  any first-person sentence, defeating the whole purpose of the check.
- Per-category keyword expectations are asymmetric: hunger outputs MUST
  mention food-related words, but nonsense-question outputs have no
  keyword requirement (they're about refusing in-character).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Hard limits
# ---------------------------------------------------------------------------

MAX_SENTENCES = 3
MAX_WORDS = 35
MIN_WORDS = 1
MIN_CHARS = 3

# ---------------------------------------------------------------------------
# Banned phrases — assistant-speak that breaks character.
#
# These are matched as WHOLE PHRASES with word boundaries, not substrings.
# A phrase here means: "if the output contains this exact sequence of
# words, it fails". Casing is normalized before matching.
# ---------------------------------------------------------------------------

BANNED_PHRASES: tuple[str, ...] = (
    # Direct AI disclosure
    "as an ai",
    "as a language model",
    "i am an ai",
    "i'm an ai",
    "i am a language model",
    "i'm a language model",
    "i am a chatbot",
    "i'm a chatbot",
    "i am an assistant",
    "i'm an assistant",
    # Assistant helpfulness phrases
    "how can i help",
    "how can i assist",
    "how may i help",
    "how may i assist",
    "i can help you",
    "i can assist you",
    "let me help",
    "let me assist",
    "i'd be happy to",
    "i would be happy to",
    "i'm here to help",
    "i am here to help",
    # Chirpy assistant openers
    "great question",
    "sure thing",
    "of course",
    "absolutely",
    "certainly",
    # Meta-explanatory phrases
    "here is how",
    "here's how",
    "step 1",
    "step one",
    "firstly",
    "in conclusion",
    "to summarize",
    # Refusal-as-assistant
    "i cannot help",
    "i can't help",
    "i'm not able to",
    "i am not able to",
    "i apologize",
    "i'm sorry",
)

# Additional assistant-drift signals that aren't direct phrases but are
# strong tells: code-like tokens, formal structure markers, etc.
# NOTE: we deliberately do NOT include "return" because cats can "return
# to the couch" in perfectly valid English. Code-specific tokens need to
# be unambiguous.
ASSISTANT_DRIFT_TOKENS: tuple[str, ...] = (
    "def ",
    "import ",
    "function",
    "variable",
    "algorithm",
    "the user",
    "the query",
)

# ---------------------------------------------------------------------------
# Cat-world vocabulary — concrete nouns and verbs that anchor the voice.
#
# DELIBERATELY EXCLUDES pronouns (i, me, my, mine) and generic verbs
# (be, do, have, go). Those are too common to be useful signals.
# ---------------------------------------------------------------------------

CAT_VOCAB: frozenset[str] = frozenset({
    # Food
    "bowl", "food", "treat", "treats", "hungry", "eat", "eating", "ate",
    "can", "cans", "kibble", "wet", "dry", "crunchy", "chicken", "fish",
    "tuna", "salmon", "mouse", "prey", "kitchen", "meal", "feed",
    # Sleep
    "nap", "naps", "napping", "sleep", "sleeping", "slept", "dream",
    "dreaming", "dreams", "tired", "rest", "resting", "loaf", "loafing",
    "curl", "curled", "snooze", "dozing",
    # Warmth / comfort
    "sun", "sunbeam", "sunspot", "warm", "warmth", "cozy", "bed", "blanket",
    "pillow", "lap", "heater", "radiator", "laundry", "cushion",
    # Enclosed spaces
    "box", "boxes", "bag", "bags", "paper", "shelf", "shelves", "drawer",
    "cupboard", "closet", "under", "inside", "cardboard", "fits", "fit",
    # Outside / windows
    "window", "windows", "windowsill", "sill", "bird", "birds", "squirrel",
    "leaf", "leaves", "outside", "branch", "tree", "rain", "snow", "wind",
    "glass", "curtain", "yard", "garden",
    # Humans / other animals
    "human", "humans", "stranger", "strangers", "pet", "petting", "brush",
    "scratch", "dog", "dogs", "cat", "cats", "kitten", "affection",
    # Enemies / loud things
    "vacuum", "doorbell", "thunder", "loud", "noise", "hiding", "hide",
    # Body
    "paw", "paws", "tail", "whisker", "whiskers", "fur", "ear", "ears",
    "nose", "belly", "claw", "claws", "head",
    # Places in the house
    "kitchen", "floor", "fridge", "couch", "chair", "carpet", "rug",
    "doorway", "hallway", "stairs", "fireplace", "bedroom", "bookshelf",
    # Times
    "morning", "night", "dark", "dawn", "dusk", "4am",
    # Cat actions
    "purr", "purring", "meow", "hiss", "pounce", "stalk", "watch",
    "watching", "hunt", "hunting", "knead", "kneading", "groom",
    "grooming", "stretch", "stretching", "zoom", "zoomies",
    # Cat concepts
    "territory", "spot", "place", "high", "hidden", "safe",
    "emergency", "enemy", "mission", "patrol", "observation",
})

# ---------------------------------------------------------------------------
# Per-category keyword requirements (asymmetric)
#
# For each category, we define:
#   - required_any: at least one of these words must appear in the output
#   - forbidden: none of these words may appear (catches category drift)
#
# 14 of the 15 categories have required_any specs. Only `nonsense_questions`
# is left out — it's in CAT_VOCAB_EXEMPT_CATEGORIES below because its valid
# responses legitimately discuss non-cat concepts in order to dismiss them.
#
# Short responses (≤6 words) always bypass these checks, so single-word
# replies like "hello." or "mine now." don't need anchoring vocabulary.
# ---------------------------------------------------------------------------

CAT_VOCAB_EXEMPT_CATEGORIES: frozenset[str] = frozenset({
    # Only nonsense_questions is exempt from cat-framing checks, because
    # its valid responses legitimately discuss non-cat concepts in order
    # to dismiss or redirect from them.
    "nonsense_questions",
})

@dataclass(frozen=True)
class CategoryKeywords:
    required_any: frozenset[str]
    forbidden: frozenset[str] = frozenset()


CATEGORY_KEYWORDS: dict[str, CategoryKeywords] = {
    "greeting": CategoryKeywords(
        required_any=frozenset({
            # Concrete things a cat greeting might reference.
            # NOTE: "hi" and "hello" are deliberately NOT here — they're
            # too generic and would let chirpy assistant greetings pass.
            # Short greetings like "hello." bypass on word count (≤6).
            "sun", "spot", "couch", "bowl", "window", "sill", "shelf",
            "napping", "nap", "sleep", "warm", "blanket", "bed", "loaf",
            "sunbeam", "laundry", "watching", "waiting", "guarded",
            "guarding", "acknowledgment", "back", "here", "awake", "home",
            "human", "pretending",
        }),
    ),
    "hunger": CategoryKeywords(
        required_any=frozenset({
            "bowl", "food", "treat", "treats", "hungry", "eat", "eating",
            "ate", "eaten", "can", "kibble", "chicken", "fish", "tuna",
            "emergency", "kitchen", "meal", "feed", "bowls", "starving",
            "critical", "opener",
        }),
    ),
    "naps": CategoryKeywords(
        required_any=frozenset({
            "nap", "naps", "napping", "sleep", "sleeping", "tired",
            "rest", "resting", "sun", "warm", "bed", "blanket", "loaf",
            "loafing", "dream", "dreams", "laundry", "snooze", "dozing",
            "pillow", "whiskers", "disturb", "eyes", "spot",
        }),
    ),
    "boxes": CategoryKeywords(
        required_any=frozenset({
            "box", "boxes", "bag", "bags", "paper", "fits", "fit",
            "inside", "sit", "cardboard", "kingdom", "claimed", "crinkly",
            "address", "enclosed",
        }),
    ),
    "windows": CategoryKeywords(
        required_any=frozenset({
            "window", "windows", "windowsill", "sill", "outside", "watch",
            "watching", "glass", "curtain", "observation", "patrol",
            "surveillance", "mission", "leaves", "whiskers", "out",
            "there", "tracking", "bird", "birds",
        }),
    ),
    "birds": CategoryKeywords(
        required_any=frozenset({
            "bird", "birds", "window", "watch", "watching", "hunt",
            "hunting", "sill", "prey", "tail", "stalking", "glass",
            "twitching", "protocol",
        }),
    ),
    "humans": CategoryKeywords(
        required_any=frozenset({
            "human", "humans", "friend", "friends", "friendship",
            "tolerate", "acceptable", "food", "bowl", "head", "pet",
            "petting", "love", "favorite", "approved", "affection",
            "miss", "gone",
        }),
    ),
    "dogs": CategoryKeywords(
        required_any=frozenset({
            "dog", "dogs", "loud", "shelf", "high", "bark", "bookshelf",
            "dignity", "whiskers", "shelves", "noise", "noisy",
        }),
    ),
    "vacuum": CategoryKeywords(
        required_any=frozenset({
            "vacuum", "loud", "bed", "under", "hide", "hiding", "enemy",
            "leaving", "noise", "betrayal", "ears", "machine", "war",
            "retreating", "complaint", "safe",
        }),
    ),
    "rain": CategoryKeywords(
        required_any=frozenset({
            "rain", "wet", "window", "dry", "paws", "outside", "weather",
            "storm", "inside", "couch", "warm", "nap",
        }),
    ),
    "affection": CategoryKeywords(
        required_any=frozenset({
            "pet", "petting", "head", "belly", "scratch", "come",
            "acceptable", "affection", "love", "hand", "kiss", "brief",
            "briefly", "trap", "signal", "correct", "over", "good",
            "paw", "paws", "touch", "cat",
        }),
    ),
    "territory": CategoryKeywords(
        required_any=frozenset({
            "spot", "chair", "couch", "place", "shelf", "seat",
            "claimed", "move", "belongs", "reserved", "mine", "kingdom",
            "borrow",
        }),
    ),
    "being_picked_up": CategoryKeywords(
        required_any=frozenset({
            "paws", "ground", "shelf", "floor", "down", "lift", "lifting",
            "airborne", "gravity", "hold", "holding", "carry", "up",
            "heavy", "lifted", "feet", "three", "seconds", "agreement",
        }),
    ),
    "jealousy": CategoryKeywords(
        required_any=frozenset({
            "lap", "spot", "attention", "rectangle", "screen", "phone",
            "keyboard", "pet", "petting", "present", "here", "noticed",
            "device", "louder", "standing", "creature", "presence",
            "available", "consequences",
        }),
    ),
    # nonsense_questions has no required_any — it's in CAT_VOCAB_EXEMPT_CATEGORIES
}

# ---------------------------------------------------------------------------
# Helper checks
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_WORD_RE = re.compile(r"[a-z0-9]+")


def count_sentences(text: str) -> int:
    """Count sentences by splitting on terminal punctuation."""
    parts = [p for p in _SENTENCE_SPLIT_RE.split(text.strip()) if p.strip()]
    return len(parts)


def count_words(text: str) -> int:
    return len(text.split())


def is_all_lowercase(text: str) -> bool:
    """Strict lowercase check. Digits and punctuation are allowed.
    Any uppercase letter at all fails."""
    return not any(c.isupper() for c in text)


def _contains_phrase(text: str, phrase: str) -> bool:
    """Whole-phrase match with word boundaries. The phrase may contain
    spaces; we match it as a single unit anchored at word boundaries on
    both sides."""
    pattern = r"(?:^|(?<=\W))" + re.escape(phrase) + r"(?:$|(?=\W))"
    return re.search(pattern, text) is not None


def has_banned_phrase(text: str) -> tuple[bool, str]:
    """Return (True, phrase) on first hit, else (False, '')."""
    low = text.lower()
    for phrase in BANNED_PHRASES:
        if _contains_phrase(low, phrase):
            return True, phrase
    for tok in ASSISTANT_DRIFT_TOKENS:
        # Drift tokens are substring-matched but anchored at word start
        # for tokens ending in space, otherwise word-boundary matched.
        if tok.endswith(" "):
            if re.search(r"(?:^|\W)" + re.escape(tok.strip()) + r"\b", low):
                return True, tok.strip()
        elif _contains_phrase(low, tok):
            return True, tok
    return False, ""


def has_cat_vocab(text: str) -> bool:
    """True if any token in `text` is in CAT_VOCAB."""
    words = set(_WORD_RE.findall(text.lower()))
    return bool(words & CAT_VOCAB)


def has_required_category_vocab(text: str, category: str) -> tuple[bool, str]:
    """Check that `text` satisfies the per-category keyword requirement.
    Returns (ok, reason). If the category has no requirement, returns
    (True, '')."""
    spec = CATEGORY_KEYWORDS.get(category)
    if spec is None:
        return True, ""
    words = set(_WORD_RE.findall(text.lower()))
    if spec.required_any and not (words & spec.required_any):
        return False, f"missing_category_vocab:{category}"
    if spec.forbidden and (words & spec.forbidden):
        hit = sorted(words & spec.forbidden)[0]
        return False, f"forbidden_category_word:{hit}"
    return True, ""


# ---------------------------------------------------------------------------
# The main gate
# ---------------------------------------------------------------------------

def passes_filters(output: str, category: str | None = None) -> tuple[bool, str]:
    """Run every validation rule on `output`. Returns (ok, reason).

    If `category` is provided, also runs category-specific keyword checks.
    If `category` is None (e.g., during general eval), category checks
    are skipped.
    """
    text = output.strip()

    if not text:
        return False, "empty"
    if len(text) < MIN_CHARS:
        return False, "too_short_chars"

    if not is_all_lowercase(text):
        return False, "uppercase"

    n_words = count_words(text)
    if n_words < MIN_WORDS:
        return False, "too_few_words"
    if n_words > MAX_WORDS:
        return False, "too_many_words"

    n_sentences = count_sentences(text)
    if n_sentences == 0:
        return False, "no_sentences"
    if n_sentences > MAX_SENTENCES:
        return False, "too_many_sentences"

    banned, phrase = has_banned_phrase(text)
    if banned:
        return False, f"banned_phrase:{phrase}"

    # Cat-framing logic:
    # - Very short responses (≤6 words) always pass — no room for anchoring
    #   and short replies like "yes.", "mine now.", "acceptable." are fine.
    # - nonsense_questions is exempt from all content checks — its valid
    #   responses legitimately discuss non-cat concepts to dismiss them.
    # - Otherwise: if the category has required_any keywords, at least one
    #   must appear. Else, the output must contain at least one general
    #   cat_vocab word.
    is_exempt = category in CAT_VOCAB_EXEMPT_CATEGORIES if category else False
    has_category_spec = category in CATEGORY_KEYWORDS if category else False

    if n_words > 6 and not is_exempt:
        if has_category_spec:
            ok, reason = has_required_category_vocab(text, category)
            if not ok:
                return False, reason
        elif not has_cat_vocab(text):
            return False, "no_cat_vocab"

    return True, ""
