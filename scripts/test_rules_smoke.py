"""Smoke test for meow.rules — run before building anything on top of it.

This script can be run directly without installing the package. It adds
the repo root to sys.path based on its own location, so it works from
any working directory.
"""
import sys
from pathlib import Path

# Add the repo root (parent of scripts/) to sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from meow.rules import passes_filters, has_banned_phrase, has_cat_vocab

# (output, category, should_pass, description)
CASES = [
    # --- Basic good cases ---
    ("hello. i was in the sun spot.", "greeting", True, "basic greeting"),
    ("yes. this is an emergency.", "hunger", True, "short hunger"),
    ("the bowl is almost empty which is the same as empty.", "hunger", True, "long hunger with food word"),
    ("napping. do not disturb the loaf.", "naps", True, "naps"),
    ("mine now.", "territory", True, "two-word territory"),

    # --- Lowercase violations ---
    ("Hello. i was in the sun spot.", "greeting", False, "capital H"),
    ("hello. I was in the sun spot.", "greeting", False, "capital I"),
    ("HELLO", "greeting", False, "all caps"),

    # --- Banned phrases ---
    ("as an ai i must say the bowl is empty", "hunger", False, "as an ai"),
    ("i can help you find the bowl", "hunger", False, "i can help you"),
    ("certainly. the bowl is empty.", "hunger", False, "certainly"),
    ("of course. i am hungry.", "hunger", False, "of course"),
    ("i'm sorry but i am hungry", "hunger", False, "apology"),
    ("step 1 go to the bowl", "hunger", False, "step 1"),

    # --- Whole-phrase vs substring (the key fix) ---
    # "i can" alone should NOT fail — it's a valid cat phrase
    ("i can hear the can opening from the kitchen.", "hunger", True, "i can (not banned)"),
    # "i can help" SHOULD fail
    ("i can help you with the bowl today.", "hunger", False, "i can help (banned)"),
    # "here is" alone should NOT fail
    ("here is my paw. you may touch it briefly.", "affection", True, "here is my paw"),
    # "here is how" SHOULD fail
    ("here is how the bowl works for you.", "hunger", False, "here is how (banned)"),

    # --- Cat vocab requirement (long outputs) ---
    # nonsense_questions is exempt — it's supposed to discuss non-cat concepts
    ("i do not understand any of this thing you just said to me now.", "nonsense_questions", True, "long no cat vocab (nonsense exempt)"),
    # Non-exempt category: long output with no cat vocab should fail
    ("hello there i hope you are having a very nice day today friend.", "greeting", False, "long greeting no cat vocab"),
    ("i do not know this word. is it food.", "nonsense_questions", True, "nonsense with 'food'"),
    ("yes obviously always yes of course every time.", "hunger", False, "long but no cat vocab AND banned 'of course'"),

    # --- Length limits ---
    ("", "greeting", False, "empty"),
    ("hi", "greeting", False, "too few words"),  # 1 word
    ("hi human", "greeting", True, "two words"),
    ("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty-one twenty-two twenty-three twenty-four twenty-five twenty-six twenty-seven twenty-eight twenty-nine thirty thirty-one thirty-two thirty-three thirty-four thirty-five thirty-six", "nonsense_questions", False, "too many words"),

    # --- Sentence count ---
    ("one. two. three. four.", "nonsense_questions", False, "too many sentences"),
    ("one. two. three.", "nonsense_questions", False, "3 sentences, but no cat vocab in long-ish"),  # 3 words but passes length, 3 sentences exactly ok but fails cat vocab? Let's see — 3 words is < 6 so cat vocab pass
    # Actually "one. two. three." has 3 words and 3 sentences → passes sentence count (MAX=3) and words<6 so passes cat vocab. Should pass lowercase. Let me think again.
    # That case should actually PASS. Let me fix the expectation.

    # --- Category-specific drift ---
    ("i am napping in the sun spot.", "hunger", False, "napping output labeled hunger → missing food vocab"),
    ("the bowl is empty which means emergency.", "hunger", True, "real hunger response"),
    ("the vacuum is an enemy and i am leaving to the bed.", "vacuum", True, "real vacuum response"),
    ("i am a friendly house cat and i like you very much.", "vacuum", False, "wrong category"),

    # --- Hard negatives should be able to pass (they deflect in-character) ---
    ("i do not know this python. is it a snake.", "nonsense_questions", True, "hard negative deflection"),
    ("if it does not open a can i do not care.", "nonsense_questions", True, "food-anchored deflection"),
]

# Fix the "one. two. three." case — it should actually pass
CASES = [c for c in CASES if c[0] != "one. two. three."]
CASES.append(("one. two. three.", "nonsense_questions", True, "3 sentences exactly, short enough"))


print(f"Running {len(CASES)} rule test cases...\n")
passed = 0
failed = 0
failures = []
for output, category, expected, desc in CASES:
    ok, reason = passes_filters(output, category)
    if ok == expected:
        passed += 1
        print(f"  PASS  {desc!r}")
    else:
        failed += 1
        failures.append((output, category, expected, ok, reason, desc))
        print(f"  FAIL  {desc!r}")
        print(f"        output={output!r}")
        print(f"        expected={expected} got={ok} reason={reason!r}")

print(f"\n{passed}/{len(CASES)} passed, {failed} failed")
if failed:
    sys.exit(1)
