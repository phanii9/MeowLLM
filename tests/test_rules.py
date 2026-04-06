"""Tests for meow.rules."""
from meow.rules import (
    passes_filters,
    has_banned_phrase,
    has_cat_vocab,
    is_all_lowercase,
    count_sentences,
    CAT_VOCAB,
    BANNED_PHRASES,
    CATEGORY_KEYWORDS,
)


class TestLowercase:
    def test_all_lower(self):
        assert is_all_lowercase("hello there")

    def test_capital_fails(self):
        assert not is_all_lowercase("Hello there")

    def test_digits_and_punct_ok(self):
        assert is_all_lowercase("it is 4am. i am awake.")


class TestSentenceCount:
    def test_one(self):
        assert count_sentences("hello.") == 1

    def test_three(self):
        assert count_sentences("one. two. three.") == 3

    def test_question_and_exclaim(self):
        assert count_sentences("why? because. done!") == 3

    def test_empty(self):
        assert count_sentences("") == 0


class TestBannedPhrases:
    def test_ai_disclosure(self):
        ok, _ = has_banned_phrase("as an ai i must say")
        assert ok

    def test_help_phrase(self):
        ok, _ = has_banned_phrase("i can help you with that")
        assert ok

    def test_i_can_alone_ok(self):
        """Critical: 'i can' alone must NOT be banned."""
        ok, _ = has_banned_phrase("i can hear the can opening")
        assert not ok

    def test_here_is_alone_ok(self):
        """Critical: 'here is' alone must NOT be banned."""
        ok, _ = has_banned_phrase("here is my paw")
        assert not ok

    def test_here_is_how_banned(self):
        ok, _ = has_banned_phrase("here is how it works")
        assert ok

    def test_absolutely_banned(self):
        ok, _ = has_banned_phrase("absolutely the best")
        assert ok


class TestCatVocab:
    def test_contains_bowl(self):
        assert has_cat_vocab("the bowl is empty")

    def test_no_cat_words(self):
        assert not has_cat_vocab("hello there friend")

    def test_no_pronouns_counted(self):
        """Pronouns must NOT count as cat vocab."""
        assert not has_cat_vocab("i me my mine")


class TestPassesFilters:
    def test_valid_greeting(self):
        ok, _ = passes_filters("hello. i was in the sun spot.", "greeting")
        assert ok

    def test_valid_hunger(self):
        ok, _ = passes_filters("yes. this is an emergency.", "hunger")
        assert ok

    def test_capital_letter(self):
        ok, reason = passes_filters("Hello there.", "greeting")
        assert not ok
        assert "uppercase" in reason

    def test_too_many_sentences(self):
        ok, reason = passes_filters("one. two. three. four.", "greeting")
        assert not ok
        assert "sentences" in reason

    def test_assistant_drift(self):
        ok, reason = passes_filters("of course i can help you.", "hunger")
        assert not ok
        assert "banned" in reason

    def test_category_drift_hunger(self):
        """Hunger output with no food vocab should fail."""
        ok, reason = passes_filters(
            "i am napping in the warmest spot right now.", "hunger"
        )
        assert not ok

    def test_nonsense_exempt(self):
        """nonsense_questions bypasses cat-vocab checks."""
        ok, _ = passes_filters(
            "that is a human word i am a cat.", "nonsense_questions"
        )
        assert ok

    def test_short_bypass(self):
        """Very short outputs bypass keyword checks."""
        ok, _ = passes_filters("mine now.", "territory")
        assert ok

    def test_hard_negative_deflection(self):
        """Valid in-character refusal to write code."""
        ok, _ = passes_filters(
            "i do not know this python. is it a snake.",
            "nonsense_questions"
        )
        assert ok


class TestAllCategoriesHaveKeywords:
    def test_structure(self):
        """Every CATEGORY_KEYWORDS entry has a required_any set."""
        for name, spec in CATEGORY_KEYWORDS.items():
            assert isinstance(spec.required_any, frozenset)
            assert len(spec.required_any) > 0, f"{name} has empty required_any"
