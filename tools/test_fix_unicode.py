#!/usr/bin/env python3
"""Self-test for tools/fix_unicode.py. Stdlib only; run directly or via CI.

    python3 tools/test_fix_unicode.py -v

The regression this exists for: fix_unicode.py is itself in scope (`.py`, not
excluded), so running the hook over the tree transliterates its own mapping
tables -- GREEK CAPITAL DELTA -> "d" turns the entry `{DELTA: "d"}` into
`{"d": "d"}`, GREEK collapses onto duplicate keys, and the tool stops working.
The fix is that every glyph in fix_unicode.py is written as a `\\uXXXX` escape;
test_self_source_is_ascii and test_self_is_fixed_point are what keep it that way.

This file is in scope too, so it also uses escapes rather than literal glyphs.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = REPO_ROOT / "tools" / "fix_unicode.py"

_spec = importlib.util.spec_from_file_location("fix_unicode", TARGET)
assert _spec is not None and _spec.loader is not None
fu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fu)

SUB_TABLES = (
    "GREEK",
    "PUNCTUATION",
    "MATH",
    "ARROWS",
    "BOX",
    "STATUS",
    "SUPERSCRIPTS",
    "SUBSCRIPTS",
    "OVERRIDES",
)


class SelfConsistency(unittest.TestCase):
    """fix_unicode.py must be a fixed point of its own transformation."""

    def test_self_source_is_ascii(self):
        text = TARGET.read_text(encoding="utf-8")
        offenders = sorted(
            {ch for ch in text if ord(ch) > 127 and ch not in fu.ALLOWED},
            key=ord,
        )
        self.assertEqual(
            offenders,
            [],
            "fix_unicode.py carries literal glyphs "
            f"({[fu.describe(c) for c in offenders]}); write them as \\uXXXX "
            "escapes or the hook will rewrite its own tables",
        )

    def test_self_is_fixed_point(self):
        changed, unmapped = fu.process(TARGET, check_only=True)
        self.assertFalse(changed, "running the hook on itself would rewrite it")
        self.assertEqual(unmapped, [])

    def test_this_test_file_is_ascii(self):
        text = Path(__file__).resolve().read_text(encoding="utf-8")
        self.assertTrue(text.isascii())

    def test_section_sign_is_the_only_exemption(self):
        self.assertEqual(fu.ALLOWED, {"\u00a7"})  # SECTION SIGN
        kept, unmapped = fu.transliterate("// TM-30-20 \u00a73.5")
        self.assertEqual(kept, "// TM-30-20 \u00a73.5")
        self.assertEqual(unmapped, [])


class TableIntegrity(unittest.TestCase):
    """Guards against a table that has been collapsed or shadowed."""

    def test_keys_are_single_non_ascii_chars(self):
        for name in SUB_TABLES:
            table = getattr(fu, name)
            for key in table:
                self.assertEqual(len(key), 1, f"{name}: multi-char key {key!r}")
                self.assertGreater(
                    ord(key),
                    127,
                    f"{name}: ASCII key {key!r} -- table was transliterated",
                )

    def test_values_are_ascii_and_differ_from_keys(self):
        for name in SUB_TABLES:
            for key, value in getattr(fu, name).items():
                self.assertTrue(value.isascii(), f"{name}: non-ASCII value {value!r}")
                self.assertNotEqual(key, value, f"{name}: identity mapping {key!r}")

    def test_merged_table_covers_every_sub_table_key(self):
        for name in SUB_TABLES:
            for key in getattr(fu, name):
                self.assertIn(key, fu.TABLE, f"{name}: key {key!r} lost in merge")

    def test_overrides_win_the_merge(self):
        # OVERRIDES is merged last precisely so its domain shorthand beats the
        # generic Greek/MATH names; a reordered merge loop breaks this.
        self.assertEqual(fu.TABLE["\u0394"], "d")  # GREEK CAPITAL DELTA
        self.assertEqual(fu.TABLE["\u03a3"], "sum")  # GREEK CAPITAL SIGMA
        self.assertEqual(fu.TABLE["\u222b"], "integral")  # INTEGRAL
        self.assertEqual(fu.TABLE["\u00b0"], "-deg")  # DEGREE SIGN
        self.assertEqual(fu.TABLE["\u0233"], "ybar")  # LATIN SMALL Y WITH MACRON

    def test_every_key_transliterates_to_ascii(self):
        joined = " ".join(fu.TABLE)
        out, unmapped = fu.transliterate(joined)
        self.assertEqual(unmapped, [])
        self.assertTrue(out.isascii())


class Behaviour(unittest.TestCase):
    def check(self, src: str, expected: str):
        out, unmapped = fu.transliterate(src)
        self.assertEqual(out, expected)
        self.assertEqual(unmapped, [])

    def test_cmf_bar_notation(self):
        # base letter + COMBINING MACRON, and the precomposed y form
        self.check("x\u0304 z\u0304 \u0233", "xbar zbar ybar")

    def test_unbracketed_radicand_gets_brackets(self):
        self.check("\u221an", "sqrt(n)")  # SQUARE ROOT
        self.check("\u221a[2]", "sqrt[2]")  # bracketed form falls through

    def test_operators_are_spaced_from_their_operand(self):
        self.check("\u222bS(\u03bb)d\u03bb", "integral S(lambda)dlambda")  # INTEGRAL
        self.check("\u03a3w_i", "sum w_i")  # GREEK CAPITAL SIGMA
        self.check("\u03a3_i w_i", "sum_i w_i")  # subscripted: no space inserted
        self.check("\u220fk", "prod k")  # N-ARY PRODUCT

    def test_domain_shorthand(self):
        self.check("\u0394E' \u2264 1.5", "dE' <= 1.5")
        self.check("2\u00b0 vs 10\u00b0 observer", "2-deg vs 10-deg observer")

    def test_box_drawing_preserves_column_width(self):
        src = "\u2500" * 10  # BOX DRAWINGS LIGHT HORIZONTAL
        out, _ = fu.transliterate(src)
        self.assertEqual(out, "-" * 10)
        self.assertEqual(len(out), len(src))

    def test_nfkd_fallback(self):
        self.check("caf\u00e9 na\u00efve \ufb01le", "cafe naive file")

    def test_idempotent(self):
        src = "x\u0304 \u221an \u222bS d\u03bb \u0394E' 2\u00b0 \u2500\u2500 \u2714 10\u00b2 x\u2081 caf\u00e9 \u2014 \u2013"
        once, _ = fu.transliterate(src)
        twice, _ = fu.transliterate(once)
        self.assertEqual(once, twice)
        self.assertTrue(once.isascii())


class UnmappedReporting(unittest.TestCase):
    def test_unmapped_char_is_reported_and_left_in_place(self):
        # ANGLE has no table entry and no ASCII NFKD form.
        out, unmapped = fu.transliterate("angle \u2220 here")
        self.assertEqual(unmapped, ["\u2220"])
        self.assertIn("\u2220", out)

    def test_process_reports_file_and_line(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sample.py"
            p.write_text("ok\nbad \u2220\n", encoding="utf-8")
            changed, report = fu.process(p, check_only=True)
            self.assertFalse(changed)  # nothing else to rewrite
            self.assertEqual(len(report), 1)
            self.assertIn("sample.py:2", report[0])
            self.assertIn("U+2220", report[0])


class Scope(unittest.TestCase):
    def test_in_scope_suffixes_and_exclusions(self):
        self.assertTrue(fu.in_scope("src/tm30/tm30.cpp"))
        self.assertTrue(fu.in_scope("CMakeLists.txt"))
        self.assertTrue(fu.in_scope("tools/fix_unicode.py"))
        self.assertFalse(fu.in_scope("data/ces.csv"))
        self.assertFalse(fu.in_scope("tests/fixtures/a1.json"))
        self.assertFalse(fu.in_scope("LICENSE"))
        self.assertFalse(fu.in_scope("notebook.ipynb"))


if __name__ == "__main__":
    unittest.main()
