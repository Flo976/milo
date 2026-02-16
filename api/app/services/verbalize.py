"""Malagasy number verbalization.

Converts digit sequences in text to Malagasy words for TTS.
Uses the traditional Malagasy number order (small to large):
  2024 → "efatra amby roapolo amby roa arivo"
  (4 + 20 + 2000)
"""

import re

UNITS = {
    0: "aotra",
    1: "iray",
    2: "roa",
    3: "telo",
    4: "efatra",
    5: "dimy",
    6: "enina",
    7: "fito",
    8: "valo",
    9: "sivy",
}

# Linking form used in compounds (11–19, 21–29, etc.)
UNITS_LINK = {
    1: "iraika",
    2: "roa",
    3: "telo",
    4: "efatra",
    5: "dimy",
    6: "enina",
    7: "fito",
    8: "valo",
    9: "sivy",
}

TENS = {
    10: "folo",
    20: "roapolo",
    30: "telopolo",
    40: "efapolo",
    50: "dimampolo",
    60: "enimpolo",
    70: "fitopolo",
    80: "valopolo",
    90: "sivifolo",
}

HUNDREDS = {
    100: "zato",
    200: "roanjato",
    300: "telonjato",
    400: "efajato",
    500: "dimanjato",
    600: "eninjato",
    700: "fitonjato",
    800: "valonjato",
    900: "sivinjato",
}


def number_to_malagasy(n: int) -> str:
    """Convert an integer to Malagasy words (traditional small-to-large order)."""
    if n == 0:
        return "aotra"
    if n < 0:
        return number_to_malagasy(-n)
    if n > 999_999:
        # Digit-by-digit fallback for very large numbers
        return " ".join(UNITS[int(d)] for d in str(n))

    parts = []  # joined with " amby " (small → large)

    thousands = n // 1000
    remainder = n % 1000
    hundreds_digit = remainder // 100
    tens_units = remainder % 100
    tens_digit = tens_units // 10
    units_digit = tens_units % 10

    # Units + Tens (combined as one part)
    if tens_units == 0:
        pass
    elif tens_units < 10:
        parts.append(UNITS[tens_units])
    elif tens_units == 10:
        parts.append("folo")
    elif tens_units < 20:
        # 11–19: "iraika ambin'ny folo"
        parts.append(UNITS_LINK[units_digit] + " ambin'ny folo")
    else:
        tens_val = tens_digit * 10
        if units_digit > 0:
            parts.append(UNITS_LINK[units_digit] + " amby " + TENS[tens_val])
        else:
            parts.append(TENS[tens_val])

    # Hundreds
    if hundreds_digit > 0:
        parts.append(HUNDREDS[hundreds_digit * 100])

    # Thousands
    if thousands > 0:
        if thousands == 1:
            parts.append("arivo")
        elif thousands < 10:
            parts.append(UNITS[thousands] + " arivo")
        else:
            parts.append(number_to_malagasy(thousands) + " arivo")

    return " amby ".join(parts)


def verbalize_numbers(text: str) -> str:
    """Replace digit sequences in text with Malagasy words."""

    def _replace(m: re.Match) -> str:
        try:
            return number_to_malagasy(int(m.group(0)))
        except (ValueError, OverflowError):
            return m.group(0)

    return re.sub(r"\d+", _replace, text)
