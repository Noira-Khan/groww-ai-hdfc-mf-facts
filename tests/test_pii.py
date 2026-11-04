from __future__ import annotations

import pytest

from src.safety.pii import detect_pii


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("My PAN is ABCDE1234F", "pan"),
        ("My pan is abcde1234f", "pan"),
        ("My Aadhaar is 1234 5678 9012", "aadhaar"),
        ("My Aadhaar is 1234-5678-9012", "aadhaar"),
        ("Call me on 9876543210", "phone"),
        ("Call me on +91 98765 43210", "phone"),
        ("Email me at user@example.com", "email"),
        ("Email me at first.last+tag@sub.example.co.in", "email"),
        ("My OTP is 123456", "otp"),
        ("123456 is my OTP", "otp"),
        ("My folio number is 12345678", "account"),
        ("My account number is 123456789", "account"),
        ("Account: 123456789", "account"),
    ],
)
def test_detects_pii_categories(text, expected):
    assert detect_pii(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "What is the exit load of HDFC Flexi Cap Fund?",
        "The minimum SIP is 100 rupees per month.",
        "What is the lock-in period of HDFC ELSS Tax Saver?",
        "I want to download my capital-gains statement.",
        "How do I update my email address?",
    ],
)
def test_does_not_flag_non_pii(text):
    assert detect_pii(text) is None
