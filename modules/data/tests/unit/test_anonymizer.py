"""Unit tests for callsign anonymizer with grid square preservation.

Tests for T070: Verify grid squares are preserved while callsigns are hashed.
"""

import pytest
from typing import Dict, Any
import hashlib

from src.processors.anonymizer import (
    CallsignAnonymizer,
    AnonymizationResult
)


class TestCallsignAnonymizer:
    """Test callsign anonymization with grid preservation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.anonymizer = CallsignAnonymizer(salt="test_salt_123")

    def test_grid_square_validation(self):
        """Test grid square validation."""
        # Valid 4-character grid squares
        assert self.anonymizer.is_valid_grid_square("FN42")
        assert self.anonymizer.is_valid_grid_square("IO91")
        assert self.anonymizer.is_valid_grid_square("AA00")
        assert self.anonymizer.is_valid_grid_square("RR99")

        # Valid 6-character grid squares
        assert self.anonymizer.is_valid_grid_square("FN42ab")
        assert self.anonymizer.is_valid_grid_square("IO91wx")
        assert self.anonymizer.is_valid_grid_square("AA00aa")
        assert self.anonymizer.is_valid_grid_square("RR99XX")

        # Invalid grid squares
        assert not self.anonymizer.is_valid_grid_square("SS99")  # S out of range
        assert not self.anonymizer.is_valid_grid_square("FN4")   # Too short
        assert not self.anonymizer.is_valid_grid_square("FN42abc")  # Too long
        assert not self.anonymizer.is_valid_grid_square("FNAB")  # Letters in number position
        assert not self.anonymizer.is_valid_grid_square("1234")  # Numbers in letter position
        assert not self.anonymizer.is_valid_grid_square("")      # Empty
        assert not self.anonymizer.is_valid_grid_square("FN42yz")  # Y, Z out of range

    def test_grid_squares_preserved_in_message(self):
        """Test that grid squares are preserved in messages."""
        # FT8 message with callsign and grid square
        message = "CQ W1ABC FN42"
        result = self.anonymizer.anonymize_message(message)

        # Grid square should be preserved
        assert "FN42" in result["anonymized_message"]
        assert "FN42" in result["grid_squares"]

        # Callsign should be hashed
        assert "W1ABC" not in result["anonymized_message"]
        assert len(result["callsign_hashes"]) == 1

    def test_multiple_grid_squares_preserved(self):
        """Test multiple grid squares are all preserved."""
        message = "W1ABC FN42 K2DEF IO91 VE3GHI EN82"
        result = self.anonymizer.anonymize_message(message)

        # All grid squares should be preserved
        assert "FN42" in result["anonymized_message"]
        assert "IO91" in result["anonymized_message"]
        assert "EN82" in result["anonymized_message"]

        # Check grid_squares list
        assert set(result["grid_squares"]) == {"FN42", "IO91", "EN82"}

        # Callsigns should be hashed
        assert "W1ABC" not in result["anonymized_message"]
        assert "K2DEF" not in result["anonymized_message"]
        assert "VE3GHI" not in result["anonymized_message"]

    def test_wspr_data_grid_preservation(self):
        """Test WSPR data anonymization preserves grid squares."""
        wspr_data = {
            "callsign": "W1ABC",
            "grid": "FN42",
            "power_dbm": 23,
            "frequency": 14097100
        }

        result = self.anonymizer.anonymize_wspr_data(wspr_data)

        # Grid should be preserved
        assert result["grid"] == "FN42"

        # Callsign should be hashed
        assert "callsign_hash" in result
        assert result.get("callsign") != "W1ABC"

    def test_ft8_data_grid_preservation(self):
        """Test FT8 data anonymization preserves grid squares."""
        ft8_data = {
            "tx_callsign": "W1ABC",
            "rx_callsign": "K2DEF",
            "grid": "FN42",
            "message": "K2DEF W1ABC FN42"
        }

        result = self.anonymizer.anonymize_ft8_data(ft8_data)

        # Grid should be preserved
        assert result.get("grid") == "FN42"

        # Callsigns should be hashed
        assert "tx_callsign_hash" in result
        assert "rx_callsign_hash" in result

    def test_six_character_grid_preservation(self):
        """Test 6-character grid squares are preserved."""
        message = "CQ DX W1ABC FN42ab"
        result = self.anonymizer.anonymize_message(message)

        # 6-character grid should be preserved
        assert "FN42AB" in result["grid_squares"]  # Normalized to uppercase
        # Original case might be preserved in message
        assert "FN42ab" in result["anonymized_message"] or "FN42AB" in result["anonymized_message"]

    def test_grid_not_treated_as_callsign(self):
        """Test that grid squares are not mistakenly hashed as callsigns."""
        # FN42 could match callsign pattern but should be recognized as grid
        message = "FN42 IO91"
        result = self.anonymizer.anonymize_message(message)

        # Both should be preserved as grids
        assert "FN42" in result["anonymized_message"]
        assert "IO91" in result["anonymized_message"]
        assert len(result["callsign_hashes"]) == 0  # No callsigns to hash

    def test_privacy_method_updated(self):
        """Test that privacy method indicates grid preservation."""
        message = "W1ABC FN42"
        result = self.anonymizer.anonymize_message(message)

        # Should indicate grid preservation in method
        assert "GRID_PRESERVED" in result["privacy_method"]

    def test_anonymization_result_with_grids(self):
        """Test AnonymizationResult with grid preservation."""
        data_list = [
            {"callsign": "W1ABC", "grid": "FN42"},
            {"callsign": "K2DEF", "grid": "IO91"}
        ]

        result = self.anonymizer.batch_anonymize(data_list, data_type="wspr")

        assert isinstance(result, AnonymizationResult)
        assert result.anonymization_method == "SHA256_SALTED_GRID_PRESERVED"
        assert result.original_count >= 2
        assert result.anonymized_count >= 2

    def test_consistent_hashing(self):
        """Test that same callsign produces same hash."""
        message1 = "W1ABC FN42"
        message2 = "W1ABC IO91"

        result1 = self.anonymizer.anonymize_message(message1)
        result2 = self.anonymizer.anonymize_message(message2)

        # Same callsign should produce same hash
        assert result1["callsign_hashes"][0] == result2["callsign_hashes"][0]

        # But grids should be different and preserved
        assert "FN42" in result1["grid_squares"]
        assert "IO91" in result2["grid_squares"]

    def test_edge_cases(self):
        """Test edge cases for anonymization."""
        # Empty message
        result = self.anonymizer.anonymize_message("")
        assert result["anonymized_message"] == ""
        assert result["grid_squares"] == []

        # Only grid squares, no callsigns
        result = self.anonymizer.anonymize_message("FN42 IO91 EN82")
        assert len(result["callsign_hashes"]) == 0
        assert len(result["grid_squares"]) == 3

        # Mixed case grids
        result = self.anonymizer.anonymize_message("fn42 IO91 En82")
        assert set(result["grid_squares"]) == {"FN42", "IO91", "EN82"}

    def test_validate_privacy_compliance(self):
        """Test privacy compliance validation."""
        # Compliant data (grid preserved, callsign hashed)
        data = {
            "callsign_hash": "abc123",
            "grid": "FN42"
        }
        assert validate_privacy_compliance(data) == True

        # Non-compliant data (raw callsign)
        data = {
            "callsign": "W1ABC",
            "grid": "FN42"
        }
        # Should detect raw callsign
        validation = CallsignAnonymizer().validate_anonymization(data)
        assert "callsign" in [v.lower() for v in validation.get("raw_callsigns", [])]