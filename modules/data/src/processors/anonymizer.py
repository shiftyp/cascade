"""Callsign anonymizer for privacy protection.

Implements T032: Callsign anonymizer (FR-005, FR-006).
"""

import hashlib
import logging
import re
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass

from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class AnonymizationResult:
    """Result of anonymization process."""

    original_count: int
    anonymized_count: int
    callsign_hashes: List[str]
    privacy_level: str
    anonymization_method: str


class CallsignAnonymizer:
    """Anonymizes callsigns while preserving propagation analysis value.

    IMPORTANT: Grid squares are preserved in cleartext to enable propagation
    distance calculations. Only callsigns are hashed for privacy.
    """

    def __init__(self, salt: Optional[str] = None):
        """Initialize anonymizer.

        Args:
            salt: Cryptographic salt for hashing (uses config if not provided)
        """
        self.salt = salt or config.CALLSIGN_SALT or "cascade_default_salt_2025"
        self.callsign_pattern = re.compile(
            r'\b([A-Z0-9]{1,3}[0-9][A-Z0-9]*(?:/[A-Z0-9]{1,4})?)\b'
        )
        # Grid square pattern (4 or 6 character Maidenhead locator)
        self.grid_pattern = re.compile(
            r'\b([A-R]{2}[0-9]{2}[A-X]{0,2})\b',
            re.IGNORECASE
        )

        # Track anonymized callsigns for consistency
        self._anonymization_cache: Dict[str, str] = {}

    def anonymize_message(self, message: str) -> Dict[str, Any]:
        """Anonymize all callsigns in a message while preserving grid squares.

        Args:
            message: Original message text

        Returns:
            Dictionary with anonymized message and metadata
        """
        if not message:
            return {
                "anonymized_message": "",
                "callsign_hashes": [],
                "grid_squares": [],
                "original_count": 0,
                "anonymized_count": 0
            }

        # Extract grid squares first (preserve these)
        grid_squares = self.grid_pattern.findall(message)
        grid_squares = [gs.upper() for gs in grid_squares]

        # Find all callsigns in message (excluding grid squares)
        potential_callsigns = self.callsign_pattern.findall(message)
        callsigns = []

        for pc in potential_callsigns:
            # Skip if it's actually a grid square
            if not self.is_valid_grid_square(pc):
                callsigns.append(pc)

        unique_callsigns = list(set(callsigns))

        # Anonymize each callsign
        anonymized_message = message
        callsign_hashes = []

        for callsign in unique_callsigns:
            anonymized_hash = self._hash_callsign(callsign)
            callsign_hashes.append(anonymized_hash)

            # Replace in message (preserving structure for FT8/WSPR analysis)
            anonymized_message = anonymized_message.replace(callsign, anonymized_hash[:8])

        return {
            "anonymized_message": anonymized_message,
            "callsign_hashes": callsign_hashes,
            "grid_squares": grid_squares,  # Preserved in cleartext
            "original_count": len(callsigns),
            "anonymized_count": len(unique_callsigns),
            "privacy_method": "SHA256_SALTED_GRID_PRESERVED"
        }

    def anonymize_ft8_data(self, ft8_data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize FT8 signal data.

        Args:
            ft8_data: FT8 signal data dictionary

        Returns:
            Anonymized FT8 data
        """
        anonymized = ft8_data.copy()

        # Anonymize message content
        if "message" in anonymized:
            result = self.anonymize_message(anonymized["message"])
            anonymized["message_hash"] = result["callsign_hashes"][0] if result["callsign_hashes"] else None
            # Remove original message for privacy
            del anonymized["message"]

        # Anonymize specific callsign fields
        for field in ["callsign", "tx_callsign", "rx_callsign"]:
            if field in anonymized:
                anonymized[f"{field}_hash"] = self._hash_callsign(anonymized[field])
                del anonymized[field]

        # Preserve propagation-relevant data
        preserve_fields = [
            "timestamp", "frequency_hz", "snr_db", "drift_hz",
            "grid_square", "distance_km", "bearing_degrees",
            "propagation_mode", "decoded_successfully"
        ]

        return {k: v for k, v in anonymized.items() if k in preserve_fields or k.endswith("_hash")}

    def anonymize_wspr_data(self, wspr_data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize WSPR signal data.

        Args:
            wspr_data: WSPR signal data dictionary

        Returns:
            Anonymized WSPR data
        """
        anonymized = wspr_data.copy()

        # Anonymize callsign fields
        for field in ["callsign", "tx_callsign", "rx_callsign"]:
            if field in anonymized:
                anonymized[f"{field}_hash"] = self._hash_callsign(anonymized[field])
                del anonymized[field]

        # Preserve propagation-relevant data
        preserve_fields = [
            "timestamp", "frequency_hz", "snr_db", "drift_hz", "power_dbm",
            "grid_square", "distance_km", "bearing_degrees",
            "propagation_mode"
        ]

        return {k: v for k, v in anonymized.items() if k in preserve_fields or k.endswith("_hash")}

    def anonymize_batch(self, data_list: List[Dict[str, Any]], data_type: str = "auto") -> AnonymizationResult:
        """Anonymize a batch of signal data.

        Args:
            data_list: List of signal data dictionaries
            data_type: Type of data ("ft8", "wspr", or "auto")

        Returns:
            Anonymization result summary
        """
        anonymized_data = []
        all_hashes: Set[str] = set()
        original_count = 0
        anonymized_count = 0

        for data in data_list:
            # Auto-detect data type if needed
            if data_type == "auto":
                if "message" in data or "message_type" in data:
                    detected_type = "ft8"
                elif "power_dbm" in data:
                    detected_type = "wspr"
                else:
                    detected_type = "generic"
            else:
                detected_type = data_type

            # Anonymize based on type
            if detected_type == "ft8":
                anonymized = self.anonymize_ft8_data(data)
            elif detected_type == "wspr":
                anonymized = self.anonymize_wspr_data(data)
            else:
                anonymized = self._anonymize_generic(data)

            anonymized_data.append(anonymized)

            # Collect statistics
            for key, value in anonymized.items():
                if key.endswith("_hash") and value:
                    all_hashes.add(value)
                    anonymized_count += 1

            # Count original callsigns
            for key, value in data.items():
                if key in ["callsign", "tx_callsign", "rx_callsign", "message"]:
                    if isinstance(value, str) and value:
                        callsigns = self.callsign_pattern.findall(value)
                        original_count += len(callsigns)

        return AnonymizationResult(
            original_count=original_count,
            anonymized_count=anonymized_count,
            callsign_hashes=list(all_hashes),
            privacy_level="HIGH",
            anonymization_method="SHA256_SALTED_GRID_PRESERVED"
        )

    def _anonymize_generic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize generic data structure.

        Args:
            data: Generic data dictionary

        Returns:
            Anonymized data
        """
        anonymized = {}

        for key, value in data.items():
            if isinstance(value, str) and self._contains_callsign(value):
                # Anonymize string fields containing callsigns
                anonymized[f"{key}_hash"] = self._hash_callsign(value)
            elif key.lower() in ["callsign", "call", "station"]:
                # Anonymize known callsign fields
                anonymized[f"{key}_hash"] = self._hash_callsign(str(value))
            else:
                # Preserve non-PII fields
                anonymized[key] = value

        return anonymized

    def _hash_callsign(self, callsign: str) -> str:
        """Hash a callsign with salt.

        Args:
            callsign: Callsign to hash

        Returns:
            Hashed callsign
        """
        if not callsign:
            return ""

        # Normalize callsign (uppercase, strip)
        normalized = callsign.upper().strip()

        # Check cache for consistency
        if normalized in self._anonymization_cache:
            return self._anonymization_cache[normalized]

        # Create salted hash
        hash_input = f"{normalized}:{self.salt}".encode('utf-8')
        hash_result = hashlib.sha256(hash_input).hexdigest()

        # Use first 16 characters for readability
        short_hash = hash_result[:16]

        # Cache for consistency
        self._anonymization_cache[normalized] = short_hash

        return short_hash

    def _contains_callsign(self, text: str) -> bool:
        """Check if text contains callsigns.

        Args:
            text: Text to check

        Returns:
            True if text contains callsigns
        """
        return bool(self.callsign_pattern.search(text))

    def validate_anonymization(self, anonymized_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that anonymization was successful.

        Args:
            anonymized_data: Anonymized data to validate

        Returns:
            Validation result
        """
        issues = []
        privacy_score = 100

        # Check for remaining callsigns
        for key, value in anonymized_data.items():
            if isinstance(value, str):
                remaining_callsigns = self.callsign_pattern.findall(value)
                if remaining_callsigns:
                    issues.append(f"Callsigns found in {key}: {remaining_callsigns}")
                    privacy_score -= 20

        # Check for PII fields
        pii_fields = ["callsign", "call", "station", "message"]
        for field in pii_fields:
            if field in anonymized_data:
                issues.append(f"PII field '{field}' not anonymized")
                privacy_score -= 15

        # Check for proper hash fields
        hash_fields = [k for k in anonymized_data.keys() if k.endswith("_hash")]
        if not hash_fields:
            issues.append("No anonymized hash fields found")
            privacy_score -= 10

        return {
            "privacy_score": max(0, privacy_score),
            "is_valid": len(issues) == 0,
            "issues": issues,
            "hash_fields": hash_fields
        }

    def get_anonymization_stats(self) -> Dict[str, Any]:
        """Get anonymization statistics.

        Returns:
            Anonymization statistics
        """
        return {
            "total_callsigns_processed": len(self._anonymization_cache),
            "salt_length": len(self.salt),
            "hash_method": "SHA256",
            "cache_size": len(self._anonymization_cache),
            "privacy_level": "HIGH",
            "reversible": False
        }

    def clear_cache(self):
        """Clear anonymization cache."""
        self._anonymization_cache.clear()
        logger.info("Anonymization cache cleared")


# Utility functions
def anonymize_ft8_message(message: str) -> str:
    """Quick anonymization of FT8 message.

    Args:
        message: FT8 message

    Returns:
        Anonymized message
    """
    anonymizer = CallsignAnonymizer()
    result = anonymizer.anonymize_message(message)
    return result["anonymized_message"]


def anonymize_callsign_list(callsigns: List[str]) -> List[str]:
    """Quick anonymization of callsign list.

    Args:
        callsigns: List of callsigns

    Returns:
        List of anonymized callsigns
    """
    anonymizer = CallsignAnonymizer()
    return [anonymizer._hash_callsign(call) for call in callsigns]


def validate_privacy_compliance(data: Dict[str, Any]) -> bool:
    """Validate data for privacy compliance.

    Args:
        data: Data to validate

    Returns:
        True if privacy compliant
    """
    anonymizer = CallsignAnonymizer()
    validation = anonymizer.validate_anonymization(data)
    return validation["is_valid"]


# Add missing methods to CallsignAnonymizer class
CallsignAnonymizer.is_valid_grid_square = lambda self, text: self._is_valid_grid_square(text)

def _is_valid_grid_square(self, text: str) -> bool:
    """Check if text is a valid Maidenhead grid square.

    Args:
        text: Text to validate

    Returns:
        True if valid grid square
    """
    if not text:
        return False

    text = text.upper()

    # Check format (4 or 6 characters)
    if len(text) not in [4, 6]:
        return False

    # Check first two characters are A-R
    if len(text) >= 2:
        if not (text[0] in 'ABCDEFGHIJKLMNOPQR' and
                text[1] in 'ABCDEFGHIJKLMNOPQR'):
            return False

    # Check next two are digits
    if len(text) >= 4:
        if not (text[2].isdigit() and text[3].isdigit()):
            return False

    # If 6 characters, check last two are A-X
    if len(text) == 6:
        if not (text[4] in 'ABCDEFGHIJKLMNOPQRSTUVWX' and
                text[5] in 'ABCDEFGHIJKLMNOPQRSTUVWX'):
            return False

    return True

# Bind the method properly
CallsignAnonymizer._is_valid_grid_square = _is_valid_grid_square