# CASCADE Model Updates and Deployment

CASCADE models can be updated over-the-air (OTA) with user consent, enabling continuous improvement while maintaining security and backward compatibility.

## Update Philosophy

**User control**: Updates NEVER install automatically
**Explicit consent**: User must click "Install"
**Cryptographic verification**: All updates signed by CASCADE team
**Backward compatibility**: Newer models work with older models
**Rollback safety**: Automatic backup and self-test

## Update Check Mechanism

**User-configurable update checking:**

```python
class UpdateChecker:
    """Non-intrusive update availability checking"""

    def __init__(self):
        self.check_enabled = user_prefs.get('auto_check_updates', True)
        self.check_interval = user_prefs.get('check_interval_hours', 24)
        self.last_check = load_last_check_time()

    def periodic_check(self):
        """Background check during idle time"""

        if not self.check_enabled:
            return  # User disabled auto-check

        if (now() - self.last_check) < self.check_interval * 3600:
            return  # Not time yet

        # Query update server (lightweight, just metadata)
        try:
            update_info = query_update_server(
                current_version=CASCADE_VERSION,
                hardware_tier=detect_hardware()
            )

            if update_info and update_info.version > CASCADE_VERSION:
                # New version available - notify user
                self.notify_update_available(update_info)

            self.last_check = now()

        except NetworkError:
            # No internet - silent fail
            pass
```

**Update notification** (non-intrusive):

```
╔═══════════════════════════════════════════════════════════╗
║  CASCADE Update Available                                 ║
╠═══════════════════════════════════════════════════════════╣
║  Current version: v1.0.2                                  ║
║  New version: v1.1.0                                      ║
║                                                           ║
║  Changes:                                                 ║
║  • Improved Signal Expert (15% better multi-user)         ║
║  • Enhanced kernel compression                            ║
║  • Bug fixes for weak signal decode                       ║
║                                                           ║
║  Size: 10.2 MB                                           ║
║  Compatibility: Works with v1.0+ stations                 ║
║                                                           ║
║  [Download Now]  [Remind Later]  [Ignore This Version]    ║
╚═══════════════════════════════════════════════════════════╝
```

## Download and Verification

**User clicks "Download Now":**

```python
def download_update(update_info):
    """Download with verification"""

    # Show progress dialog
    progress = show_progress_dialog(
        title="Downloading CASCADE v{update_info.version}",
        cancelable=True
    )

    try:
        # Download during idle time (pauses during active QSOs)
        update_file = download_with_resume(
            url=update_info.download_url,
            expected_size=update_info.size_bytes,
            expected_hash=update_info.sha256_hash,
            progress_callback=lambda pct: progress.set_percent(pct),
            pause_during_tx=True  # Don't compete with active transmissions
        )

        # Verify SHA-256 hash
        if hash_file(update_file) != update_info.sha256_hash:
            raise VerificationError("Hash mismatch!")

        # Verify cryptographic signature (RSA-4096 or Ed25519)
        signature = update_info.signature
        if not verify_signature(update_file, signature, CASCADE_PUBLIC_KEY):
            raise VerificationError("Signature invalid!")

        # Verification passed
        progress.close()
        return update_file

    except Exception as e:
        alert(f"Download failed: {e}\nPlease try again later.")
        return None
```

**Cryptographic verification:**

```python
CASCADE_PUBLIC_KEY = """
-----BEGIN PUBLIC KEY-----
MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEA...
-----END PUBLIC KEY-----
"""

def verify_signature(update_file, signature, public_key):
    """Verify update authenticity"""

    # Compute hash of update file
    file_hash = sha256(update_file)

    # Verify signature using CASCADE public key
    try:
        rsa_key = RSA.import_key(public_key)
        verifier = PKCS1_v1_5.new(rsa_key)
        verified = verifier.verify(file_hash, signature)
        return verified
    except Exception:
        return False
```

## Installation Process

**User confirms installation:**

```python
def install_update(update_file):
    """Install verified update with safety checks"""

    # 1. Backup current version
    backup_path = backup_current_model()
    log(f"Current version backed up to: {backup_path}")

    # 2. Extract update package
    model_new = extract_update(update_file)

    # 3. Verify model structure
    if not validate_model_structure(model_new):
        alert("Update package malformed!")
        return False

    # 4. Check pattern compatibility (CRITICAL)
    if not patterns_match(model_new.patterns, current_model.patterns):
        alert("CRITICAL: Patterns don't match! Update rejected.")
        return False

    # 5. Install new model
    install_model_files(model_new)

    # 6. Self-test
    if not self_test():
        alert("Self-test failed! Rolling back...")
        rollback(backup_path)
        return False

    # 7. Success
    show_success(f"Successfully updated to v{model_new.version}")
    return True

def self_test():
    """Verify new model works"""

    try:
        # Test basic encode/decode
        test_data = b"TEST MESSAGE"
        encoded = model.encode(test_data, default_kernel)
        decoded = model.decode(encoded)

        if decoded != test_data:
            return False

        # Test pattern correlation (ensure patterns unchanged)
        for pattern_id in range(64):
            corr = test_pattern_orthogonality(pattern_id)
            if corr > -30:  # dB
                return False  # Patterns corrupted

        # Test beacon decode
        test_beacon = generate_test_beacon()
        decoded_beacon = model.decode(test_beacon)
        if not decoded_beacon:
            return False

        # All tests passed
        return True

    except Exception as e:
        log(f"Self-test exception: {e}")
        return False
```

## Version Compatibility in Practice

**Mixed-version network example:**

```
Network has:
- 10 stations running v1.0
- 35 stations running v2.0
- 5 stations running v3.0

All communicate seamlessly:
- v1.0 ↔ v1.0: Full v1.0 features
- v1.0 ↔ v2.0: v2.0 falls back to v1.0 mode
- v1.0 ↔ v3.0: v3.0 falls back to v1.0 mode
- v2.0 ↔ v2.0: Full v2.0 features
- v2.0 ↔ v3.0: v3.0 falls back to v2.0 mode
- v3.0 ↔ v3.0: Full v3.0 features

Network functions: All stations interoperate
Performance: Pairs use best common version
```

## Update Package Format

```python
cascade_update_package = {
    'version': '2.0.1',
    'release_date': '2027-03-15',
    'min_compatible_version': '1.0.0',

    # Model files
    'model_weights_int8': 'cascade_v2.0.1_int8.pth',    # 10MB
    'model_weights_fp16': 'cascade_v2.0.1_fp16.pth',    # 18MB (optional)
    'patterns_verification': 'patterns_checksum.txt',    # Verify patterns unchanged

    # Metadata
    'changelog': 'changelog_v2.0.1.md',
    'size_mb': 10.2,
    'sha256': 'a3f2b91c7e4d8c2a...',

    # Cryptographic verification
    'signature': 'RSA_signature_bytes',
    'signing_key_id': 'CASCADE_RELEASE_2024',

    # Compatibility
    'v1_compat_mode': True,         # Can operate as v1.0
    'v2_compat_mode': True,         # Native v2.0
    'breaking_changes': []          # None (backward compatible)
}
```

## User Preferences

**Update settings:**

```
CASCADE Settings > Updates

[✓] Automatically check for updates
    Check interval: [Daily ▼] [Weekly] [Monthly] [Manual only]

[✓] Download updates in background
    [ ] Notify me before downloading
    [✓] Only download on WiFi (not cellular)

Update channel:
    [●] Stable (recommended)
    [ ] Beta (early access, may have bugs)
    [ ] Dev (bleeding edge, frequent updates)

Current version: v1.0.2
Last check: 2 hours ago
[Check Now]
```

## Rollback Procedure

**If update causes problems:**

```python
# Automatic rollback (if self-test fails)
if not self_test_after_update():
    rollback_to_backup()
    alert("Update failed self-test, automatically rolled back to v1.0.2")

# Manual rollback (user-initiated)
def manual_rollback():
    backups = list_available_backups()

    show_dialog(
        "Available Backups:\n"
        "• v1.0.2 (previous, backed up 2 hours ago)\n"
        "• v1.0.1 (backed up 2 weeks ago)\n\n"
        "Select version to restore: [v1.0.2] [v1.0.1] [Cancel]"
    )

    if user_selects_backup:
        restore_from_backup(selected_version)
        restart_cascade()
```

## Security Considerations

**Public key distribution:**
- CASCADE public key embedded in initial installation
- Updated only via signed key rotation (rare, requires user consent)
- Key rotation announced via multiple channels (website, email list, QRZ)

**Update server security:**
- HTTPS only (TLS 1.3+)
- Certificate pinning (prevent MITM)
- Multiple CDN mirrors (prevent single point of failure)
- Transparency log (all updates publicly logged)

**Malicious update prevention:**
- Signature verification (prevents fake updates)
- Self-test before finalizing (prevents broken updates)
- Community verification (checksums published, users can verify)
- Gradual rollout (10% → 50% → 100% over days)

## See Also

- **[Version Compatibility](version_compatibility.md)** - How versions interoperate
- **[Hardware Requirements](hardware_requirements.md)** - Deployment tiers
- **[Signal Specification](../protocol/signal_specification.md)** - Protocol elements (frozen)
