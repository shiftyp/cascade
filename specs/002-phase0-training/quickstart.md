# Quickstart: Phase 0 Vetting Validation

**Feature**: 002-phase0-training
**Purpose**: Validate that Phase 0 vetting system works end-to-end

---

## Prerequisites

- Python 3.11+ installed
- PyTorch 2.0+ with CUDA support
- 1x RTX 4090 GPU (or equivalent)
- 64 GB RAM available
- CASCADE repository cloned

---

## Quick Validation Steps

### Step 1: Environment Setup

```bash
cd /workspaces/cascade

# Create vetting environment
python -m venv venv-vetting
source venv-vetting/bin/activate

# Install dependencies
pip install torch numpy scipy pytest reedsolo

# Verify GPU available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Expected: CUDA available: True
```

### Step 2: Run Phase 0 Vetting

```bash
# Run full vetting suite (60 hours GPU time)
python -m modules.training.vetting.validator \
    --output-dir ./vetting_results \
    --gpu 0

# Expected output:
# Test 1/7: Single User Baseline
#   Users: 1, Accuracy: 99.9%, Shannon: 96.2% ✓ PASS
# Test 2/7: Pattern Orthogonality
#   Users: 10, Accuracy: 98.3%, Shannon: 92.7% ✓ PASS
# Test 3/7: Frequency Reuse
#   Users: 20, Accuracy: 95.8%, Shannon: 90.4% ✓ PASS
# Test 4/7: Time Reuse
#   Users: 30, Accuracy: 93.5%, Shannon: 88.1% ✓ PASS
# Test 5/7: Full Chaos (CRITICAL)
#   Users: 45, Accuracy: 90.2%, Shannon: 85.6% ✓ PASS ← Key result
# Test 6/7: Kernel Coordination
#   Users: 45, Accuracy: 92.1%, Shannon: 87.3% ✓ PASS
# Test 7/7: SNR Degradation
#   SNR Sweep: +15 to -22 dB, Graceful degradation ✓ PASS
#
# =====================================
# VETTING RESULT: ✓ ARCHITECTURE VALIDATED
# =====================================
# Test 5 achieved 85.6% Shannon (>= 85% threshold)
# Recommendation: PROCEED_REAL_DATA, PROCEED_SYNTHETIC, or PROCEED_HYBRID
# See ./vetting_results/validation_report.md for details
```

### Step 3: Review Validation Report

```bash
# Open generated report
cat ./vetting_results/validation_report.md

# Expected sections:
# - Executive Summary (PASS/FAIL)
# - Test-by-Test Results
# - Critical Test Analysis (Test 5 deep dive)
# - Shannon Efficiency Analysis
# - Recommendations (3 deployment paths)
# - Next Steps
```

### Step 4: Verify Critical Metrics

```bash
# Check Test 5 specifically
grep "Test 5" ./vetting_results/validation_report.md

# Expected:
# Test 5: Full Chaos (45 Users)
# - Achieved: 85.6% Shannon efficiency
# - Target: >= 85%
# - Status: ✓ PASS
# - Implication: Architecture validated, 78-85% Shannon in real HF is realistic
```

### Step 5: Review Recommendations

```bash
# Check recommended next steps
grep -A10 "Recommendations" ./vetting_results/validation_report.md

# Expected (if Test 5 passed):
# Recommendations:
#
# Path A - Conservative: Wait 18 months for 150K hours real data
#   → V1.0 launch at 78-85% Shannon
#
# Path B - Aggressive: Train on synthetic (3 weeks)
#   → V0.5 beta at 65-70% Shannon, improve via telemetry
#
# Path C - Balanced: Hybrid 5K real + synthetic (2-3 months)
#   → V0.8 at 70-75% Shannon
```

---

## Expected Outcomes

### If Vetting Passes (Test 5 >= 85%)

**Console output shows:**
```
✓ ARCHITECTURE VALIDATED
  Test 5 (45-user chaos): 85.6% Shannon
  Frequency reuse: Working (Test 3 passed)
  Time reuse: Working (Test 4 passed)
  Kernel coordination: +2.1% improvement (Test 6 passed)

Recommendation: Architecture is sound, proceed with chosen path
```

**Next steps:**
- Review validation_report.md
- Choose deployment path (A, B, or C)
- If Path A: Start real data collection
- If Path B: Begin synthetic training
- If Path C: Collect 5K hours quick dataset

### If Vetting Fails (Test 5 < 80%)

**Console output shows:**
```
✗ ARCHITECTURE NEEDS REVISION
  Test 5 (45-user chaos): 78.2% Shannon (< 85% threshold)

Identified Issues:
  - Pattern orthogonality may be insufficient
  - Chaos overlap tolerance lower than expected

Recommendation: FIX_ARCHITECTURE before data collection
```

**Next steps:**
- Review which tests failed
- Analyze identified issues
- Revise architecture (more patterns? Better orthogonality? Add coordination?)
- Re-run vetting after fixes

---

## Validation Checklist

After running quickstart, verify:

- [ ] Vetting completed without crashes
- [ ] All 7 tests executed and reported results
- [ ] Test 5 (45-user chaos) showed Shannon efficiency measurement
- [ ] Overall pass/fail determination made (based on 85% threshold)
- [ ] Validation report generated in markdown
- [ ] Recommendations provided based on results
- [ ] Total duration approximately 60 hours
- [ ] Results reproducible with same seed

---

## Troubleshooting

**If vetting crashes:**
- Check GPU memory (need 16+ GB VRAM)
- Check RAM usage (need 64 GB)
- Reduce num_samples if memory limited

**If results seem wrong:**
- Verify AWGN noise power calculation
- Check Shannon formula implementation
- Ensure reproducibility (use fixed seeds)

**If Test 5 fails unexpectedly:**
- Review earlier test results (may indicate systemic issue)
- Check if frequency reuse working (Test 3)
- Verify chaos overlap handling (should tolerate via RS)

---

## Success Criteria

**Quickstart validation succeeds if:**
- ✓ Vetting runs to completion
- ✓ Test 5 produces Shannon efficiency measurement
- ✓ Pass/fail logic works correctly
- ✓ Validation report is generated and readable
- ✓ Recommendations make sense based on results

**This validates the vetting system itself is working correctly.**

The actual vetting RESULTS (pass vs fail) depend on CASCADE model performance, which is what we're testing.
