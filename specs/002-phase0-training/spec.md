# Feature Specification: Phase 0 Training - Ideal Conditions Vetting

**Feature Branch**: `002-phase0-training`
**Created**: 2025-10-04
**Status**: Draft
**Input**: User description: "phase0_training"

## Module Context

### Target Module
**Training Module** - Validates CASCADE architecture before investing in 18-month real HF data collection

### Module Dependencies
- **Pattern generation** (prerequisite): Must have 128 patterns generated with -37.5 dB orthogonality
- **Model architecture** (prerequisite): Expert networks and conductor must be implemented
- **Data module** (independent): Phase 0 runs BEFORE data collection starts

### Module Interfaces
- **Input**: Synthetic CASCADE signal generation (pure Python, no real data)
- **Output**: Validation report showing architecture performance in ideal conditions
- **Decision point**: Pass/fail determines whether to proceed with data collection

---

## User Scenarios & Testing

### Primary User Story

As a **CASCADE project lead**, I need to validate that the 128-pattern chaos architecture can achieve target performance (85%+ Shannon efficiency with 45 users) in ideal conditions before committing 18 months and $5K+ to real HF data collection. This de-risks the project by catching fundamental architecture flaws early (2.5 days vs 18 months).

### Acceptance Scenarios

1. **Given** CASCADE architecture is designed but untested, **When** Phase 0 vetting runs 7 progressive tests in AWGN-only conditions, **Then** system achieves 85%+ Shannon efficiency with 45 simultaneous users

2. **Given** vetting shows 85%+ Shannon in AWGN, **When** project team reviews results, **Then** team has confidence that 78-85% Shannon in real HF is achievable (accounting for 10-15% degradation from real impairments)

3. **Given** vetting shows pattern orthogonality test achieves 98% accuracy with 10 users, **When** test progresses to frequency reuse (same pattern on different tones), **Then** 20 users achieve 95% accuracy proving frequency reuse works

4. **Given** vetting shows kernel coordination test, **When** prokernels and antikernels are used, **Then** performance improves by 2-5% over no-kernel baseline proving emergent coordination mechanism works

5. **Given** vetting achieves <80% Shannon efficiency, **When** results are reviewed, **Then** architecture issues are identified and system does NOT proceed to expensive data collection

### Edge Cases

- **What happens when** pattern orthogonality is insufficient (<-30 dB)?
  - Test 2 (10 users) will fail to separate users, showing architecture needs more patterns or better orthogonality

- **What happens when** frequency reuse causes unexpected interference?
  - Test 3 will show same pattern on different tones still interferes, invalidating 1,024 user capacity claim

- **What happens when** kernel coordination provides no benefit or makes performance worse?
  - Test 6 will show kernels don't help, invalidating kernel-driven coordination design

- **What happens when** RS(32,20) cannot handle 37.5% symbol erasures in chaos?
  - Tests 4-5 will fail, showing chaos overlaps exceed RS tolerance

---

## Requirements

### Functional Requirements

#### Validation Testing
- **FR-001**: System MUST train CASCADE model in 7 progressive test scenarios from single user to 45-user chaos
- **FR-002**: System MUST use AWGN-only channel (no QRN, QRM, multipath, fading, or Doppler)
- **FR-003**: System MUST keep multi-user interference, chaos overlaps, async timing, clock drift, and RS erasures
- **FR-004**: System MUST measure Shannon efficiency, decode accuracy, and per-user throughput for each test
- **FR-005**: System MUST generate synthetic CASCADE signals following 128-pattern specification with RS(32,20) structure

#### Test Progression
- **FR-006**: Test 1 (Single User) MUST achieve 99.9% accuracy and 95%+ Shannon to validate basic encode/decode
- **FR-007**: Test 2 (10 Users) MUST achieve 98% accuracy and 92% Shannon to validate pattern orthogonality
- **FR-008**: Test 3 (20 Users) MUST test frequency reuse with same patterns on different tone selections
- **FR-009**: Test 4 (30 Users) MUST test time reuse with asynchronous starts and partial overlaps
- **FR-010**: Test 5 (45 Users Full Chaos) MUST achieve 85%+ Shannon efficiency as critical threshold
- **FR-011**: Test 6 (Kernel Coordination) MUST show 2%+ improvement from prokernel/antikernel exchange
- **FR-012**: Test 7 (SNR Sweep) MUST test degradation from +15 dB to -22 dB with graceful capacity reduction

#### Success Criteria
- **FR-013**: System MUST determine PASS if Test 5 achieves ≥85% Shannon with 45 users
- **FR-014**: System MUST determine FAIL if Test 5 achieves <80% Shannon requiring architecture revision
- **FR-015**: System MUST generate comprehensive validation report with pass/fail for each test

#### Decision Framework
- **FR-016**: System MUST provide clear recommendation on next steps based on vetting results
- **FR-017**: If vetting passes, system MUST present 3 deployment path options (wait for real data, synthetic training, or hybrid)
- **FR-018**: If vetting fails, system MUST identify likely architecture issues and recommend fixes

### Non-Functional Requirements

#### Performance
- **NFR-001**: Vetting MUST complete within 60 hours of GPU time (2.5 days on 1x RTX 4090)
- **NFR-002**: Each test MUST generate 5,000-50,000 training samples depending on complexity
- **NFR-003**: Memory usage MUST stay under 64 GB RAM during training

#### Accuracy
- **NFR-004**: Measurements MUST be reproducible with same random seed
- **NFR-005**: Shannon efficiency calculations MUST match theoretical limits within 1%
- **NFR-006**: Reported metrics MUST include confidence intervals

### Key Entities

**Test Scenario**: Represents one of 7 progressive vetting tests
- Attributes: number of users, patterns used, SNR level, target accuracy, target Shannon efficiency
- Relationships: Tests build on each other (must pass Test N before Test N+1)

**Vetting Result**: Outcome of running all 7 tests
- Attributes: pass/fail status, achieved Shannon efficiency, decode accuracy per test, identified issues
- Relationships: Determines which deployment path is recommended

**Training Sample**: Single multi-user AWGN scenario
- Attributes: number of users, user configurations (pattern, tones, start time, drift), ground truth labels
- Relationships: Thousands of samples per test scenario

**Validation Report**: Comprehensive output document
- Attributes: test results, performance metrics, architecture validation status, next step recommendations
- Relationships: Used for project decision-making (proceed with data collection or fix architecture)

---

## Review & Acceptance Checklist

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs (de-risk 18-month investment)
- [x] Written for non-technical stakeholders (project leads making go/no-go decisions)
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable (85%+ Shannon threshold)
- [x] Scope is clearly bounded (AWGN only, 7 specific tests)
- [x] Dependencies identified (needs 128 patterns, model architecture)

---

## Execution Status

- [x] User description parsed ("phase0_training" - validate architecture in ideal conditions)
- [x] Key concepts extracted (AWGN vetting, progressive tests, 85% threshold, risk reduction)
- [x] Ambiguities marked (none - phase0_vetting.md provides clear specification)
- [x] User scenarios defined (project lead validating architecture before data investment)
- [x] Requirements generated (18 functional requirements covering all 7 tests)
- [x] Entities identified (Test Scenario, Vetting Result, Training Sample, Validation Report)
- [x] Review checklist passed (no tech details, testable requirements, clear scope)

---

## Additional Context

### Reference Documentation
- **Detailed specification**: docs/training/phase0_vetting.md (875 lines)
- **Architecture overview**: docs/architecture.md
- **Training strategy**: docs/training/README.md
- **Pattern architecture**: docs/model/pattern_architecture.md

### Success Threshold Rationale

**85% Shannon in AWGN** chosen because:
- Real HF impairments (QRN, multipath, fading) cost 10-15% efficiency
- 85% - 10% = 75% (low end of 78-85% target) ✓
- 85% - 5% = 80% (if conditions favorable) ✓
- Provides margin for real-world complexity

If vetting shows <80% in ideal AWGN, then 78-85% in real HF is unrealistic.

### Three Deployment Paths (if vetting passes)

**Path A - Conservative**: Wait 18 months for 150K hours real data → 78-85% at V1.0 launch

**Path B - Aggressive**: Train on synthetic models (3 weeks) → Deploy V0.5 beta at 65-70% → Improve via telemetry

**Path C - Balanced**: Quick 5K hours real + synthetic (2-3 months) → Deploy V0.8 at 70-75% → V1.0 at 12 months

This specification defines the validation phase that informs which path to take.
