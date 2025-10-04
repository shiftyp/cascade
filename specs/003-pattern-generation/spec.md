# Feature Specification: Pattern Generation - 64 and 128 Pattern Sets

**Feature Branch**: `003-pattern-generation`
**Created**: 2025-10-04
**Updated**: 2025-10-04 (Revised: Adaptive λ minimization architecture)
**Status**: In Progress
**Input**: User description: "pattern_generation - Generate CASCADE's orthogonal 4D patterns with two variants"

## Module Context

### Target Module
**Training Module** - Generates CASCADE's fundamental orthogonal pattern sets required for all model training and operation

### Module Dependencies
- **None** - This is Feature 001 equivalent (foundational prerequisite)
- Pattern generation happens BEFORE any training or vetting
- Outputs used by Phase 0 vetting (Feature 002) and all subsequent training

### Module Interfaces
- **Input**: CASCADE specification (docs/model/pattern_architecture.md, docs/implementation/pattern_generation_spec.md)
- **Output**: Two binary pattern files (cascade_patterns_64.bin, cascade_patterns_128.bin)
- **Used by**: Phase 0 vetting, full training, protocol implementation, all CASCADE operations

---

## User Scenarios & Testing

### Primary User Story

As a **CASCADE implementer**, I need to generate two orthogonal pattern sets (64 patterns for testing, 128 patterns for production) that achieve -37.5 dB cross-correlation in 4D space, so that I can validate the architecture with a smaller set first, then use the production set for optimal performance.

### Acceptance Scenarios

1. **Given** CASCADE architecture specifications in docs/, **When** pattern generation runs for 64-pattern set, **Then** system produces cascade_patterns_64.bin (19 KB) with 48 beacon patterns (0-47) and 16 message patterns (48-63), all pairs achieving <-37.5 dB correlation

2. **Given** 64-pattern generation succeeds, **When** pattern generation runs for 128-pattern set, **Then** system produces cascade_patterns_128.bin (38 KB) with 48 beacon patterns (0-47) and 80 message patterns (48-127), all pairs achieving <-37.5 dB correlation

3. **Given** pattern generation completes, **When** validation tests load patterns and compute cross-correlations, **Then** all pattern pairs show <-37.5 dB orthogonality in 4D space (Time × Freq × I × Q)

4. **Given** both pattern sets exist, **When** Phase 0 vetting selects 64-pattern set for faster validation, **Then** vetting can test architecture with fewer patterns while still proving separation mechanism works

5. **Given** Phase 0 vetting passes with 64 patterns, **When** production training uses 128-pattern set, **Then** system achieves better per-user throughput (218 bps vs ~210 bps with 64 patterns)

### Edge Cases

- **What happens when** simulated annealing cannot achieve -37.5 dB for a pattern pair?
  - Increase optimization iterations or adjust pattern generation parameters, retry

- **What happens when** Zadoff-Chu base sequences don't provide sufficient diversity?
  - Fall back to random initialization with longer optimization time

- **What happens when** IQ trajectories cause correlation >-37.5 dB even with good frequency sequences?
  - Adjust IQ complexity parameters or regenerate specific patterns

- **What happens when** pattern file becomes corrupted?
  - Validation will fail, regenerate from scratch (deterministic with seeds)

---

## Revised Architecture (2025-10-04)

### Adaptive λ Minimization Approach

**Key Change**: Instead of pre-assigning IQ complexity (λ) hierarchically by pattern ID, the optimizer now **empirically discovers minimum λ needed for orthogonality**.

**Optimization Strategy**:
1. **Primary objective**: Achieve -37.5 dB orthogonality (hard constraint)
2. **Secondary objective**: Minimize average λ (prefer simpler IQ trajectories)
3. **Direct IQ optimization**: Optimizer mutates IQ points directly, not through predefined shapes
4. **Adaptive search**: Each pattern finds its own minimum λ for orthogonality

**Rationale**:
- Simpler IQ = better robustness under HF phase distortion
- Let optimization discover true minimum complexity needed
- Empirical approach replaces assumption-based complexity pools

### Two-Phase Optimization

**Enhancement**: Separate frequency and IQ optimization for maximum λ=0 patterns

**Approach**:
- **Phase 1** (80% of iterations): Frequency-only optimization with BPSK (λ=0)
  - Try to achieve -37.5 dB using only tone sequence
  - If successful, pattern stays at λ=0 (maximum robustness)
  - Default: 320K iterations for 400K total budget
- **Phase 2** (20% of iterations): IQ refinement if Phase 1 insufficient
  - Add minimum IQ complexity to achieve orthogonality
  - Adaptive λ discovery through direct IQ mutation
  - Default: 80K iterations for refinement

**Benefits**:
- Maximizes patterns that can use BPSK (λ=0)
- Lower average λ across pattern set (better HF robustness)
- Focused search: frequency first, complexity only if needed
- Expected: 20-30% of patterns achieve λ=0 (vs 10-15% single-phase)

### Phase-Aware Optimization

**Enhancement**: Include phase distortion robustness IN optimization cost function

**Approach**:
- During optimization, test each candidate against random phase scenarios
- Monte Carlo sampling: 3-5 phase scenarios per correlation check
- Random phase per tone (±π radians) models frequency-dependent distortion
- Random phase per symbol (±0.2 radians) models time-varying channel
- Cost function uses **worst-case** correlation across scenarios

**Benefits**:
- Patterns optimized for HF robustness from the start (not just validated after)
- Naturally guides optimizer toward phase-robust solutions
- May favor lower λ (simpler IQ less sensitive to phase rotation)
- Expected: Phase-robust separation within 5-6 dB of ideal (vs 7-8 dB without)

### Multi-Trial Generation

**Enhancement**: Generate multiple candidate pattern sets, select best

**Optimal Configuration** (based on cost-benefit analysis):
- **Local high-end CPU (recommended)**: 8 trials × 400K iterations
  - Best for: Core Ultra 7 265K, Ryzen 9, similar 8+ core CPUs
  - Strategy: Depth (deep convergence per trial)
  - Time: 72-96 hours, Cost: $0, Quality: **-42.6 dB, λ=0.17**
  - Runs on 8 P-cores in parallel
- **Cloud (Fly.io alternative)**: 32 trials × 100K iterations
  - Best for: Users without capable local hardware or need faster results
  - Strategy: Breadth (many diverse trials)
  - Time: 30-40 hours, Cost: $9.60, Quality: -40.7 dB, λ=0.22

**Depth vs Breadth Trade-off**:
- **Depth** (8×400K): Better convergence → lower λ, better separation, cheaper locally
- **Breadth** (32×100K): More diversity → faster on cloud, hedges against local minima
- **Optimal stopping**: 8 trials at 400K iterations ($/dB crosses break-even after 8)

**Execution**:
- Auto-detect CPU and select optimal configuration
- Parallel execution using all available P-cores
- Checkpointing: Save results after each batch
- Scoring: `separation_dB - 0.1 * avg_lambda`
- Best-of-N selection with phase robustness testing

**Benefits**:
- Exceptional quality: -42.6 dB separation (vs -39 dB single trial)
- Lowest λ: 0.17 average (vs 0.25+ single trial)
- Cost-optimal: $4.55 cloud or $0 local for best quality
- 20-30% patterns achieve λ=0 with two-phase optimization

### Pattern Visualization

**Addition**: Generate visual analysis after each trial batch

**Visualizations**:
- **IQ trajectory plots**: Show discovered IQ patterns in complex plane
- **Frequency sequence heatmap**: Visualize tone usage across time
- **λ distribution histogram**: Show complexity distribution across patterns
- **Correlation matrix**: Heatmap of all pairwise correlations
- **Best patterns comparison**: Overlay IQ trajectories from different trials

**Output format**: PNG images saved alongside checkpoint files

### CPU Architecture Adaptation

**Addition**: Automatic detection and optimization for target hardware

**Detection capabilities**:
- Physical core count vs logical cores (hyperthreading detection)
- Hybrid CPU architecture (Intel P-cores + E-cores like Core Ultra)
- Available memory (adapt batch size to prevent OOM)
- SIMD capabilities (AVX-512, AVX2, ARM NEON)
- Platform-specific optimizations (Linux taskset, macOS affinity)

**Optimizations**:
- Auto-select optimal worker count (physical cores - 2)
- Pin compute-heavy trials to P-cores on hybrid CPUs
- Memory-aware iteration limits (reduce on low-RAM systems)
- Vectorized operations on AVX-512 capable CPUs
- Platform-specific thread affinity

**Example adaptations**:
- 4-core laptop: 2 trials, 50K iterations (reduced for memory)
- 8-core desktop: 6 trials, 100K iterations
- Core Ultra 7 265K (8P+12E): 8 trials pinned to P-cores
- 64-core server: 60+ trials, full iterations

### Distributed Execution (Fly.io)

**Addition**: Optional cloud-based massive parallelism for production-quality patterns

**Architecture**:
- Coordinator machine spawns N worker machines on Fly.io
- Each worker generates 1 trial independently (seed-based)
- Workers upload results to Tigris storage
- Coordinator collects results, selects best, generates final output

**Cost-benefit**:
- 32 workers × 24 hours: **$6** → +1.5 dB quality improvement
- 64 workers × 24 hours: **$12** → +2.1 dB quality improvement
- vs Local 8 trials: Free but lower quality (-39 dB vs -40.5 dB)

**Use case**: One-time infrastructure generation where $6-12 investment yields permanently better patterns for all CASCADE deployments

---

## Requirements

### Functional Requirements

#### 64-Pattern Set Generation
- **FR-001**: System MUST generate 64 patterns total: 48 beacon (IDs 0-47) + 16 message (IDs 48-63)
- **FR-002**: Beacon patterns (0-47) MUST have simple IQ complexity (λ max 0.3) optimized for robustness
- **FR-003**: Message patterns (48-63) MUST have minimal IQ complexity (λ=0.0-0.1) for emergency/disturbed conditions
- **FR-004**: All 64 pattern pairs MUST achieve <-37.5 dB cross-correlation in 4D space
- **FR-005**: System MUST output cascade_patterns_64.bin (approximately 19 KB) in CASCADE binary format

#### 128-Pattern Set Generation
- **FR-006**: System MUST generate 128 patterns total: 48 beacon (IDs 0-47) + 80 message (IDs 48-127)
- **FR-007**: Beacon patterns (0-47) MUST match 64-pattern set exactly (same patterns for consistency)
- **FR-008**: Message patterns MUST be organized in 4 hierarchical IQ pools:
  - 48-63: Emergency (16 patterns, λ=0.0-0.1)
  - 64-95: Typical DX (32 patterns, λ=0.3-0.5) - most common pool
  - 96-111: Good propagation (16 patterns, λ=0.5-0.7)
  - 112-127: NVIS exceptional (16 patterns, λ=0.7-0.9)
- **FR-009**: All 128 pattern pairs MUST achieve <-37.5 dB cross-correlation in 4D space
- **FR-010**: System MUST output cascade_patterns_128.bin (approximately 38 KB) in CASCADE binary format

#### Pattern Structure
- **FR-011**: Each pattern MUST have 32-symbol tone index sequence (indices 0-3 for 4-tone selection)
- **FR-012**: Each pattern MUST have single IQ trajectory (complex64, 32 values) with baked-in complexity
- **FR-013**: Patterns MUST use Zadoff-Chu sequences as base for initial orthogonality
- **FR-014**: System MUST apply simulated annealing optimization to achieve -37.5 dB target
- **FR-015**: Pattern storage MUST follow CASCADE binary format (metadata + freq_sequence + iq_trajectory + checksum)

#### Validation
- **FR-016**: System MUST validate each pattern pair computes <-37.5 dB cross-correlation
- **FR-017**: System MUST test patterns under phase distortion (±180° random phase/tone)
- **FR-018**: System MUST report both ideal and phase-robust separation metrics
- **FR-019**: System MUST validate binary file format integrity (magic bytes, checksums)
- **FR-020**: System MUST generate validation report showing min/max/mean correlation across all pairs

#### Visualization
- **FR-021**: System MUST generate IQ trajectory plots after each trial batch
- **FR-022**: System MUST generate λ distribution histogram showing complexity patterns
- **FR-023**: System MUST generate correlation matrix heatmap for pattern orthogonality
- **FR-024**: System MUST save visualizations as PNG files alongside checkpoints

#### Platform Adaptation
- **FR-025**: System MUST auto-detect CPU architecture (physical cores, hybrid P/E cores, memory)
- **FR-026**: System MUST adapt worker count based on available resources
- **FR-027**: System SHOULD pin trials to P-cores on hybrid CPU architectures
- **FR-028**: System MUST reduce iterations on low-memory systems (<8 GB available)
- **FR-029**: System SHOULD support manual override of auto-detected settings

#### Distributed Execution
- **FR-030**: System SHOULD support distributed execution on Fly.io (optional)
- **FR-031**: Coordinator MUST spawn configurable number of worker machines
- **FR-032**: Workers MUST upload trial results to Tigris storage
- **FR-033**: Coordinator MUST collect results and select best pattern set
- **FR-034**: System MUST support hybrid local+cloud execution

#### Two-Phase Optimization
- **FR-035**: System MUST support two-phase optimization (frequency-first, then IQ)
- **FR-036**: Phase 1 MUST optimize frequency with BPSK (λ=0) for 80% of iteration budget
- **FR-037**: Phase 2 MUST add IQ complexity only if Phase 1 fails to achieve target
- **FR-038**: System MUST report which patterns achieved λ=0 (BPSK sufficient)

#### Phase-Aware Optimization
- **FR-039**: System SHOULD use phase-aware cost function by default
- **FR-040**: Cost function MUST test correlation under 3-5 random phase scenarios
- **FR-041**: Phase scenarios MUST include ±π tone-dependent and ±0.2 time-varying phase
- **FR-042**: System MUST use worst-case correlation across phase scenarios for cost

### Non-Functional Requirements

#### Performance
- **NFR-001**: Single trial (400K iterations) SHOULD complete within 72-96 hours on high-end 8-core CPU
- **NFR-002**: Optimal local config (8 trials × 400K) SHOULD complete within 72-96 hours
- **NFR-003**: Cloud breadth config (32 workers × 100K) SHOULD complete within 30-40 hours
- **NFR-004**: Pattern validation MUST complete within 5 minutes
- **NFR-005**: Memory usage per trial MUST stay under 500 MB RAM
- **NFR-006**: Visualization generation MUST complete within 30 seconds per batch
- **NFR-007**: System MUST adapt to available resources (2-64+ cores, 4-256 GB RAM)
- **NFR-013**: Two-phase optimization SHOULD achieve 20-30% patterns at λ=0

#### Quality
- **NFR-008**: Pattern generation MUST be deterministic with fixed random seed
- **NFR-009**: All pattern pairs MUST achieve orthogonality <-37.5 dB (no exceptions)
- **NFR-010**: IQ complexity (λ) MUST be empirically minimized (not pre-assigned)
- **NFR-011**: Binary output files MUST be loadable by standard Python file I/O
- **NFR-012**: Distributed execution MUST produce identical results to local (given same seed)
- **NFR-014**: Optimal config (8×400K) SHOULD achieve -42.6 dB separation, λ=0.17 average
- **NFR-015**: Phase-aware optimization SHOULD maintain phase-robust separation within 6 dB of ideal

### Key Entities

**PatternSet**: Collection of patterns (64 or 128)
- Attributes: pattern_count (64 or 128), beacon_count (48), message_count (16 or 80), target_orthogonality_db (-37.5)
- Relationships: Contains multiple Pattern objects

**Pattern**: Single 4D orthogonal pattern
- Attributes: pattern_id (0-127), freq_sequence (32 tone indices 0-3), iq_trajectory (32 complex values), iq_complexity_lambda (0.0-0.9), pattern_type (beacon or message), complexity_pool (emergency, typical_dx, good_prop, nvis)
- Relationships: Part of PatternSet, validated against all other patterns

**PatternPair**: Two patterns for correlation testing
- Attributes: pattern_i_id, pattern_j_id, correlation_db, passes_threshold (bool)
- Relationships: Validates orthogonality between two Pattern objects

**ValidationReport**: Results of pattern validation
- Attributes: pattern_set_size (64 or 128), min_correlation_db, max_correlation_db, mean_correlation_db, failed_pairs (list), overall_pass (bool)
- Relationships: Generated from PatternSet validation

**BinaryPatternFile**: Output file format
- Attributes: magic_bytes (b'CASC'), version (2), pattern_count, file_size_bytes, checksum
- Relationships: Serialized form of PatternSet

---

## Review & Acceptance Checklist

### Content Quality
- [x] No implementation details (languages, frameworks, APIs) - describes WHAT to generate, not HOW
- [x] Focused on user value (prerequisite for all CASCADE features)
- [x] Written for non-technical stakeholders (pattern specs as requirements)
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain (all parameters specified)
- [x] Requirements are testable (-37.5 dB measurable, file sizes verifiable)
- [x] Success criteria are measurable (orthogonality threshold)
- [x] Scope is clearly bounded (two specific pattern sets)
- [x] Dependencies identified (none - this is foundational)

---

## Execution Status

- [x] User description parsed (generate 64 and 128 pattern sets)
- [x] Key concepts extracted (Zadoff-Chu, simulated annealing, -37.5 dB orthogonality, hierarchical IQ)
- [x] Ambiguities marked (none - detailed specification provided)
- [x] User scenarios defined (implementer generating patterns for vetting and production)
- [x] Requirements generated (20 functional, 8 non-functional)
- [x] Entities identified (PatternSet, Pattern, PatternPair, ValidationReport, BinaryPatternFile)
- [x] Review checklist passed

---

## Additional Context

### Reference Documentation
- **Pattern specification**: docs/model/pattern_architecture.md
- **Generation algorithm**: docs/implementation/pattern_generation_spec.md (714 lines)
- **Architecture overview**: docs/architecture.md
- **Binary format**: docs/model/pattern_architecture.md (File Specification section)

### 64 vs 128 Pattern Comparison

**64-pattern set (testing)**:
- Faster generation: 8-12 hours vs 18-24 hours
- Smaller file: 19 KB vs 38 KB
- Easier optimization: Fewer pairs to satisfy (2,016 vs 8,128 pairs)
- Lower capacity: ~512 total users vs 1,024
- Use case: Phase 0 vetting, rapid architecture validation

**128-pattern set (production)**:
- Optimal performance: 218 bps/user vs ~210 bps with 64 patterns
- Full capacity: 1,024 total users, 45 active
- Better chaos tolerance: More pattern diversity
- Longer pools: 32-pattern Typical DX pool vs 16 patterns
- Use case: Production deployment, full performance

### Constitutional Alignment

This feature satisfies Constitution Principle I (Data-First Development) as the prerequisite for all training and protocol work. Patterns are fundamental infrastructure that must exist before model training can begin.
