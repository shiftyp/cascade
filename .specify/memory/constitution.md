<!-- Sync Impact Report
Version change: 1.0.0 (initial)
Modified principles: N/A (initial creation)
Added sections: All sections (initial document)
Removed sections: N/A
Templates requiring updates:
  ✅ spec-template.md - updated with module context section
  ✅ plan-template.md - updated with monorepo structure and constitution gates
  ✅ tasks-template.md - updated with module path conventions
  ✅ agent-file-template.md - checked (no changes needed)
Follow-up TODOs: None
Date: 2025-09-29
-->

# CASCADE Constitution

## Core Principles

### I. Data-First Development
All development begins with data gathering and processing. Real-world atmospheric noise (QRN) and propagation data must be collected before model training. Model training must precede protocol implementation. Protocol must exist before applications. This sequence is mandatory and ensures empirical validation at each stage.

### II. Monorepo Module Architecture
The project must be organized as a monorepo with distinct modules: data (gathering/processing scripts), training (model development), protocol (modem implementation), and applications (user-facing tools). Each module must be independently buildable with clear interfaces between modules. No module may bypass its dependencies in the development sequence.

### III. Clean Separation of Concerns
Protocol and model layers must remain strictly separated. Protocol handles discrete decisions (WHO, WHETHER, WHAT identities and routing), while model handles continuous optimization (HOW, HOW MUCH encoding and patterns). No layer shall cross into the other's domain. This separation ensures verifiability, maintainability, and comprehensibility.

### IV. Test-Driven Development
TDD is mandatory. Tests must be written first, fail appropriately, then implementation follows to make them pass. Red-Green-Refactor cycle must be strictly enforced. Data processing requires validation tests. Model training requires performance benchmarks. Protocol requires contract tests. Applications require integration tests.

### V. Real-World Data Priority
System training and validation must use actual atmospheric noise and propagation data from real transmissions. Synthetic models are prohibited for core functionality. Initial specs must focus on data collection scripts for WebSDR recordings and FT8/WSPR transmission extraction. Performance claims must be validated against real-world conditions.

### VI. Privacy-Preserving Design
All telemetry and learning must implement differential privacy (ε≤1.0). No personally identifiable information may be collected or transmitted. Data gathering scripts must anonymize callsigns and locations. Federated learning must use Byzantine-robust aggregation. User data sovereignty must be absolute.

### VII. Reproducible Research Standards
All data gathering must be scriptable and reproducible. Processing pipelines must be deterministic. Training procedures must include random seeds and versioned datasets. Model checkpoints must be tracked with metrics. Results must be independently verifiable.

## Module Structure

### Data Module Requirements
- Scripts for WebSDR recording collection
- QRN/QRM extraction and categorization tools
- Propagation data parsing from FT8/WSPR
- Data validation and quality metrics
- Storage format specifications

### Training Module Requirements
- Dataset versioning and management
- Model architecture definitions
- Training pipeline automation
- Performance benchmarking suite
- Checkpoint management

### Protocol Module Requirements
- Clean API between protocol and model layers
- Message encoding/decoding libraries
- Beacon system implementation
- Priority handling mechanisms
- Link adaptation algorithms

### Applications Module Requirements
- CLI interfaces for all functionality
- Text I/O protocol (stdin/args → stdout)
- Support JSON and human-readable formats
- Integration with protocol module only

## Development Workflow

### Specification Sequence
1. Data gathering and processing specs (FIRST PRIORITY)
2. Model training pipeline specs
3. Protocol implementation specs
4. Application development specs

### Module Dependencies
- Applications depend on Protocol
- Protocol depends on Training
- Training depends on Data
- No circular dependencies permitted
- No skipping dependency chain

## Governance

### Amendment Procedure
Constitutional amendments require:
1. Documented rationale with problem statement
2. Impact analysis on module boundaries
3. Migration plan for affected modules
4. Approval through pull request review
5. Version increment per semantic versioning

### Versioning Policy
- MAJOR: Removing principles or changing module structure
- MINOR: Adding principles or expanding requirements
- PATCH: Clarifications and non-semantic improvements

### Compliance Review
All pull requests must verify constitutional compliance. Module boundaries must be respected. Development sequence must be maintained. Use `.specify/memory/constitution.md` as authoritative reference. Automated gates enforce principle adherence during CI/CD.

**Version**: 1.0.0 | **Ratified**: 2025-09-29 | **Last Amended**: 2025-09-29