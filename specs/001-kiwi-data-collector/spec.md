# Feature Specification: KiwiSDR Data Collector

**Feature Branch**: `001-kiwi-data-collector`
**Created**: 2025-09-29
**Status**: Draft
**Input**: User description: "kiwi-data-collector"

## Execution Flow (main)
```
1. Parse user description from Input
   → If empty: ERROR "No feature description provided"
2. Extract key concepts from description
   → Identify: actors, actions, data, constraints
3. For each unclear aspect:
   → Mark with [NEEDS CLARIFICATION: specific question]
4. Fill User Scenarios & Testing section
   → If no clear user flow: ERROR "Cannot determine user scenarios"
5. Generate Functional Requirements
   → Each requirement must be testable
   → Mark ambiguous requirements
6. Identify Key Entities (if data involved)
7. Run Review Checklist
   → If any [NEEDS CLARIFICATION]: WARN "Spec has uncertainties"
   → If implementation details found: ERROR "Remove tech details"
8. Return: SUCCESS (spec ready for planning)
```

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no tech stack, APIs, code structure)
- 👥 Written for business stakeholders, not developers

### Section Requirements
- **Mandatory sections**: Must be completed for every feature
- **Optional sections**: Include only when relevant to the feature
- When a section doesn't apply, remove it entirely (don't leave as "N/A")

### For AI Generation
When creating this spec from a user prompt:
1. **Mark all ambiguities**: Use [NEEDS CLARIFICATION: specific question] for any assumption you'd need to make
2. **Don't guess**: If the prompt doesn't specify something (e.g., "login system" without auth method), mark it
3. **Think like a tester**: Every vague requirement should fail the "testable and unambiguous" checklist item
4. **Common underspecified areas**:
   - Module placement (data/training/protocol/applications)
   - Data privacy and anonymization requirements
   - Real-world vs synthetic data sources
   - Performance targets and scale
   - Error handling behaviors
   - Integration requirements
   - Security/compliance needs

---

## Clarifications

### Session 2025-09-29
- Q: What is the target collection timeline for the 100-500 hours of QRN data? → A: seasonal cycle geographic diversity
- Q: What is the primary collection strategy for the 18-month period? → A: all (continuous baseline + event-triggered + FT8/WSPR focused)
- Q: What are the storage requirements and retention policies? → A: 35-75TB total for 150,000-300,000 hours of compressed data
- Q: What is the recording bandwidth and format specification? → A: 12 kHz IQ, 16-bit, 12kHz sample rate, FLAC compressed
- Q: How many simultaneous KiwiSDR stations for baseline collection? → A: Variable (6 quiet, 20+ active)
- Q: What is the data quality validation threshold? → A: Keep all data (complete archive)

## Module Context *(mandatory for CASCADE)*

### Target Module
Data Module - This is the foundational data collection component

### Module Dependencies
None - This is the first module in the CASCADE dependency chain

### Module Interfaces
Outputs to Training Module: Processed QRN recordings, FT8/WSPR propagation data, metadata files

## User Scenarios & Testing *(mandatory)*

### Primary User Story
As a CASCADE developer, I need to collect real-world atmospheric noise (QRN) and propagation data from KiwiSDR receivers worldwide, so that the neural network models can be trained on actual radio conditions rather than synthetic data.

### Acceptance Scenarios
1. **Given** a list of public KiwiSDR receivers, **When** the collector is started with QRN collection parameters, **Then** it records atmospheric noise from multiple geographic locations with GPS timestamps
2. **Given** active amateur radio bands, **When** the collector monitors FT8/WSPR frequencies, **Then** it extracts propagation characteristics without storing message content
3. **Given** collected raw audio data, **When** privacy processing is applied, **Then** all callsigns and personally identifiable information are anonymized
4. **Given** a recording schedule, **When** the collector runs unattended, **Then** it respects receiver usage limits and automatically rotates between sources

### Edge Cases & Error Handling
- When a KiwiSDR goes offline during recording: Save partial data, mark as incomplete, retry with another SDR (FR-009, FR-032)
- When daily usage limits are reached: Rotate to next available SDR, track usage per SDR (FR-008, FR-014)
- When network connectivity is interrupted: Buffer locally for up to 1 hour, retry transmission when restored (new task needed)
- When recordings are corrupted: Log error, save for debugging, continue collection (FR-019)
- When all SDRs unavailable: Enter standby mode, retry every 5 minutes, alert operators immediately (FR-032, FR-033, FR-034, FR-035)

## Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: System MUST connect to public KiwiSDR receivers programmatically
- **FR-002**: System MUST record atmospheric noise (QRN) from HF bands (10 kHz - 30 MHz)
- **FR-003**: System MUST collect MINIMUM 10,000 hours per HF band (60,000 hours baseline) with expanded event-driven collection targeting 150,000-300,000 total hours over 18 months from geographically and temporally diverse locations
- **FR-004**: System MUST extract propagation data from FT8 and WSPR transmissions
- **FR-005**: System MUST anonymize all callsigns and location information
- **FR-006**: System MUST preserve GPS timestamps for all recordings
- **FR-007**: System MUST store audio in FLAC format (lossless compressed) with 12 kHz IQ, 16-bit depth, 12kHz sample rate
- **FR-008**: System MUST rotate between multiple KiwiSDR sources to avoid overuse
- **FR-009**: System MUST handle receiver disconnections gracefully with automatic retry
- **FR-010**: System MUST log all collection activities for audit purposes
- **FR-011**: Users MUST be able to configure recording duration and schedule via configuration files and CLI parameters
- **FR-012**: Users MUST be able to specify frequency ranges and modes through configuration files
- **FR-013**: System MUST validate data quality before storage (GPS lock present, sample rate within ±1%, no gaps >1 second, SNR measurement capability down to -30dB)
- **FR-014**: System MUST respect public receiver usage policies (daily limits)
- **FR-015**: System MUST organize recordings by date, frequency, and location metadata
- **FR-016**: System MUST implement continuous baseline collection using 6-10 simultaneous SDRs with intelligent rotation for propagation cycle coverage (diurnal, geographic, seasonal), scaling to 20-50 SDRs during events
- **FR-017**: System MUST maintain expandable storage capacity starting at 35TB, scaling to 75TB+ for 150,000-300,000 hours of compressed IQ data
- **FR-018**: System MUST support variable station count (6-10 baseline, 20+ during geomagnetic storms, 50+ during rare events)
- **FR-019**: System MUST retain all collected data regardless of quality for complete archive (validation from FR-013 logs issues but does not reject data)
- **FR-020**: System MUST capture 12 kHz windows centered on FT8 frequencies to include both signals and adjacent quiet zones
- **FR-021**: System MUST record at these specific center frequencies to maximize quiet spectrum coverage while capturing propagation indicators:
  - 80m: 3576 kHz (captures WSPR 3568.6, FT8 3573, quiet zone 3576-3582)
  - 40m: 7080 kHz (captures FT8 7074, quiet digital sub-band 7078-7086)
  - 20m: 14080 kHz (captures FT8 14074, quiet zone 14078-14086)
  - 15m: 21080 kHz (captures FT8 21074, quiet zone 21078-21086)
  - 10m: 28080 kHz (captures FT8 28074, quiet zone 28078-28086)
  - 6m: 50303 kHz (captures WSPR 50293, quiet zone 50297-50309)
- **FR-022**: System MUST implement propagation-aware SDR rotation to ensure coverage of:
  - All 24 UTC hours for each geographic region
  - Day/night terminator crossings (gray-line propagation)
  - Auroral zones during geomagnetic disturbances
  - Trans-equatorial paths during equinoxes
  - Long-path vs short-path variations
  - Seasonal ionospheric changes
- **FR-023**: System MUST dynamically scale SDR collection for propagation events:
  - Gray-line enhancement: Scale to 15+ SDRs along terminator
  - Geomagnetic activity (K≥3): Scale to 12+ SDRs
  - Meteor showers: Scale 6m coverage to 15+ SDRs
  - Sporadic-E season: Increase mid-latitude SDRs
  - Amateur radio contests: Scale to 20+ SDRs for QRM analysis
  - Solar flux changes >10 units/day: Increase coverage
  - Tropical storms/atmospheric fronts: Regional density increase
- **FR-024**: System MUST prioritize "interesting" over "routine" when at capacity, always preferring rare propagation events, unusual band openings, and unique geographic paths
- **FR-025**: System MUST extract and label propagation mode for each recording (F2, Aurora, Sporadic-E, TEP, meteor scatter) to support diversity-biased training dataset creation
- **FR-026**: System MUST identify and extract quiet periods (no signals present) within recordings for noise characterization, maintaining temporal correlation with propagation data from same recording
- **FR-027**: System MUST capture X-ray flux classification from NOAA space weather (X, M, C, B, A class flares) for rarity scoring during dataset curation
- **FR-028**: System MUST maintain correlation links between noise samples and propagation records from the same recording session to preserve natural channel correlations
- **FR-029**: System MUST support extraction of overlapping 2.5 kHz channels (9 channels with 50% overlap) from each 12 kHz recording for frequency-dependent noise characterization
- **FR-030**: System MUST preserve transmitter and receiver grid squares from decoded FT8/WSPR messages to enable path geometry calculations and geographic context for propagation characterization
- **FR-031**: System MUST use FLAC compression to achieve 45-55% size reduction while maintaining lossless quality for IQ data storage
- **FR-032**: System MUST implement graceful degradation when SDRs become unavailable, maintaining operation with as few as 1 SDR while prioritizing bands by propagation likelihood
- **FR-033**: System MUST continuously monitor SDR availability and automatically attempt reconnection to offline SDRs every 5 minutes
- **FR-034**: System MUST alert operators when available SDRs drop below 50% of target count or when all SDRs for a specific band are unavailable
- **FR-035**: System MUST maintain a minimum dataset collection even with only 1 SDR by cycling through all 6 bands on a 10-minute rotation
- **FR-036**: System MUST implement QA sampling by storing 1% of collected data (randomly selected) in hot object storage for manual verification, with weekly rotation
- **FR-037**: System MUST generate daily QA reports listing sampled files with metadata (timestamp, SDR source, band, SNR, file size) for review
- **FR-038**: System MUST flag and quarantine recordings that fail automated quality checks for manual inspection before archival or deletion
- **FR-039**: System MUST coordinate distributed collection across 2-10 Fly.io worker machines using Redis/KeyDB message queue
- **FR-040**: System MUST implement distributed locks to prevent multiple workers from claiming the same KiwiSDR simultaneously
- **FR-041**: System MUST auto-scale worker machines from 2 (baseline) to 10 (during propagation events) based on SDR queue depth and CPU metrics
- **FR-042**: System MUST maintain centralized scheduler process that monitors propagation conditions and publishes SDR assignments to Redis queue
- **FR-043**: Workers MUST report health status to Redis every 30 seconds and gracefully complete recordings before shutdown
- **FR-044**: Dashboard MUST provide waterfall (spectrogram) visualization for QA samples stored in hot storage
- **FR-045**: Dashboard MUST support IQ sample replay with time/frequency selection, zoom, and measurement cursors
- **FR-046**: Waterfall display MUST show signal strength using color mapping with configurable dynamic range (minimum 60dB)
- **FR-047**: Dashboard MUST provide searchable interface for QA samples with filters for: date/time range, frequency band, SDR source, propagation mode, SNR range, space weather conditions (K-index, X-ray class)
- **FR-048**: QA sample search MUST support sorting by: timestamp, SNR, file size, review status, propagation event type, rarity score
- **FR-049**: Dashboard MUST display QA sample metadata alongside waterfall including: correlation ID, linked FT8/WSPR detections, quiet period markers, QRN characteristics, space weather at collection time

### Key Entities *(include if feature involves data)*
- **Recording Session**: Represents a single data collection session with start/end times, source receiver, frequency, mode, quality metrics, and correlation ID for linking related samples
- **KiwiSDR Source**: Represents a public receiver with URL, location (anonymized), available frequency ranges, and usage statistics
- **QRN Sample**: Atmospheric noise recording with timestamp, frequency, bandwidth, signal characteristics, geographic region, quiet period markers, and correlation ID
- **Propagation Record**: Extracted FT8/WSPR data showing signal strength, path distance, frequency, time, detected propagation mode, confidence score, and correlation ID (no message content)
- **SpaceWeatherData**: Solar and geomagnetic conditions including K-index, solar flux, X-ray class and flux values for event correlation
- **Collection Schedule**: Defines automated recording times, durations, rotation patterns, and target data volumes
- **QA Sample**: Quality assurance sample with hot storage path, review status, quality metrics, and reviewer notes
- **NotificationConfig**: Alert configuration for SDR availability and error conditions

---

## Review & Acceptance Checklist
*GATE: Automated checks run during main() execution*

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [ ] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

---

## Execution Status
*Updated by main() during processing*

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [ ] Review checklist passed

---