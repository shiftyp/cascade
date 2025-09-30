# Tasks: KiwiSDR Data Collector

**Input**: Design documents from `/specs/001-kiwi-data-collector/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory ✓
   → Tech stack: Python 3.11, kiwiclient, numpy, scipy, PostgreSQL
2. Load optional design documents ✓
   → data-model.md: 7 entities + notification tables + correlation support
   → contracts/: collector_api.yaml, processor_api.yaml
   → quickstart.md: Test scenarios for collection pipeline
3. Generate tasks by category ✓
4. Apply task rules ✓
5. Number tasks sequentially ✓ (96 tasks total)
6. Generate dependency graph ✓
7. Create parallel execution examples ✓
8. Validate task completeness ✓
9. Return: SUCCESS (tasks ready for execution)
```

**Note**: Expanded scope to 200,000-250,000 hours with hybrid KiwiSDR/WebSDR collection strategy and event-driven scaling

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Phase 3.1: Setup
- [x] T001 Create project structure in modules/data/ per plan.md
- [x] T002 Initialize Python 3.11 project with requirements.txt dependencies (including redis-py)
- [x] T003 [P] Configure pytest and ruff for linting
- [x] T004 [P] Setup PostgreSQL database and run initial schema migration
- [x] T005 Configure Fly.io deployment files (fly.toml, Dockerfile) with auto-scaling
- [x] T005a Configure center frequencies per FR-021 in modules/data/src/config/frequencies.py
- [x] T005b [P] Setup Redis/KeyDB instance on Fly.io for distributed coordination
- [x] T005c [P] Configure Fly.io machines (scheduler, workers) in fly.toml
- [x] T005d [P] Setup Tigris S3 bucket configuration for FLAC storage

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### Contract Tests
- [x] T006 [P] Contract test POST /collectors/connect in modules/data/tests/contract/test_collector_connect.py
- [x] T007 [P] Contract test POST /collectors/start in modules/data/tests/contract/test_collector_start.py
- [x] T008 [P] Contract test POST /collectors/stop in modules/data/tests/contract/test_collector_stop.py
- [x] T009 [P] Contract test GET /collectors/status in modules/data/tests/contract/test_collector_status.py
- [x] T010 [P] Contract test POST /processors/decode/ft8 in modules/data/tests/contract/test_processor_ft8.py
- [x] T011 [P] Contract test POST /processors/decode/wspr in modules/data/tests/contract/test_processor_wspr.py
- [x] T012 [P] Contract test POST /processors/analyze/qrn in modules/data/tests/contract/test_processor_qrn.py

### Integration Tests
- [x] T013 [P] Integration test KiwiSDR connection rotation in modules/data/tests/integration/test_sdr_rotation.py
- [x] T013b [P] Integration test WebSDR connection and session management in modules/data/tests/integration/test_websdr_connection.py
- [x] T013c [P] Integration test hybrid SDR selection algorithm in modules/data/tests/integration/test_hybrid_sdr_selection.py
- [x] T014 [P] Integration test recording pipeline in modules/data/tests/integration/test_recording_pipeline.py
- [x] T015 [P] Integration test FT8/WSPR extraction in modules/data/tests/integration/test_signal_extraction.py
- [x] T016 [P] Integration test space weather trigger in modules/data/tests/integration/test_space_weather.py
- [x] T017 [P] Integration test Tigris storage upload in modules/data/tests/integration/test_storage_upload.py
- [x] T017a [P] Integration test correlation preservation in modules/data/tests/integration/test_correlation_preservation.py
- [x] T017b [P] Integration test multi-channel extraction in modules/data/tests/integration/test_multichannel_extraction.py
- [x] T017c [P] Integration test graceful degradation in modules/data/tests/integration/test_graceful_degradation.py
- [x] T017d [P] Integration test all-SDR failure recovery in modules/data/tests/integration/test_total_failure_recovery.py
- [x] T017e [P] Integration test Redis queue operations in modules/data/tests/integration/test_redis_queue.py
- [x] T017f [P] Integration test distributed worker coordination in modules/data/tests/integration/test_worker_coordination.py
- [x] T017g [P] Integration test Fly.io auto-scaling in modules/data/tests/integration/test_fly_autoscale.py

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Database Models
- [x] T018 [P] RecordingSession model with correlation_id in modules/data/src/models/recording_session.py
- [x] T019 [P] KiwiSDRSource model in modules/data/src/models/kiwisdr_source.py
- [x] T020 [P] QRNSample model with quiet_periods and correlation_id in modules/data/src/models/qrn_sample.py
- [x] T021 [P] PropagationRecord model with propagation_mode and correlation_id in modules/data/src/models/propagation_record.py
- [x] T022 [P] SpaceWeatherData model with xray_class and xray_flux in modules/data/src/models/space_weather_data.py
- [x] T023 [P] CollectionSchedule model in modules/data/src/models/collection_schedule.py
- [x] T024 [P] NotificationConfig model in modules/data/src/models/notification_config.py
- [x] T024a [P] CollectionAlerts model in modules/data/src/models/collection_alerts.py
- [x] T024b [P] NotificationTemplates model in modules/data/src/models/notification_templates.py

### Collector Services
- [x] T025 [P] KiwiClient connection manager (implements FR-001, FR-008, FR-009) in modules/data/src/collectors/kiwi_client.py
- [x] T025b [P] WebSDR client connection manager (implements FR-001, FR-065, FR-066) in modules/data/src/collectors/websdr_client.py
- [x] T026 [P] Recorder orchestrator (implements FR-002, FR-007, FR-015) in modules/data/src/collectors/recorder.py
- [x] T027 Collection scheduler (implements FR-011, FR-016, FR-042) in modules/data/src/collectors/scheduler.py
- [x] T027a [P] Distributed worker process in modules/data/src/collectors/worker.py
- [x] T027b [P] Redis queue manager in modules/data/src/collectors/queue_manager.py
- [x] T027c [P] Distributed lock manager for SDR claims in modules/data/src/collectors/lock_manager.py
- [x] T028 [P] Propagation-aware SDR rotation algorithm (implements FR-008, FR-014, FR-022) in modules/data/src/collectors/sdr_manager.py
- [x] T028e [P] Hybrid SDR selection algorithm (implements FR-067) in modules/data/src/collectors/hybrid_sdr_selector.py
- [x] T028a Event-based SDR scaling logic (implements FR-023, FR-024, FR-041) in modules/data/src/collectors/event_scaler.py
- [x] T028b [P] Graceful degradation handler (implements FR-032) in modules/data/src/collectors/degradation_handler.py
- [x] T028c [P] SDR health monitor with auto-reconnect (implements FR-033) in modules/data/src/collectors/health_monitor.py
- [x] T028d [P] Minimum collection scheduler for 1-SDR mode (implements FR-035) in modules/data/src/collectors/minimum_scheduler.py

### Processor Services
- [x] T029 [P] FT8 decoder with propagation mode detection (implements FR-004, FR-025, FR-030) in modules/data/src/processors/ft8_decoder.py
- [x] T030 [P] WSPR decoder with propagation mode detection (implements FR-004, FR-025, FR-030) in modules/data/src/processors/wspr_decoder.py
- [x] T031 [P] QRN analyzer with quiet period detection in modules/data/src/processors/qrn_analyzer.py
- [x] T032 [P] Callsign anonymizer (implements FR-005, FR-006) in modules/data/src/processors/anonymizer.py
- [x] T032a [P] Multi-channel QRN extractor (9x 2.5kHz overlapping) in modules/data/src/processors/multichannel_qrn.py
- [x] T032b [P] Propagation mode classifier in modules/data/src/processors/mode_classifier.py
- [x] T032c [P] Path context calculator for propagation geometry in modules/data/src/processors/path_context.py

### Storage Services
- [x] T033 [P] FLAC compression utility (implements FR-007, FR-031) in modules/data/src/storage/compression.py
- [x] T034 [P] File manager for FLAC in modules/data/src/storage/file_manager.py
- [x] T035 [P] PostgreSQL metadata interface in modules/data/src/storage/metadata_db.py
- [x] T036 [P] Tigris S3 storage client in modules/data/src/storage/tigris_storage.py

### Validators
- [x] T037 [P] Quality validation (implements FR-013, FR-019) in modules/data/src/validators/quality_check.py
- [x] T038 [P] Coverage validation in modules/data/src/validators/coverage_check.py
- [x] T038a [P] QA sampler for 1% hot storage (implements FR-036) in modules/data/src/validators/qa_sampler.py
- [x] T038b [P] QA report generator (implements FR-037) in modules/data/src/validators/qa_reporter.py
- [x] T038c [P] Quarantine manager for failed QC (implements FR-038) in modules/data/src/validators/quarantine_manager.py

### API Endpoints
- [x] T039 POST /collectors/connect endpoint in modules/data/src/api/collectors.py
- [x] T040 POST /collectors/start endpoint in modules/data/src/api/collectors.py
- [x] T041 POST /collectors/stop endpoint in modules/data/src/api/collectors.py
- [x] T042 GET /collectors/status endpoint in modules/data/src/api/collectors.py
- [x] T043 POST /processors/decode/ft8 endpoint in modules/data/src/api/processors.py
- [x] T044 POST /processors/decode/wspr endpoint in modules/data/src/api/processors.py
- [x] T045 POST /processors/analyze/qrn endpoint in modules/data/src/api/processors.py

## Phase 3.4: Integration

### External Services
- [x] T046 NOAA space weather API client with X-ray data in modules/data/src/external/noaa_client.py
- [x] T047 Gmail notification service in modules/data/src/notifications/gmail_notifier.py
- [x] T047a [P] Correlation manager for paired samples in modules/data/src/processors/correlation_manager.py

### Dashboard & Monitoring
- [x] T049 Dashboard SQL views in modules/data/src/dashboard/views.sql
- [x] T050 Terminal dashboard script in modules/data/src/dashboard/terminal_dashboard.py
- [x] T051 FastAPI dashboard web app with QA waterfall viewer (implements FR-044, FR-045, FR-046) in modules/data/src/dashboard/web_dashboard.py
- [ ] T051a [P] Waterfall generator for IQ samples (implements FR-044) in modules/data/src/dashboard/waterfall_generator.py
- [ ] T051b [P] IQ file streaming reader for FLAC samples in modules/data/src/dashboard/iq_reader.py
- [ ] T051c [P] Frontend waterfall display component (implements FR-045, FR-046) in modules/data/src/dashboard/static/waterfall.js
- [ ] T051d [P] QA sample search API endpoints (implements FR-047, FR-048) in modules/data/src/dashboard/qa_search.py
- [ ] T051e [P] QA metadata aggregator (implements FR-049) in modules/data/src/dashboard/qa_metadata.py
- [ ] T051f [P] Frontend search interface with filters (implements FR-047, FR-048) in modules/data/src/dashboard/static/qa_search.js

### System Integration
- [ ] T052 Connect all services to PostgreSQL database (Note: Models can be developed without live DB)
- [ ] T053 Configure multi-source event detection system (NOAA, ephemeris, weather)
- [ ] T053a Implement propagation event triggers in modules/data/src/events/trigger_manager.py
- [ ] T053b Gray-line calculator in modules/data/src/events/grayline.py
- [ ] T053c Contest calendar integration in modules/data/src/events/contest_calendar.py
- [ ] T054 Setup alert triggers and thresholds (including FR-034 SDR availability alerts)
- [ ] T055 Configure Tigris lifecycle policies
- [ ] T056 Implement retry logic for failed recordings
- [ ] T056a Implement correlation preservation pipeline in modules/data/src/processors/correlation_pipeline.py
- [ ] T056b Configure rarity scoring system for dataset curation (implements FR-024 prioritization) in modules/data/src/curation/rarity_scorer.py

## Phase 3.5: Polish

### Unit Tests
- [ ] T057 [P] Unit tests for KiwiClient in modules/data/tests/unit/test_kiwi_client.py
- [ ] T058 [P] Unit tests for FLAC compression in modules/data/tests/unit/test_compression.py
- [ ] T059 [P] Unit tests for anonymizer in modules/data/tests/unit/test_anonymizer.py
- [ ] T060 [P] Unit tests for quality validation in modules/data/tests/unit/test_quality_check.py
- [ ] T060a [P] Unit tests for multi-channel QRN extractor in modules/data/tests/unit/test_multichannel_qrn.py
- [ ] T060b [P] Unit tests for propagation mode classifier in modules/data/tests/unit/test_mode_classifier.py
- [ ] T060c [P] Unit tests for correlation manager in modules/data/tests/unit/test_correlation_manager.py
- [ ] T060d [P] Unit tests for rarity scorer in modules/data/tests/unit/test_rarity_scorer.py
- [ ] T060e [P] Unit tests for path context calculator in modules/data/tests/unit/test_path_context.py
- [ ] T060f [P] Unit tests for QA sampler in modules/data/tests/unit/test_qa_sampler.py
- [ ] T060g [P] Unit tests for quarantine manager in modules/data/tests/unit/test_quarantine_manager.py
- [ ] T060h [P] Unit tests for waterfall generator in modules/data/tests/unit/test_waterfall_generator.py
- [ ] T060i [P] Unit tests for QA search functionality in modules/data/tests/unit/test_qa_search.py

### Performance & Documentation
- [ ] T061 Performance test: 300-400 hours/day collection rate
- [ ] T062 Load test: 100+ concurrent SDR connections during events
- [ ] T063 [P] Update README.md with deployment instructions
- [ ] T064 [P] Create API documentation in docs/api.md
- [ ] T065 Run quickstart.md scenarios for validation
- [ ] T066 Validate embedding training data requirements
- [ ] T067 Test diversity-biased dataset curation pipeline

### Anonymization Clarification Tasks
- [ ] T068 [P] Update anonymizer to preserve grid squares while hashing callsigns in modules/data/src/processors/anonymizer.py
- [ ] T069 [P] Add grid square validation to ensure propagation distance calculations in modules/data/src/processors/path_context.py
- [ ] T070 [P] Update unit tests for anonymizer to verify grid preservation in modules/data/tests/unit/test_anonymizer.py
- [x] T071 [P] Document privacy-preserving approach for grid squares in docs/privacy.md

### Station Fingerprinting Tasks
- [x] T072 [P] Design station fingerprint extraction from signal history in modules/data/src/processors/station_fingerprint.py
- [x] T073 [P] Create station_fingerprints database table and tracking schema in modules/data/migrations/add_station_fingerprints.sql
- [x] T074 [P] Implement persistent path characterization using tx_hash pairs in modules/data/src/processors/persistent_paths.py
- [x] T075 [P] Add station-aware embedding generation with equipment signatures in modules/data/src/embeddings/station_aware.py
- [x] T076 [P] Extract equipment characteristics (phase noise, drift, linearity) in modules/data/src/processors/equipment_signature.py
- [ ] T076a [P] Implement phase noise analysis from FT8 symbol centers in modules/data/src/processors/phase_noise_analyzer.py
- [ ] T076b [P] Build frequency drift tracker across multiple observations in modules/data/src/processors/drift_tracker.py
- [ ] T076c [P] Create PA linearity estimator from spectral regrowth in modules/data/src/processors/pa_linearity.py
- [ ] T076d [P] Implement equipment-propagation separator using multi-diversity in modules/data/src/processors/signal_separator.py
- [ ] T076e [P] Build reciprocal path analyzer for equipment isolation in modules/data/src/processors/reciprocal_analyzer.py
- [ ] T076f [P] Create multi-receiver correlation for TX equipment extraction in modules/data/src/processors/multi_rx_correlation.py
- [ ] T076g [P] Implement temporal scale separator (fast vs slow variations) in modules/data/src/processors/timescale_separator.py
- [ ] T076h [P] Build known propagation removal using decoded FT8 content in modules/data/src/processors/propagation_removal.py
- [ ] T076i [P] Create frequency diversity analyzer for equipment invariants in modules/data/src/processors/frequency_diversity.py
- [ ] T076j [P] Implement statistical confidence scorer for equipment signatures in modules/data/src/processors/signature_confidence.py
- [x] T077 [P] Build station activity pattern analyzer (time-of-day, band preferences) in modules/data/src/analytics/station_patterns.py
- [x] T078 [P] Create privacy-safe aggregation for station statistics in modules/data/src/analytics/station_aggregator.py
- [x] T079 [P] Implement anomaly detection for unusual station behavior in modules/data/src/validators/station_anomaly.py
- [ ] T080 [P] Unit tests for station fingerprinting pipeline in modules/data/tests/unit/test_station_fingerprint.py
- [ ] T080a [P] Unit tests for phase noise analyzer in modules/data/tests/unit/test_phase_noise.py
- [ ] T080b [P] Unit tests for drift tracker in modules/data/tests/unit/test_drift_tracker.py
- [ ] T080c [P] Unit tests for PA linearity estimator in modules/data/tests/unit/test_pa_linearity.py
- [ ] T080d [P] Unit tests for signal separator in modules/data/tests/unit/test_signal_separator.py
- [ ] T080e [P] Unit tests for multi-receiver correlation in modules/data/tests/unit/test_multi_rx.py
- [ ] T081 [P] Integration test for persistent path learning in modules/data/tests/integration/test_persistent_paths.py
- [ ] T081a [P] Integration test for equipment-propagation separation in modules/data/tests/integration/test_equipment_separation.py
- [ ] T081b [P] Integration test for multi-diversity fingerprinting in modules/data/tests/integration/test_diversity_fingerprint.py
- [ ] T082 [P] Document data collection ethics and public nature of amateur radio in docs/ethics_statement.md

## Dependencies

### Critical Path
1. **Setup** (T001-T005a) → Blocks everything
2. **Tests** (T006-T017b) → Must fail before implementation
3. **Models** (T018-T024b) → Blocks services and API
4. **Core Services** (T025-T032b) → Blocks API endpoints
5. **API Endpoints** (T039-T045) → Blocks integration tests passing
6. **Integration** (T046-T056b) → Blocks full system operation
7. **Polish** (T057-T065) → Final quality assurance

### Service Dependencies
- T025 (KiwiClient) blocks T026 (Recorder)
- T026 (Recorder) blocks T027 (Scheduler)
- T035 (metadata_db) blocks T052 (DB connection)
- T046 (NOAA client) blocks T053 (scheduled polling)

## Parallel Execution Examples

### Launch all contract tests together (T006-T012):
```bash
# In separate terminals or using Task agents:
Task: "Contract test POST /collectors/connect in modules/data/tests/contract/test_collector_connect.py"
Task: "Contract test POST /collectors/start in modules/data/tests/contract/test_collector_start.py"
Task: "Contract test POST /collectors/stop in modules/data/tests/contract/test_collector_stop.py"
Task: "Contract test GET /collectors/status in modules/data/tests/contract/test_collector_status.py"
Task: "Contract test POST /processors/decode/ft8 in modules/data/tests/contract/test_processor_ft8.py"
Task: "Contract test POST /processors/decode/wspr in modules/data/tests/contract/test_processor_wspr.py"
Task: "Contract test POST /processors/analyze/qrn in modules/data/tests/contract/test_processor_qrn.py"
```

### Launch all model creation tasks together (T018-T024):
```bash
Task: "RecordingSession model in modules/data/src/models/recording_session.py"
Task: "KiwiSDRSource model in modules/data/src/models/kiwisdr_source.py"
Task: "QRNSample model in modules/data/src/models/qrn_sample.py"
Task: "PropagationRecord model in modules/data/src/models/propagation_record.py"
Task: "SpaceWeatherData model in modules/data/src/models/space_weather_data.py"
Task: "CollectionSchedule model in modules/data/src/models/collection_schedule.py"
Task: "NotificationConfig model in modules/data/src/models/notification_config.py"
```

### Launch processor services together (T029-T032):
```bash
Task: "FT8 decoder in modules/data/src/processors/ft8_decoder.py"
Task: "WSPR decoder in modules/data/src/processors/wspr_decoder.py"
Task: "QRN analyzer in modules/data/src/processors/qrn_analyzer.py"
Task: "Callsign anonymizer in modules/data/src/processors/anonymizer.py"
```

## Notes
- Each task creates a single file (enables parallel execution)
- Contract tests must be written and fail before implementation
- API endpoints (T039-T045) modify same file, run sequentially
- Database connection (T052) blocks most integration tasks
- Fly.io deployment can be tested locally with `fly local`

## Validation Checklist
- ✓ All contracts have corresponding tests (collector_api → T006-T009, processor_api → T010-T012)
- ✓ All entities have model tasks (7 entities + 2 notification tables → T018-T024b)
- ✓ All tests come before implementation (Phase 3.2 before 3.3)
- ✓ Parallel tasks truly independent (different files)
- ✓ Each task specifies exact file path
- ✓ No [P] task modifies same file as another [P] task (API endpoints sequential)
- ✓ FR-021 center frequencies now have configuration task (T005a)
- ✓ Event scaling requirements have implementation tasks (T028a, T053a-c)
- ✓ Training pipeline requirements supported (T032a-b, T047a, T056a-b, T060a-e)
- ✓ Correlation preservation ensured (T018, T020, T021 updated with correlation_id)
- ✓ Multi-channel extraction supported (T032a for 9x 2.5kHz channels)
- ✓ Propagation mode detection included (T029, T030, T032b)
- ✓ X-ray classification captured (T022, T046 updated)
- ✓ Path context extraction for geographic propagation (T032c, T060e, FR-030)

### Geographic Bias Mitigation Tasks
- [x] T083 [P] Implement latitude band quota system in modules/data/src/collectors/geographic_quotas.py
- [x] T083a [P] Define latitude bands (Arctic >66.5°, Temperate 23.5-66.5°, Tropical ±23.5°, Antarctic <-66.5°)
- [x] T083b [P] Create quota configuration for minimum hours per band (20% each band minimum)
- [x] T083c [P] Add hemispheric balance requirements (40% North, 40% South, 20% Equatorial ±10°)
- [x] T084 [P] Add under-represented region boosting to hybrid_sdr_selector.py
- [x] T084a [P] Implement scarcity scoring based on SDR density per grid square prefix
- [x] T084b [P] Apply 2x-3x weight multiplier for scarce regions (< 5 SDRs per 1000km²)
- [x] T084c [P] Add ocean/land path classification and balancing (30% ocean paths minimum)
- [x] T085 [P] Create geographic diversity metrics in modules/data/src/validators/geographic_diversity.py
- [x] T085a [P] Calculate Simpson's diversity index for geographic distribution
- [x] T085b [P] Implement hemispheric balance score (target: 0.8-1.2 ratio)
- [x] T085c [P] Add continental coverage tracker (all 7 continents represented)
- [x] T085d [P] Create latitude distribution histogram with Chi-square test for uniformity
- [x] T086 [P] Implement reciprocal path inference for sparse regions in modules/data/src/processors/reciprocal_inference.py
- [x] T086a [P] Identify bidirectional paths from existing data
- [x] T086b [P] Generate synthetic southern observations from northern TX->southern RX paths
- [x] T086c [P] Weight inferred data at 0.5x compared to direct observations
- [ ] T087 [S] Add geographic bias monitoring to dashboard
- [ ] T087a [P] Create real-time geographic distribution heatmap visualization
- [ ] T087b [P] Add latitude band collection progress bars
- [ ] T087c [P] Implement bias warning system (alert when any region <50% of target)
- [ ] T087d [P] Add automatic rebalancing trigger recommendations
- [ ] T088 [S] Update collection scheduler for diversity-aware rotation
- [ ] T088a [P] Reserve 20% of collection slots for under-represented regions
- [ ] T088b [P] Implement "geographic diversity hour" daily (rotate through scarce regions)
- [ ] T088c [P] Add prefer_scarce_regions flag to scheduler configuration
- [ ] T088d [P] Create trade-off algorithm (efficiency vs diversity based on progress)
- [ ] T089 [P] Create southern hemisphere priority collector in modules/data/src/collectors/southern_priority.py
- [ ] T089a [P] Maintain prioritized list of southern hemisphere SDRs
- [ ] T089b [P] Implement 3x collection weight for southern stations
- [ ] T089c [P] Add failover to reciprocal inference when southern SDRs unavailable
- [ ] T090 [P] Unit tests for geographic bias mitigation
- [ ] T090a [P] Test quota enforcement in modules/data/tests/unit/test_geographic_quotas.py
- [ ] T090b [P] Test diversity metrics calculation in modules/data/tests/unit/test_diversity_metrics.py
- [ ] T090c [P] Test reciprocal path inference in modules/data/tests/unit/test_reciprocal.py
- [ ] T091 [P] Integration tests for bias-aware collection
- [ ] T091a [P] Test end-to-end collection with diversity requirements in modules/data/tests/integration/test_diversity_collection.py
- [ ] T091b [P] Test automatic rebalancing triggers in modules/data/tests/integration/test_rebalancing.py
- [x] T092 [P] Document geographic bias mitigation strategy in docs/geographic_diversity.md

### Phase 3.8: Advanced Power Estimation [P]
*Sophisticated power estimation from FT8/WSPR signals for improved propagation modeling*

- [x] T093 [P] Implement SNR-distance power estimator in modules/data/src/processors/power_estimator.py
- [x] T093a [P] Create triangulation algorithm using multiple receiver SNR reports
- [x] T093b [P] Implement free-space path loss calculations with atmospheric corrections
- [x] T093c [P] Add confidence scoring based on number of receivers and SNR consistency
- [x] T094 [P] Build statistical power analyzer in modules/data/src/processors/statistical_power.py
- [x] T094a [P] Create amateur radio power distribution model (QRP 5W, typical 100W, QRO 1500W)
- [x] T094b [P] Implement Bayesian inference for power level estimation
- [x] T094c [P] Add band-specific power probability distributions
- [x] T095 [P] Create reciprocal path power detector in modules/data/src/processors/reciprocal_power.py
- [x] T095a [P] Identify bidirectional QSO pairs from FT8 logs
- [x] T095b [P] Calculate power asymmetry from SNR differences
- [x] T095c [P] Add statistical aggregation across multiple QSOs
- [x] T096 [P] Implement PA compression analyzer in modules/data/src/processors/pa_compression.py
- [x] T096a [P] Detect harmonic content and spectral regrowth patterns
- [x] T096b [P] Identify IMD products indicating compression
- [x] T096c [P] Estimate actual vs reported power from compression indicators
- [x] T097 [P] Build multi-band power correlator in modules/data/src/processors/multiband_power.py
- [x] T097a [P] Track stations across multiple bands
- [x] T097b [P] Correlate power estimates accounting for band-specific propagation
- [x] T097c [P] Flag inconsistent power patterns for review
- [x] T098 [P] Create WSPR power calibrator in modules/data/src/processors/wspr_calibration.py
- [x] T098a [P] Extract explicit power reports from WSPR messages
- [x] T098b [P] Build calibration model mapping WSPR to FT8 power estimates
- [x] T098c [P] Apply calibration corrections to FT8-only stations
- [x] T099 [P] Implement temporal power pattern tracker in modules/data/src/processors/temporal_power.py
- [x] T099a [P] Detect time-of-day power adjustment patterns
- [x] T099b [P] Identify contest vs casual operation power levels
- [x] T099c [P] Build station-specific power profiles over time

---
*Generated from CASCADE implementation plan v1.0.0*