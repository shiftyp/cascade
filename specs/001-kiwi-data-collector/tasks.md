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

**Note**: Expanded scope from 60,000 to 150,000-300,000 hours with event-driven scaling and training pipeline support

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Phase 3.1: Setup
- [ ] T001 Create project structure in modules/data/ per plan.md
- [ ] T002 Initialize Python 3.11 project with requirements.txt dependencies (including redis-py)
- [ ] T003 [P] Configure pytest and ruff for linting
- [ ] T004 [P] Setup PostgreSQL database and run initial schema migration
- [ ] T005 Configure Fly.io deployment files (fly.toml, Dockerfile) with auto-scaling
- [ ] T005a Configure center frequencies per FR-021 in modules/data/src/config/frequencies.py
- [ ] T005b [P] Setup Redis/KeyDB instance on Fly.io for distributed coordination
- [ ] T005c [P] Configure Fly.io machines (scheduler, workers) in fly.toml
- [ ] T005d [P] Setup Tigris S3 bucket configuration for FLAC storage

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### Contract Tests
- [ ] T006 [P] Contract test POST /collectors/connect in modules/data/tests/contract/test_collector_connect.py
- [ ] T007 [P] Contract test POST /collectors/start in modules/data/tests/contract/test_collector_start.py
- [ ] T008 [P] Contract test POST /collectors/stop in modules/data/tests/contract/test_collector_stop.py
- [ ] T009 [P] Contract test GET /collectors/status in modules/data/tests/contract/test_collector_status.py
- [ ] T010 [P] Contract test POST /processors/decode/ft8 in modules/data/tests/contract/test_processor_ft8.py
- [ ] T011 [P] Contract test POST /processors/decode/wspr in modules/data/tests/contract/test_processor_wspr.py
- [ ] T012 [P] Contract test POST /processors/analyze/qrn in modules/data/tests/contract/test_processor_qrn.py

### Integration Tests
- [ ] T013 [P] Integration test KiwiSDR connection rotation in modules/data/tests/integration/test_sdr_rotation.py
- [ ] T014 [P] Integration test recording pipeline in modules/data/tests/integration/test_recording_pipeline.py
- [ ] T015 [P] Integration test FT8/WSPR extraction in modules/data/tests/integration/test_signal_extraction.py
- [ ] T016 [P] Integration test space weather trigger in modules/data/tests/integration/test_space_weather.py
- [ ] T017 [P] Integration test Tigris storage upload in modules/data/tests/integration/test_storage_upload.py
- [ ] T017a [P] Integration test correlation preservation in modules/data/tests/integration/test_correlation_preservation.py
- [ ] T017b [P] Integration test multi-channel extraction in modules/data/tests/integration/test_multichannel_extraction.py
- [ ] T017c [P] Integration test graceful degradation in modules/data/tests/integration/test_graceful_degradation.py
- [ ] T017d [P] Integration test all-SDR failure recovery in modules/data/tests/integration/test_total_failure_recovery.py
- [ ] T017e [P] Integration test Redis queue operations in modules/data/tests/integration/test_redis_queue.py
- [ ] T017f [P] Integration test distributed worker coordination in modules/data/tests/integration/test_worker_coordination.py
- [ ] T017g [P] Integration test Fly.io auto-scaling in modules/data/tests/integration/test_fly_autoscale.py

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Database Models
- [ ] T018 [P] RecordingSession model with correlation_id in modules/data/src/models/recording_session.py
- [ ] T019 [P] KiwiSDRSource model in modules/data/src/models/kiwisdr_source.py
- [ ] T020 [P] QRNSample model with quiet_periods and correlation_id in modules/data/src/models/qrn_sample.py
- [ ] T021 [P] PropagationRecord model with propagation_mode and correlation_id in modules/data/src/models/propagation_record.py
- [ ] T022 [P] SpaceWeatherData model with xray_class and xray_flux in modules/data/src/models/space_weather_data.py
- [ ] T023 [P] CollectionSchedule model in modules/data/src/models/collection_schedule.py
- [ ] T024 [P] NotificationConfig model in modules/data/src/models/notification_config.py
- [ ] T024a [P] CollectionAlerts model in modules/data/src/models/collection_alerts.py
- [ ] T024b [P] NotificationTemplates model in modules/data/src/models/notification_templates.py

### Collector Services
- [ ] T025 [P] KiwiClient connection manager (implements FR-001, FR-008, FR-009) in modules/data/src/collectors/kiwi_client.py
- [ ] T026 [P] Recorder orchestrator (implements FR-002, FR-007, FR-015) in modules/data/src/collectors/recorder.py
- [ ] T027 Collection scheduler (implements FR-011, FR-016, FR-042) in modules/data/src/collectors/scheduler.py
- [ ] T027a [P] Distributed worker process in modules/data/src/collectors/worker.py
- [ ] T027b [P] Redis queue manager in modules/data/src/collectors/queue_manager.py
- [ ] T027c [P] Distributed lock manager for SDR claims in modules/data/src/collectors/lock_manager.py
- [ ] T028 [P] Propagation-aware SDR rotation algorithm (implements FR-008, FR-014, FR-022) in modules/data/src/collectors/sdr_manager.py
- [ ] T028a Event-based SDR scaling logic (implements FR-023, FR-024, FR-041) in modules/data/src/collectors/event_scaler.py
- [ ] T028b [P] Graceful degradation handler (implements FR-032) in modules/data/src/collectors/degradation_handler.py
- [ ] T028c [P] SDR health monitor with auto-reconnect (implements FR-033) in modules/data/src/collectors/health_monitor.py
- [ ] T028d [P] Minimum collection scheduler for 1-SDR mode (implements FR-035) in modules/data/src/collectors/minimum_scheduler.py

### Processor Services
- [ ] T029 [P] FT8 decoder with propagation mode detection (implements FR-004, FR-025, FR-030) in modules/data/src/processors/ft8_decoder.py
- [ ] T030 [P] WSPR decoder with propagation mode detection (implements FR-004, FR-025, FR-030) in modules/data/src/processors/wspr_decoder.py
- [ ] T031 [P] QRN analyzer with quiet period detection in modules/data/src/processors/qrn_analyzer.py
- [ ] T032 [P] Callsign anonymizer (implements FR-005, FR-006) in modules/data/src/processors/anonymizer.py
- [ ] T032a [P] Multi-channel QRN extractor (9x 2.5kHz overlapping) in modules/data/src/processors/multichannel_qrn.py
- [ ] T032b [P] Propagation mode classifier in modules/data/src/processors/mode_classifier.py
- [ ] T032c [P] Path context calculator for propagation geometry in modules/data/src/processors/path_context.py

### Storage Services
- [ ] T033 [P] FLAC compression utility (implements FR-007, FR-031) in modules/data/src/storage/compression.py
- [ ] T034 [P] File manager for FLAC in modules/data/src/storage/file_manager.py
- [ ] T035 [P] PostgreSQL metadata interface in modules/data/src/storage/metadata_db.py
- [ ] T036 [P] Tigris S3 storage client in modules/data/src/storage/tigris_storage.py

### Validators
- [ ] T037 [P] Quality validation (implements FR-013, FR-019) in modules/data/src/validators/quality_check.py
- [ ] T038 [P] Coverage validation in modules/data/src/validators/coverage_check.py
- [ ] T038a [P] QA sampler for 1% hot storage (implements FR-036) in modules/data/src/validators/qa_sampler.py
- [ ] T038b [P] QA report generator (implements FR-037) in modules/data/src/validators/qa_reporter.py
- [ ] T038c [P] Quarantine manager for failed QC (implements FR-038) in modules/data/src/validators/quarantine_manager.py

### API Endpoints
- [ ] T039 POST /collectors/connect endpoint in modules/data/src/api/collectors.py
- [ ] T040 POST /collectors/start endpoint in modules/data/src/api/collectors.py
- [ ] T041 POST /collectors/stop endpoint in modules/data/src/api/collectors.py
- [ ] T042 GET /collectors/status endpoint in modules/data/src/api/collectors.py
- [ ] T043 POST /processors/decode/ft8 endpoint in modules/data/src/api/processors.py
- [ ] T044 POST /processors/decode/wspr endpoint in modules/data/src/api/processors.py
- [ ] T045 POST /processors/analyze/qrn endpoint in modules/data/src/api/processors.py

## Phase 3.4: Integration

### External Services
- [ ] T046 NOAA space weather API client with X-ray data in modules/data/src/external/noaa_client.py
- [ ] T047 Gmail notification service in modules/data/src/notifications/gmail_notifier.py
- [ ] T047a [P] Correlation manager for paired samples in modules/data/src/processors/correlation_manager.py

### Dashboard & Monitoring
- [ ] T049 Dashboard SQL views in modules/data/src/dashboard/views.sql
- [ ] T050 Terminal dashboard script in modules/data/src/dashboard/terminal_dashboard.py
- [ ] T051 FastAPI dashboard web app with QA waterfall viewer (implements FR-044, FR-045, FR-046) in modules/data/src/dashboard/web_dashboard.py
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
- [ ] T061 Performance test: 200-500 hours/day collection rate
- [ ] T062 Load test: 50+ concurrent SDR connections during events
- [ ] T063 [P] Update README.md with deployment instructions
- [ ] T064 [P] Create API documentation in docs/api.md
- [ ] T065 Run quickstart.md scenarios for validation
- [ ] T066 Validate embedding training data requirements
- [ ] T067 Test diversity-biased dataset curation pipeline

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

---
*Generated from CASCADE implementation plan v1.0.0*