# CASCADE Data Collector - Scaling Guide

## Overview
The CASCADE Data Collector is designed to start small for testing and scale progressively to production levels without code changes or data loss.

## Collection Targets
- **Testing Phase**: 6-10 SDRs collecting ~20-50 hours/day
- **Staging Phase**: 20-50 SDRs collecting ~100-200 hours/day
- **Production Phase**: 50-100+ SDRs collecting 200-500 hours/day
- **Event Scaling**: Up to 200-300 SDRs during rare propagation events

## Scaling Methods

### 1. Initial Configuration (Set Once at Deployment)

Edit environment variables in `fly.toml` or set in Fly.io secrets:

```toml
[env]
  # Start conservative for testing
  BASELINE_SDR_COUNT = "6"      # Increase to 50-100 for production
  MAX_SDR_COUNT = "50"           # Increase to 200-300 for production

  # Alert thresholds
  ALERT_MIN_HOURS_DAY = "20"    # Increase to 200+ for production
  ALERT_SDR_AVAILABILITY = "50" # % of SDRs that must be available
```

Deploy with:
```bash
fly deploy
```

### 2. Dynamic Scaling (No Restart Required)

Use the scaling control script to adjust collection in real-time:

```bash
# View current configuration
./scripts/scale_collection.py --show

# Scale for testing (start here)
./scripts/scale_collection.py --baseline 6 --max 50 --min-hours 20

# Scale for staging (after successful testing)
./scripts/scale_collection.py --baseline 20 --max 100 --min-hours 50

# Scale for production (FR-016, FR-018 targets)
./scripts/scale_collection.py --baseline 100 --max 300 --min-hours 200

# Connect to production Redis
./scripts/scale_collection.py --redis redis://cascade-keydb.internal:6379 --baseline 100
```

**Important**: Changes take effect within 30 seconds with no data loss or session interruption.

## Scaling Decision Tree

```
Start Testing (Week 1-2)
├── 6-10 SDRs
├── Monitor stability
├── Check data quality
└── If stable → Scale to Staging

Staging (Week 3-4)
├── 20-50 SDRs
├── Monitor costs ($100-200/month)
├── Validate geographic diversity
└── If targets met → Scale to Production

Production (Month 2+)
├── 50-100 baseline SDRs
├── Auto-scale to 200+ during events
├── Monitor collection rate (target: 200-500 hrs/day)
└── Adjust based on:
    ├── Available SDRs
    ├── Storage costs
    └── Event frequency
```

## Resource Requirements by Scale

### Testing (6-10 SDRs)
- **Workers**: 1-2 machines
- **Storage**: ~100GB/month
- **Cost**: ~$50/month
- **Collection**: 20-50 hours/day

### Staging (20-50 SDRs)
- **Workers**: 2-5 machines
- **Storage**: ~500GB/month
- **Cost**: ~$150/month
- **Collection**: 100-200 hours/day

### Production (50-100+ SDRs)
- **Workers**: 5-10 machines (auto-scaling)
- **Storage**: 2-4TB/month
- **Cost**: ~$300-500/month
- **Collection**: 200-500 hours/day

### Full Scale (200-300 SDRs during events)
- **Workers**: 10-20 machines (burst)
- **Storage**: 5-10TB/month
- **Cost**: ~$800-1200/month (during events)
- **Collection**: 500-1000 hours/day (peak)

## Monitoring Collection Performance

### Check Current Status
```bash
# Via dashboard
curl https://cascade-kiwi-collector.fly.dev/api/status

# Via logs
fly logs -a cascade-kiwi-collector | grep "collection_rate"

# Via Redis
fly ssh console -C "redis-cli get scheduler:metrics"
```

### Key Metrics to Monitor
1. **Collection Rate**: Should approach target hours/day
2. **SDR Availability**: Should stay above 50%
3. **Geographic Diversity**: Simpson's index > 0.8
4. **Storage Growth**: ~2-4TB/month at full scale
5. **Worker CPU/Memory**: Should stay below 70%/80%

## Troubleshooting Scaling Issues

### Not Enough Collection Hours
```bash
# Increase baseline SDRs
./scripts/scale_collection.py --baseline 150

# Or prioritize WebSDRs (longer sessions)
fly secrets set PREFER_WEBSDR=true
```

### Too Many Failed Connections
```bash
# Reduce SDR count temporarily
./scripts/scale_collection.py --baseline 30

# Check SDR availability
fly ssh console -C "python -m modules.data.src.collectors.sdr_manager check"
```

### Storage Growing Too Fast
```bash
# Reduce QA sampling temporarily
fly secrets set QA_SAMPLE_PERCENTAGE=0.01

# Check compression efficiency
fly ssh console -C "ls -lh /nvme/recordings"
```

## Progressive QA Sampling Strategy

The system automatically adjusts QA sampling over 18 months:

| Phase | Months | QA Rate | Method | Purpose |
|-------|--------|---------|--------|---------|
| Bootstrap | 1-2 | 3% | Random | Initial training data |
| Hybrid | 3-4 | 8% | Mixed | Transition period |
| Production | 5-18 | 12% | Intelligent | Pattern diversity |

This is automatic and requires no configuration.

## WebSDR Integration

To increase collection capacity with longer sessions:

1. Add WebSDR sources to database
2. They automatically get prioritized for sessions >90 minutes
3. No daily limits for most WebSDRs = more collection hours

## Emergency Procedures

### Stop All Collection
```bash
./scripts/scale_collection.py --baseline 0 --max 0
```

### Reset to Safe Defaults
```bash
./scripts/scale_collection.py --baseline 6 --max 50 --min-hours 20
```

### Force Restart (Last Resort)
```bash
fly apps restart cascade-kiwi-collector
```

## Best Practices

1. **Start Small**: Begin with 6 SDRs, validate everything works
2. **Scale Gradually**: Double SDR count each week until target reached
3. **Monitor Costs**: Storage is the main cost at scale
4. **Watch Geographic Diversity**: Ensure global coverage
5. **Respond to Events**: Scale up quickly for rare propagation
6. **Document Changes**: Log when and why you scaled

## Configuration Reference

### Environment Variables (fly.toml)
- `BASELINE_SDR_COUNT`: Normal collection SDR count (default: 6)
- `MAX_SDR_COUNT`: Maximum during events (default: 50)
- `ALERT_MIN_HOURS_DAY`: Minimum collection hours before alert (default: 20)
- `ALERT_SDR_AVAILABILITY`: Minimum % SDRs available (default: 50%)

### Redis Dynamic Config Keys
- `scheduler:dynamic_config`: JSON config for live updates
- `scheduler:metrics`: Current collection metrics
- `alerts:operator`: Recent operator alerts

### Collection Formulas
- **Daily Hours** = Active SDRs × Average Session Length × Sessions per Day
- **Monthly Storage** = Daily Hours × 30 × Compression Ratio (0.45-0.55)
- **Worker Count** = Active SDRs ÷ 5 (each worker handles 5-10 SDRs)

## Contact

For scaling assistance or questions:
- Check logs: `fly logs -a cascade-kiwi-collector`
- Check status: `fly status -a cascade-kiwi-collector`
- SSH access: `fly ssh console -a cascade-kiwi-collector`