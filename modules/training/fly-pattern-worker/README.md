# CASCADE Pattern Generation - Fly.io Distributed Execution

Distributed pattern generation using Fly.io workers for massive parallelism.

## Cost Analysis

| Workers | Time | Cost | Expected Quality |
|---------|------|------|------------------|
| 16 | 18-24h | $3 | -39.5 dB, λ=0.24 |
| 32 | 18-24h | $6 | -40.2 dB, λ=0.22 |
| 64 | 18-24h | $12 | -40.8 dB, λ=0.21 |

**vs Local**: 8 trials free, 18-24h, -39 dB, λ=0.25

## Setup

1. **Create Fly.io app**:
```bash
cd modules/training/fly-pattern-worker
fly apps create cascade-pattern-worker
```

2. **Set Tigris credentials**:
```bash
fly secrets set AWS_ACCESS_KEY_ID=your_key
fly secrets set AWS_SECRET_ACCESS_KEY=your_secret
```

3. **Create Tigris bucket**:
```bash
fly storage create cascade-patterns
```

## Usage

**From main CLI**:
```bash
cd /workspaces/cascade

# Distributed execution with 32 workers
python -m modules.training.patterns generate \
    --count 128 \
    --distributed \
    --workers 32 \
    --seed 42

# Expected: 18-24 hours, $6 cost, -40.2 dB quality
```

**Manual coordinator**:
```bash
python coordinator.py --workers 32 --count 128 --seed 42
```

## Architecture

```
Coordinator (local)
  ↓ spawns
32 Workers (Fly.io performance-1x, $0.19/day each)
  ↓ each generates 1 trial
Tigris Storage (cascade-patterns bucket)
  ├── trials/trial_0.bin + trial_0_metadata.json
  ├── trials/trial_1.bin + trial_1_metadata.json
  └── ...
  ↓ coordinator downloads best
cascade_patterns_128.bin (final output)
```

## Worker Environment Variables

- `TRIAL_ID`: Trial number (0-N)
- `PATTERN_COUNT`: 64 or 128
- `SEED_BASE`: Base random seed
- `TIGRIS_BUCKET`: S3 bucket name
- `AWS_ACCESS_KEY_ID`: Tigris access key
- `AWS_SECRET_ACCESS_KEY`: Tigris secret key

## Monitoring

**Check Tigris for progress**:
```bash
fly storage dashboard cascade-patterns
# Look for trials/trial_*_metadata.json files
```

**View worker logs**:
```bash
fly logs --app cascade-pattern-worker
```

## Cleanup

Workers auto-cleanup after completion. Manual cleanup if needed:
```bash
fly machine list --app cascade-pattern-worker
fly machine destroy MACHINE_ID --app cascade-pattern-worker
```

## Troubleshooting

**Worker fails immediately**:
- Check Tigris credentials are set
- Verify bucket exists
- Check worker logs: `fly logs`

**Workers hang**:
- Insufficient memory (increase to 4GB in fly.toml)
- Check for Python errors in logs

**Results not uploading**:
- Verify Tigris credentials
- Check network connectivity
- Verify bucket permissions
