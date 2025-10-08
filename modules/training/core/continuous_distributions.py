"""
Continuous probability distributions for scenario parameter sampling.

Key principle: Use continuous distributions instead of discrete bins to prevent overfitting.
Every sample should be unique, forcing the model to generalize.

Example:
    Instead of: K-index ∈ {1, 3, 5, 7, 9}  # Model memorizes these 5 values
    Use: K-index ~ Beta(α, β) → {1.2, 2.8, 5.3, 7.1, 8.9, ...}  # Infinite variation
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class DistributionParams:
    """Parameters for a probability distribution."""
    dist_type: str  # 'beta', 'gamma', 'truncnorm', 'uniform', 'lognormal'
    params: dict  # Distribution-specific parameters
    range: Tuple[float, float]  # Valid range (min, max)


class ContinuousDistribution:
    """
    Continuous probability distribution with range constraints.

    Supports:
    - Beta: Good for bounded variables (0-1 range, then scaled)
    - Gamma: Good for right-skewed positive variables
    - TruncatedNormal: Good for bell-shaped with hard limits
    - Uniform: For truly random variables
    - LogNormal: For multiplicative processes
    """

    def __init__(self, dist_params: DistributionParams, seed: Optional[int] = None):
        self.dist_params = dist_params
        self.rng = np.random.default_rng(seed)

        # Create scipy distribution
        if dist_params.dist_type == 'beta':
            alpha = dist_params.params['alpha']
            beta = dist_params.params['beta']
            self._base_dist = stats.beta(alpha, beta)

        elif dist_params.dist_type == 'gamma':
            shape = dist_params.params['shape']
            scale = dist_params.params.get('scale', 1.0)
            self._base_dist = stats.gamma(shape, scale=scale)

        elif dist_params.dist_type == 'truncnorm':
            mean = dist_params.params['mean']
            std = dist_params.params['std']
            low, high = dist_params.range
            a = (low - mean) / std
            b = (high - mean) / std
            self._base_dist = stats.truncnorm(a, b, loc=mean, scale=std)

        elif dist_params.dist_type == 'uniform':
            low, high = dist_params.range
            self._base_dist = stats.uniform(loc=low, scale=high - low)

        elif dist_params.dist_type == 'lognormal':
            mean = dist_params.params['mean']
            sigma = dist_params.params['sigma']
            self._base_dist = stats.lognorm(s=sigma, scale=np.exp(mean))

        else:
            raise ValueError(f"Unknown distribution type: {dist_params.dist_type}")

    def sample(self, size: int = 1) -> np.ndarray:
        """Sample from the distribution."""
        if self.dist_params.dist_type in ['beta', 'gamma', 'lognormal']:
            # Sample from base distribution, then scale to range
            samples = self._base_dist.rvs(size=size, random_state=self.rng)

            # Scale beta from [0,1] to target range
            if self.dist_params.dist_type == 'beta':
                low, high = self.dist_params.range
                samples = low + samples * (high - low)

            # Clip gamma and lognormal to range
            elif self.dist_params.dist_type in ['gamma', 'lognormal']:
                low, high = self.dist_params.range
                samples = np.clip(samples, low, high)

        else:
            # truncnorm and uniform already respect range
            samples = self._base_dist.rvs(size=size, random_state=self.rng)

        return samples if size > 1 else samples[0]

    def mean(self) -> float:
        """Expected value."""
        if self.dist_params.dist_type == 'beta':
            low, high = self.dist_params.range
            beta_mean = self._base_dist.mean()
            return low + beta_mean * (high - low)
        else:
            return self._base_dist.mean()

    def std(self) -> float:
        """Standard deviation."""
        if self.dist_params.dist_type == 'beta':
            low, high = self.dist_params.range
            beta_std = self._base_dist.std()
            return beta_std * (high - low)
        else:
            return self._base_dist.std()


# Pre-defined distributions for common scenario parameters

def create_solar_flux_dist(scenario: str) -> ContinuousDistribution:
    """
    Solar Flux Index distribution (70-300 range).

    Scenarios:
    - excellent: High solar activity (SFI ~200-250)
    - good: Above average (SFI ~180-220)
    - moderate: Average solar activity (SFI ~120-180)
    - poor: Low solar activity (SFI ~70-120)
    - solar_minimum: Very low (SFI ~70-90)
    """
    if scenario == 'excellent':
        # High solar activity: skewed toward high values
        params = DistributionParams('beta', {'alpha': 8, 'beta': 2}, (180, 280))
    elif scenario == 'good':
        # Above average: centered around 200
        params = DistributionParams('truncnorm', {'mean': 200, 'std': 20}, (150, 250))
    elif scenario == 'moderate':
        # Average: broad distribution
        params = DistributionParams('truncnorm', {'mean': 150, 'std': 30}, (100, 200))
    elif scenario == 'poor':
        # Low activity: skewed toward low values
        params = DistributionParams('beta', {'alpha': 2, 'beta': 5}, (70, 150))
    elif scenario == 'solar_minimum':
        # Very low, tight distribution
        params = DistributionParams('truncnorm', {'mean': 80, 'std': 10}, (70, 100))
    else:
        # Default: uniform across full range
        params = DistributionParams('uniform', {}, (70, 300))

    return ContinuousDistribution(params)


def create_k_index_dist(scenario: str) -> ContinuousDistribution:
    """
    Planetary K-index distribution (0-9 continuous).

    Scenarios:
    - quiet: K < 2 (calm geomagnetic field)
    - unsettled: K ~ 3-4 (minor disturbances)
    - active: K ~ 4-5 (active conditions)
    - minor_storm: K ~ 5-6 (G1 storm)
    - major_storm: K ~ 6-7 (G2 storm)
    - severe_storm: K ~ 7-9 (G3-G5 storm)
    """
    if scenario == 'quiet':
        # Very low K-index, skewed toward 0-1
        params = DistributionParams('gamma', {'shape': 1.5, 'scale': 0.5}, (0, 2.5))
    elif scenario == 'unsettled':
        # Minor disturbances around K=3
        params = DistributionParams('truncnorm', {'mean': 3.5, 'std': 0.7}, (2.5, 4.5))
    elif scenario == 'active':
        # Active conditions K=4-5
        params = DistributionParams('truncnorm', {'mean': 4.5, 'std': 0.6}, (3.5, 5.5))
    elif scenario == 'minor_storm':
        # G1 storm: K ~ 5-6
        params = DistributionParams('beta', {'alpha': 5, 'beta': 3}, (5.0, 6.5))
    elif scenario == 'major_storm':
        # G2 storm: K ~ 6-7
        params = DistributionParams('beta', {'alpha': 5, 'beta': 3}, (6.0, 7.5))
    elif scenario == 'severe_storm':
        # G3-G5 storm: K ~ 7-9
        params = DistributionParams('beta', {'alpha': 3, 'beta': 2}, (7.0, 9.0))
    else:
        # Default: uniform
        params = DistributionParams('uniform', {}, (0, 9))

    return ContinuousDistribution(params)


def create_thunderstorm_activity_dist(scenario: str) -> ContinuousDistribution:
    """
    Thunderstorm activity distribution (0-1 continuous).

    Scenarios:
    - none: No thunderstorms (0-0.1)
    - light: Light activity (0.1-0.3)
    - moderate: Moderate storms (0.3-0.6)
    - severe: Severe thunderstorms (0.6-1.0)
    """
    if scenario == 'none':
        # Very low, near zero
        params = DistributionParams('beta', {'alpha': 1, 'beta': 10}, (0, 0.15))
    elif scenario == 'light':
        # Light activity
        params = DistributionParams('beta', {'alpha': 3, 'beta': 5}, (0.05, 0.35))
    elif scenario == 'moderate':
        # Moderate storms
        params = DistributionParams('beta', {'alpha': 5, 'beta': 3}, (0.25, 0.65))
    elif scenario == 'severe':
        # Severe thunderstorms, skewed high
        params = DistributionParams('beta', {'alpha': 8, 'beta': 2}, (0.55, 1.0))
    else:
        # Default: uniform
        params = DistributionParams('uniform', {}, (0, 1))

    return ContinuousDistribution(params)


def create_time_of_day_dist(scenario: str) -> ContinuousDistribution:
    """
    UTC hour distribution (0-24 continuous).

    Scenarios:
    - day: Daytime (6-18 UTC, adjusted for longitude)
    - night: Nighttime (18-6 UTC)
    - greyline: Around sunrise/sunset (±2 hours)
    - random: Uniform across day
    """
    if scenario == 'day':
        # Daytime hours with peak at solar noon
        params = DistributionParams('truncnorm', {'mean': 12, 'std': 3}, (6, 18))
    elif scenario == 'night':
        # Nighttime hours (two peaks: evening and early morning)
        # Use uniform for simplicity (avoid wraparound issues)
        # Sample from 20-24 or 0-6
        choice = np.random.choice(['evening', 'morning'])
        if choice == 'evening':
            params = DistributionParams('uniform', {}, (20, 24))
        else:
            params = DistributionParams('uniform', {}, (0, 6))
    elif scenario == 'greyline':
        # Dawn or dusk (we'll use dawn as example: 4-8 UTC)
        choice = np.random.choice(['dawn', 'dusk'])
        if choice == 'dawn':
            params = DistributionParams('truncnorm', {'mean': 6, 'std': 1.5}, (4, 9))
        else:
            params = DistributionParams('truncnorm', {'mean': 18, 'std': 1.5}, (16, 21))
    else:
        # Uniform across day
        params = DistributionParams('uniform', {}, (0, 24))

    return ContinuousDistribution(params)


def create_latitude_dist(scenario: str) -> ContinuousDistribution:
    """
    Station latitude distribution (-90 to +90).

    Scenarios:
    - tropical: ±23.5° (tropics)
    - mid_latitude: 30-60°
    - polar: >60°
    - random: Uniform (with bias toward populated mid-latitudes)
    """
    if scenario == 'tropical':
        # Tropical belt
        params = DistributionParams('uniform', {}, (-23.5, 23.5))
    elif scenario == 'mid_latitude':
        # Mid-latitudes (Northern hemisphere bias)
        params = DistributionParams('truncnorm', {'mean': 45, 'std': 10}, (25, 60))
    elif scenario == 'polar':
        # High latitudes (both hemispheres)
        choice = np.random.choice(['north', 'south'])
        if choice == 'north':
            params = DistributionParams('truncnorm', {'mean': 70, 'std': 8}, (60, 85))
        else:
            params = DistributionParams('truncnorm', {'mean': -70, 'std': 8}, (-85, -60))
    else:
        # Realistic: most stations at mid-latitudes
        params = DistributionParams('truncnorm', {'mean': 40, 'std': 25}, (-70, 70))

    return ContinuousDistribution(params)


def create_frequency_dist(band: str) -> ContinuousDistribution:
    """
    Operating frequency distribution for HF bands.

    Bands: 80m, 40m, 30m, 20m, 17m, 15m, 12m, 10m, 6m
    """
    band_ranges = {
        '160m': (1.8, 2.0),
        '80m': (3.5, 4.0),
        '60m': (5.3, 5.4),
        '40m': (7.0, 7.3),
        '30m': (10.1, 10.15),
        '20m': (14.0, 14.35),
        '17m': (18.068, 18.168),
        '15m': (21.0, 21.45),
        '12m': (24.89, 24.99),
        '10m': (28.0, 29.7),
        '6m': (50.0, 54.0)
    }

    if band in band_ranges:
        low, high = band_ranges[band]
        # Uniform within band
        params = DistributionParams('uniform', {}, (low, high))
    else:
        # Default: 20m band
        params = DistributionParams('uniform', {}, (14.0, 14.35))

    return ContinuousDistribution(params)


def create_day_of_year_dist(season: str) -> ContinuousDistribution:
    """
    Day of year distribution (1-365).

    Seasons:
    - winter: Dec-Feb (NH), days 335-365, 1-59
    - spring: Mar-May, days 60-151
    - summer: Jun-Aug, days 152-243
    - fall: Sep-Nov, days 244-334
    - random: Uniform across year
    """
    if season == 'winter':
        # Simplified: centered on Jan 15 (day 15)
        params = DistributionParams('truncnorm', {'mean': 15, 'std': 30}, (1, 70))
    elif season == 'spring':
        # Centered on Apr 15 (day 105)
        params = DistributionParams('truncnorm', {'mean': 105, 'std': 25}, (60, 151))
    elif season == 'summer':
        # Centered on Jul 15 (day 196)
        params = DistributionParams('truncnorm', {'mean': 196, 'std': 25}, (152, 243))
    elif season == 'fall':
        # Centered on Oct 15 (day 288)
        params = DistributionParams('truncnorm', {'mean': 288, 'std': 25}, (244, 334))
    else:
        # Uniform across year
        params = DistributionParams('uniform', {}, (1, 365))

    return ContinuousDistribution(params)


def demo_continuous_distributions():
    """Demonstrate continuous variation vs discrete bins."""
    print("=" * 80)
    print("CONTINUOUS DISTRIBUTIONS DEMONSTRATION")
    print("=" * 80)

    # K-index for severe storm
    print("\n### Severe Storm K-index Distribution ###")
    print("Instead of K ∈ {7, 8, 9}, we sample continuously:")
    k_dist = create_k_index_dist('severe_storm')
    samples = [k_dist.sample() for _ in range(10)]
    print(f"10 samples: {[f'{s:.2f}' for s in samples]}")
    print(f"Mean: {k_dist.mean():.2f}, Std: {k_dist.std():.2f}")
    print("→ Every sample is unique, model must generalize!")

    # Solar flux for excellent conditions
    print("\n### Excellent Conditions SFI Distribution ###")
    sfi_dist = create_solar_flux_dist('excellent')
    samples = [sfi_dist.sample() for _ in range(10)]
    print(f"10 samples: {[f'{s:.1f}' for s in samples]}")
    print(f"Mean: {sfi_dist.mean():.1f}, Std: {sfi_dist.std():.1f}")

    # Thunderstorm activity
    print("\n### Severe Thunderstorm Activity Distribution ###")
    ts_dist = create_thunderstorm_activity_dist('severe')
    samples = [ts_dist.sample() for _ in range(10)]
    print(f"10 samples: {[f'{s:.3f}' for s in samples]}")
    print(f"Mean: {ts_dist.mean():.3f}, Std: {ts_dist.std():.3f}")

    # Time of day - greyline
    print("\n### Greyline Time Distribution ###")
    time_dist = create_time_of_day_dist('greyline')
    samples = [time_dist.sample() for _ in range(10)]
    print(f"10 samples (UTC hours): {[f'{s:.2f}' for s in samples]}")

    print("\n" + "=" * 80)
    print("OVERFITTING PREVENTION:")
    print("- Discrete bins: Model memorizes {1, 3, 5, 7, 9}")
    print("- Continuous: Model sees {1.2, 2.8, 5.3, 7.1, 8.9, ...} → generalizes")
    print("=" * 80)


if __name__ == "__main__":
    demo_continuous_distributions()
