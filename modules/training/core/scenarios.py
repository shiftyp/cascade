"""
Scenario templates for physics-based CASCADE training data generation.

9 fundamental scenarios representing real-world HF conditions:
1. Excellent - High solar activity, quiet geomagnetic, good propagation
2. Good - Above average conditions
3. Moderate - Average/typical conditions
4. Poor - Low solar activity or poor propagation
5. Geomagnetic Storm - Disturbed ionosphere (K>5)
6. High Atmospheric Noise - Thunderstorms, tropical QRN
7. Low Band Challenges - 80m/40m specific issues
8. Greyline - Enhanced propagation at sunrise/sunset
9. Polar - High latitude challenges (aurora, absorption)

Each scenario generates MANY variations through continuous sampling.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

from physics_coupling import CorePhysicalDrivers, CoupledPhysicsCalculator, DerivedConditions
from continuous_distributions import (
    create_solar_flux_dist, create_k_index_dist, create_thunderstorm_activity_dist,
    create_time_of_day_dist, create_latitude_dist, create_frequency_dist,
    create_day_of_year_dist
)


class ScenarioType(Enum):
    """9 fundamental scenario types."""
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"
    GEOMAGNETIC_STORM = "geomagnetic_storm"
    HIGH_ATMOSPHERIC_NOISE = "high_atmospheric_noise"
    LOW_BAND_CHALLENGES = "low_band_challenges"
    GREYLINE = "greyline"
    POLAR = "polar"


@dataclass
class ScenarioTemplate:
    """
    Template for generating scenario instances.

    Each template defines distributions for core physical drivers.
    Actual instances are sampled continuously from these distributions.
    """
    name: str
    description: str
    scenario_type: ScenarioType

    # Distribution names (not actual values!)
    solar_flux_scenario: str
    k_index_scenario: str
    thunderstorm_scenario: str
    time_scenario: str
    latitude_scenario: str
    season: str
    bands: List[str]  # Which HF bands to use

    # Relative weight for sampling (balanced-realistic)
    weight: float = 1.0

    # Physics constraints (incompatible combinations)
    incompatible_with: List[str] = field(default_factory=list)


class ScenarioLibrary:
    """Library of 9 fundamental scenarios + variations."""

    def __init__(self):
        self.templates = self._create_fundamental_scenarios()
        self.physics_calc = CoupledPhysicsCalculator()

    def _create_fundamental_scenarios(self) -> Dict[str, ScenarioTemplate]:
        """Create the 9 fundamental scenario templates."""
        templates = {}

        # 1. EXCELLENT - High solar activity, quiet geomagnetic, good propagation
        templates['excellent'] = ScenarioTemplate(
            name="Excellent Conditions",
            description="High solar flux, quiet geomagnetic field, optimal propagation",
            scenario_type=ScenarioType.EXCELLENT,
            solar_flux_scenario='excellent',
            k_index_scenario='quiet',
            thunderstorm_scenario='none',
            time_scenario='day',
            latitude_scenario='mid_latitude',
            season='summer',
            bands=['20m', '17m', '15m', '12m', '10m'],
            weight=0.15,  # 15% of training data (oversampled vs ~5% reality)
            incompatible_with=[]
        )

        # 2. GOOD - Above average conditions
        templates['good'] = ScenarioTemplate(
            name="Good Conditions",
            description="Above average solar activity and propagation",
            scenario_type=ScenarioType.GOOD,
            solar_flux_scenario='good',
            k_index_scenario='quiet',
            thunderstorm_scenario='none',
            time_scenario='random',
            latitude_scenario='mid_latitude',
            season='random',
            bands=['20m', '17m', '15m', '40m', '30m'],
            weight=0.25,  # 25% of training data
            incompatible_with=[]
        )

        # 3. MODERATE - Average/typical conditions
        templates['moderate'] = ScenarioTemplate(
            name="Moderate Conditions",
            description="Average propagation, typical noise levels",
            scenario_type=ScenarioType.MODERATE,
            solar_flux_scenario='moderate',
            k_index_scenario='unsettled',
            thunderstorm_scenario='light',
            time_scenario='random',
            latitude_scenario='random',
            season='random',
            bands=['20m', '40m', '30m', '15m', '80m'],
            weight=0.30,  # 30% of training data (most common)
            incompatible_with=[]
        )

        # 4. POOR - Low solar activity or poor propagation
        templates['poor'] = ScenarioTemplate(
            name="Poor Conditions",
            description="Low solar flux, high absorption, weak signals",
            scenario_type=ScenarioType.POOR,
            solar_flux_scenario='poor',
            k_index_scenario='active',
            thunderstorm_scenario='moderate',
            time_scenario='random',
            latitude_scenario='mid_latitude',
            season='winter',
            bands=['40m', '80m', '30m', '20m'],
            weight=0.15,  # 15% (oversampled vs ~10% reality)
            incompatible_with=[]
        )

        # 5. GEOMAGNETIC STORM - Disturbed ionosphere
        templates['geomagnetic_storm_minor'] = ScenarioTemplate(
            name="Minor Geomagnetic Storm (G1)",
            description="K=5-6, enhanced auroral activity, some absorption",
            scenario_type=ScenarioType.GEOMAGNETIC_STORM,
            solar_flux_scenario='moderate',
            k_index_scenario='minor_storm',
            thunderstorm_scenario='none',
            time_scenario='random',
            latitude_scenario='mid_latitude',
            season='random',
            bands=['40m', '80m', '30m', '20m'],
            weight=0.05,  # 5% (oversampled vs ~2% reality)
            incompatible_with=['excellent', 'greyline']
        )

        templates['geomagnetic_storm_major'] = ScenarioTemplate(
            name="Major Geomagnetic Storm (G2-G3)",
            description="K=6-8, severe disturbances, auroral absorption",
            scenario_type=ScenarioType.GEOMAGNETIC_STORM,
            solar_flux_scenario='moderate',
            k_index_scenario='major_storm',
            thunderstorm_scenario='none',
            time_scenario='random',
            latitude_scenario='mid_latitude',
            season='random',
            bands=['40m', '80m', '160m'],
            weight=0.03,  # 3% (oversampled vs ~0.5% reality)
            incompatible_with=['excellent', 'greyline', 'high_atmospheric_noise']
        )

        templates['geomagnetic_storm_severe'] = ScenarioTemplate(
            name="Severe Geomagnetic Storm (G4-G5)",
            description="K=8-9, extreme disturbances, blackout conditions",
            scenario_type=ScenarioType.GEOMAGNETIC_STORM,
            solar_flux_scenario='good',  # Need high SFI for CME
            k_index_scenario='severe_storm',
            thunderstorm_scenario='none',
            time_scenario='random',
            latitude_scenario='mid_latitude',
            season='random',
            bands=['80m', '160m', '40m'],
            weight=0.01,  # 1% (rare but critical to learn)
            incompatible_with=['excellent', 'greyline', 'high_atmospheric_noise']
        )

        # 6. HIGH ATMOSPHERIC NOISE - Thunderstorms, tropical QRN
        templates['high_atmospheric_noise'] = ScenarioTemplate(
            name="High Atmospheric Noise",
            description="Thunderstorms, static crashes, tropical QRN",
            scenario_type=ScenarioType.HIGH_ATMOSPHERIC_NOISE,
            solar_flux_scenario='good',
            k_index_scenario='quiet',
            thunderstorm_scenario='severe',
            time_scenario='night',  # Worse at night
            latitude_scenario='tropical',
            season='summer',  # Peak thunderstorm season
            bands=['40m', '80m', '160m', '30m'],  # Low bands most affected
            weight=0.08,  # 8% (oversampled vs ~3% reality)
            incompatible_with=['polar']  # No thunderstorms at poles
        )

        # 7. LOW BAND CHALLENGES - 80m/40m specific issues
        templates['low_band_challenges'] = ScenarioTemplate(
            name="Low Band Challenges",
            description="High atmospheric noise, D-layer absorption, crowding",
            scenario_type=ScenarioType.LOW_BAND_CHALLENGES,
            solar_flux_scenario='moderate',
            k_index_scenario='unsettled',
            thunderstorm_scenario='moderate',
            time_scenario='night',  # Better at night (no D-layer)
            latitude_scenario='mid_latitude',
            season='random',
            bands=['80m', '160m', '40m'],
            weight=0.10,  # 10% (important to learn)
            incompatible_with=[]
        )

        # 8. GREYLINE - Enhanced propagation at sunrise/sunset
        templates['greyline'] = ScenarioTemplate(
            name="Greyline Propagation",
            description="Enhanced DX at sunrise/sunset, low absorption",
            scenario_type=ScenarioType.GREYLINE,
            solar_flux_scenario='moderate',
            k_index_scenario='quiet',  # Need quiet for greyline enhancement
            thunderstorm_scenario='none',
            time_scenario='greyline',
            latitude_scenario='mid_latitude',
            season='random',
            bands=['20m', '40m', '30m', '17m', '15m'],
            weight=0.05,  # 5% (special conditions)
            incompatible_with=['geomagnetic_storm_major', 'geomagnetic_storm_severe']
        )

        # 9. POLAR - High latitude challenges
        templates['polar'] = ScenarioTemplate(
            name="Polar/Arctic Conditions",
            description="Auroral absorption, high K-index effects, auroral hiss",
            scenario_type=ScenarioType.POLAR,
            solar_flux_scenario='moderate',
            k_index_scenario='active',  # Higher K-index at high lat
            thunderstorm_scenario='none',  # No thunderstorms at poles
            time_scenario='random',
            latitude_scenario='polar',
            season='winter',  # Polar winter
            bands=['40m', '80m', '30m', '20m'],
            weight=0.03,  # 3% (rare but important)
            incompatible_with=['high_atmospheric_noise', 'greyline']
        )

        return templates

    def generate_scenario_instance(self, template_name: str, seed: Optional[int] = None) -> CorePhysicalDrivers:
        """
        Generate a single scenario instance by sampling from continuous distributions.

        Each call produces a UNIQUE set of parameters (continuous variation).
        """
        if seed is not None:
            np.random.seed(seed)

        template = self.templates[template_name]

        # Sample from continuous distributions
        sfi_dist = create_solar_flux_dist(template.solar_flux_scenario)
        sfi = sfi_dist.sample()

        k_dist = create_k_index_dist(template.k_index_scenario)
        k_index = k_dist.sample()

        ts_dist = create_thunderstorm_activity_dist(template.thunderstorm_scenario)
        thunderstorm_activity = ts_dist.sample()

        time_dist = create_time_of_day_dist(template.time_scenario)
        utc_hour = time_dist.sample()

        lat_dist = create_latitude_dist(template.latitude_scenario)
        latitude = lat_dist.sample()

        day_dist = create_day_of_year_dist(template.season)
        day_of_year = int(day_dist.sample())

        # Random longitude
        longitude = np.random.uniform(-180, 180)

        # Select random band from template's band list
        band = np.random.choice(template.bands)
        freq_dist = create_frequency_dist(band)
        frequency_mhz = freq_dist.sample()

        # Derive other correlated parameters
        # Sunspot number correlates with SFI
        sunspot_number = (sfi - 70) * 1.5 + np.random.normal(0, 10)
        sunspot_number = max(0, min(250, sunspot_number))

        # A-index from K-index (roughly K^2 * 3)
        a_index = k_index ** 2 * 3 + np.random.normal(0, 5)
        a_index = max(0, min(400, a_index))

        # Dst from K-index (storms have negative Dst)
        if k_index > 5:
            dst_index = -30 * (k_index - 4) + np.random.normal(0, 20)
        else:
            dst_index = np.random.normal(-20, 15)
        dst_index = max(-500, min(50, dst_index))

        # Precipitation correlates with thunderstorm activity
        if thunderstorm_activity > 0.3:
            precipitation_rate = thunderstorm_activity * 50 + np.random.exponential(10)
        else:
            precipitation_rate = np.random.exponential(2)

        # Create CorePhysicalDrivers instance
        drivers = CorePhysicalDrivers(
            sfi=sfi,
            sunspot_number=sunspot_number,
            k_index=k_index,
            a_index=a_index,
            dst_index=dst_index,
            utc_hour=utc_hour,
            day_of_year=day_of_year,
            latitude=latitude,
            longitude=longitude,
            thunderstorm_activity=thunderstorm_activity,
            precipitation_rate=precipitation_rate,
            frequency_mhz=frequency_mhz
        )

        return drivers

    def generate_balanced_realistic_batch(self, batch_size: int,
                                         for_test: bool = False,
                                         seed: Optional[int] = None) -> List[CorePhysicalDrivers]:
        """
        Generate a batch of scenarios with balanced-realistic weighting.

        Args:
            batch_size: Number of scenarios to generate
            for_test: If True, use harder distribution (more storms, fewer excellent)
            seed: Random seed for reproducibility

        Returns:
            List of CorePhysicalDrivers instances
        """
        if seed is not None:
            np.random.seed(seed)

        # Get weights
        if for_test:
            # Harder test distribution
            weights = self._get_test_weights()
        else:
            # Training/validation distribution
            weights = {name: t.weight for name, t in self.templates.items()}

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        # Sample template names according to weights
        template_names = list(weights.keys())
        template_weights = list(weights.values())

        selected_templates = np.random.choice(
            template_names,
            size=batch_size,
            p=template_weights,
            replace=True
        )

        # Generate instances
        batch = []
        for template_name in selected_templates:
            drivers = self.generate_scenario_instance(template_name)
            batch.append(drivers)

        return batch

    def _get_test_weights(self) -> Dict[str, float]:
        """
        Test distribution: Harder than training.

        More severe conditions, fewer excellent conditions.
        """
        weights = {}
        for name, template in self.templates.items():
            if template.scenario_type == ScenarioType.EXCELLENT:
                weights[name] = 0.0  # No excellent conditions in test!
            elif template.scenario_type == ScenarioType.GOOD:
                weights[name] = 0.10  # Reduced
            elif template.scenario_type == ScenarioType.MODERATE:
                weights[name] = 0.20  # Reduced
            elif template.scenario_type == ScenarioType.POOR:
                weights[name] = 0.25  # Increased
            elif template.scenario_type == ScenarioType.GEOMAGNETIC_STORM:
                # Significantly increased storms in test
                if 'severe' in name:
                    weights[name] = 0.10  # 10% severe storms (vs 1% training)
                elif 'major' in name:
                    weights[name] = 0.15  # 15% major storms (vs 3% training)
                else:
                    weights[name] = 0.10  # 10% minor storms (vs 5% training)
            else:
                weights[name] = template.weight  # Keep others same

        return weights

    def get_all_template_names(self) -> List[str]:
        """Get list of all template names."""
        return list(self.templates.keys())

    def get_template(self, name: str) -> ScenarioTemplate:
        """Get a specific template."""
        return self.templates[name]


def demo_scenario_generation():
    """Demonstrate scenario generation with continuous variation."""
    print("=" * 80)
    print("SCENARIO GENERATION DEMONSTRATION")
    print("=" * 80)

    library = ScenarioLibrary()
    calc = CoupledPhysicsCalculator()

    # Generate 3 instances of "excellent" scenario
    print("\n### 3 instances of 'Excellent Conditions' scenario ###")
    print("Notice: Each instance has DIFFERENT parameters (continuous variation)\n")

    for i in range(3):
        drivers = library.generate_scenario_instance('excellent', seed=i)
        conditions = calc.calculate_all_effects(drivers)

        print(f"Instance {i+1}:")
        print(f"  SFI: {drivers.sfi:.1f}, K-index: {drivers.k_index:.2f}")
        print(f"  Frequency: {drivers.frequency_mhz:.3f} MHz, Time: {drivers.utc_hour:.1f}h UTC")
        print(f"  → MUF: {conditions.muf_mhz:.1f} MHz, SNR: {conditions.effective_snr_db:.1f} dB")
        print(f"  → Propagation: {conditions.propagation_mode.value}, QRN: {conditions.dominant_qrn_type.value}")
        print()

    # Generate batch with balanced-realistic weighting
    print("\n### Balanced-Realistic Batch (Training) ###")
    batch = library.generate_balanced_realistic_batch(1000, for_test=False, seed=42)

    # Count scenarios by type
    scenario_counts = {}
    for drivers in batch:
        # Classify by characteristics
        if drivers.k_index > 7:
            stype = "Severe Storm"
        elif drivers.k_index > 5:
            stype = "Major Storm"
        elif drivers.k_index > 4:
            stype = "Minor Storm"
        elif drivers.sfi > 200:
            stype = "Excellent"
        elif drivers.sfi > 150:
            stype = "Good"
        elif drivers.sfi < 100:
            stype = "Poor"
        else:
            stype = "Moderate"

        scenario_counts[stype] = scenario_counts.get(stype, 0) + 1

    print("Distribution of 1000 training samples:")
    for stype, count in sorted(scenario_counts.items(), key=lambda x: -x[1]):
        print(f"  {stype}: {count} ({count/10:.1f}%)")

    # Test distribution
    print("\n### Harder Test Distribution ###")
    test_batch = library.generate_balanced_realistic_batch(1000, for_test=True, seed=42)

    test_counts = {}
    for drivers in test_batch:
        if drivers.k_index > 7:
            stype = "Severe Storm"
        elif drivers.k_index > 5:
            stype = "Major Storm"
        elif drivers.k_index > 4:
            stype = "Minor Storm"
        elif drivers.sfi > 200:
            stype = "Excellent"
        elif drivers.sfi > 150:
            stype = "Good"
        elif drivers.sfi < 100:
            stype = "Poor"
        else:
            stype = "Moderate"

        test_counts[stype] = test_counts.get(stype, 0) + 1

    print("Distribution of 1000 test samples (harder):")
    for stype, count in sorted(test_counts.items(), key=lambda x: -x[1]):
        print(f"  {stype}: {count} ({count/10:.1f}%)")

    print("\n" + "=" * 80)
    print("KEY FEATURES:")
    print("1. Continuous variation: No two instances are identical")
    print("2. Balanced-realistic: Rare conditions oversampled (vs real world)")
    print("3. Test harder than train: More storms, no excellent conditions")
    print("4. Physics coupling: All effects derived from same drivers")
    print("=" * 80)


if __name__ == "__main__":
    demo_scenario_generation()
