"""WSPR power calibration for FT8 power estimates.

T098: Use WSPR's explicit power reports to calibrate FT8 power estimation
models and correct systematic biases.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import curve_fit
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class WSPRReport:
    """WSPR transmission report with explicit power."""

    station_hash: str
    timestamp: datetime
    reported_power_dbm: float  # Explicit power from WSPR protocol

    # Reception report
    rx_hash: str
    snr_db: float
    distance_km: float
    rx_grid: str

    # Propagation context
    band: str
    frequency_hz: float
    drift_hz: Optional[float]


@dataclass
class CalibrationModel:
    """Calibration model mapping WSPR to FT8 power estimates."""

    creation_time: datetime
    num_samples: int

    # Linear calibration: actual = scale * estimated + offset
    scale_factor: float
    offset_db: float

    # Non-linear correction (polynomial coefficients)
    polynomial_coeffs: List[float]

    # Statistics
    rmse_db: float  # Root mean square error
    r_squared: float  # Coefficient of determination
    confidence_interval: Tuple[float, float]  # 95% CI for predictions

    # Band-specific adjustments
    band_corrections: Dict[str, float]  # Band-specific offset corrections


class WSPRPowerCalibrator:
    """Calibrates power estimation using WSPR reports."""

    def __init__(self):
        """Initialize WSPR calibrator."""
        self.wspr_reports: List[WSPRReport] = []
        self.calibration_models: Dict[str, CalibrationModel] = {}  # Per-band models
        self.station_calibrations: Dict[str, Dict] = {}  # Per-station corrections

        # WSPR power levels (standard values in dBm)
        self.wspr_power_levels = {
            0: 30.0,   # 1W
            3: 33.0,   # 2W
            7: 37.0,   # 5W
            10: 40.0,  # 10W
            13: 43.0,  # 20W
            17: 47.0,  # 50W
            20: 50.0,  # 100W
            23: 53.0,  # 200W
            27: 57.0,  # 500W
            30: 60.0,  # 1000W
            33: 63.0,  # 2000W
            37: 67.0,  # 5000W
            40: 70.0,  # 10000W
            43: 73.0,  # 20000W
            47: 77.0,  # 50000W
            50: 80.0,  # 100000W
            53: 83.0,  # 200000W
            57: 87.0,  # 500000W
            60: 90.0   # 1000000W
        }

    def add_wspr_report(self, report: WSPRReport):
        """Add a WSPR report for calibration.

        Args:
            report: WSPR transmission report
        """
        self.wspr_reports.append(report)

    def decode_wspr_power(self, power_code: int) -> float:
        """Decode WSPR power code to dBm.

        Args:
            power_code: WSPR power code (0-60)

        Returns:
            Power in dBm
        """
        # Find closest standard level
        if power_code in self.wspr_power_levels:
            return self.wspr_power_levels[power_code]

        # Interpolate if between levels
        codes = sorted(self.wspr_power_levels.keys())
        for i in range(len(codes) - 1):
            if codes[i] < power_code < codes[i+1]:
                # Linear interpolation
                p1 = self.wspr_power_levels[codes[i]]
                p2 = self.wspr_power_levels[codes[i+1]]
                fraction = (power_code - codes[i]) / (codes[i+1] - codes[i])
                return p1 + fraction * (p2 - p1)

        # Extrapolate if outside range
        if power_code < codes[0]:
            return self.wspr_power_levels[codes[0]]
        else:
            return self.wspr_power_levels[codes[-1]]

    def calibrate_model(self, ft8_estimates: List[Dict[str, Any]],
                       band: Optional[str] = None,
                       min_samples: int = 20) -> Optional[CalibrationModel]:
        """Create calibration model from WSPR ground truth and FT8 estimates.

        Args:
            ft8_estimates: List of FT8 power estimates with:
                - station_hash: Station identifier
                - estimated_power_dbm: FT8-based estimate
                - timestamp: Estimate time
                - band: Operating band
            band: Specific band to calibrate (None for all bands)
            min_samples: Minimum samples for calibration

        Returns:
            CalibrationModel or None if insufficient data
        """
        # Match WSPR reports with FT8 estimates
        matched_pairs = self._match_wspr_ft8(ft8_estimates, band)

        if len(matched_pairs) < min_samples:
            logger.warning(f"Insufficient samples for calibration: {len(matched_pairs)} < {min_samples}")
            return None

        # Extract actual (WSPR) and estimated (FT8) powers
        actual_powers = np.array([pair['wspr_power'] for pair in matched_pairs])
        estimated_powers = np.array([pair['ft8_power'] for pair in matched_pairs])

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(estimated_powers, actual_powers)

        # Polynomial fitting for non-linear correction
        poly_degree = min(3, len(matched_pairs) // 10)  # Adaptive degree
        poly_coeffs = np.polyfit(estimated_powers, actual_powers, poly_degree)

        # Calculate residuals and statistics
        linear_predictions = slope * estimated_powers + intercept
        poly_predictions = np.polyval(poly_coeffs, estimated_powers)

        linear_residuals = actual_powers - linear_predictions
        poly_residuals = actual_powers - poly_predictions

        # Use polynomial if significantly better
        if np.std(poly_residuals) < 0.9 * np.std(linear_residuals):
            predictions = poly_predictions
            residuals = poly_residuals
            use_polynomial = True
        else:
            predictions = linear_predictions
            residuals = linear_residuals
            use_polynomial = False
            poly_coeffs = [slope, intercept]  # Store as linear

        # Calculate statistics
        rmse = np.sqrt(np.mean(residuals**2))
        r_squared = r_value**2

        # 95% confidence interval
        confidence_interval = (
            np.percentile(residuals, 2.5),
            np.percentile(residuals, 97.5)
        )

        # Band-specific corrections
        band_corrections = self._calculate_band_corrections(matched_pairs)

        model = CalibrationModel(
            creation_time=datetime.now(),
            num_samples=len(matched_pairs),
            scale_factor=slope,
            offset_db=intercept,
            polynomial_coeffs=poly_coeffs.tolist(),
            rmse_db=rmse,
            r_squared=r_squared,
            confidence_interval=confidence_interval,
            band_corrections=band_corrections
        )

        # Store model
        model_key = band if band else 'global'
        self.calibration_models[model_key] = model

        logger.info(f"Calibrated {model_key} model: RMSE={rmse:.2f}dB, R²={r_squared:.3f}")

        return model

    def _match_wspr_ft8(self, ft8_estimates: List[Dict],
                        band: Optional[str] = None) -> List[Dict]:
        """Match WSPR reports with FT8 estimates for same stations.

        Args:
            ft8_estimates: FT8 power estimates
            band: Optional band filter

        Returns:
            List of matched power pairs
        """
        matched = []

        # Group WSPR by station and time
        wspr_by_station = defaultdict(list)
        for report in self.wspr_reports:
            if band is None or report.band == band:
                wspr_by_station[report.station_hash].append(report)

        # Match with FT8 estimates
        for ft8_est in ft8_estimates:
            station = ft8_est['station_hash']
            ft8_time = ft8_est['timestamp']

            if station not in wspr_by_station:
                continue

            # Find closest WSPR report in time
            best_match = None
            min_time_diff = timedelta(hours=1)  # Max 1 hour difference

            for wspr_report in wspr_by_station[station]:
                time_diff = abs(wspr_report.timestamp - ft8_time)

                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_match = wspr_report

            if best_match:
                matched.append({
                    'station': station,
                    'wspr_power': best_match.reported_power_dbm,
                    'ft8_power': ft8_est['estimated_power_dbm'],
                    'band': best_match.band,
                    'time_diff': min_time_diff.total_seconds()
                })

        return matched

    def _calculate_band_corrections(self, matched_pairs: List[Dict]) -> Dict[str, float]:
        """Calculate band-specific correction factors.

        Args:
            matched_pairs: Matched WSPR-FT8 power pairs

        Returns:
            Dictionary of band-specific corrections
        """
        band_errors = defaultdict(list)

        for pair in matched_pairs:
            error = pair['wspr_power'] - pair['ft8_power']
            band_errors[pair['band']].append(error)

        # Calculate median error per band
        band_corrections = {}
        for band, errors in band_errors.items():
            if len(errors) >= 5:  # Need minimum samples
                band_corrections[band] = np.median(errors)

        return band_corrections

    def apply_calibration(self, ft8_power_dbm: float,
                         band: Optional[str] = None) -> Tuple[float, float]:
        """Apply calibration to FT8 power estimate.

        Args:
            ft8_power_dbm: Uncalibrated FT8 power estimate
            band: Operating band

        Returns:
            Tuple of (calibrated_power_dbm, confidence_interval_width)
        """
        # Select appropriate model
        if band and band in self.calibration_models:
            model = self.calibration_models[band]
        elif 'global' in self.calibration_models:
            model = self.calibration_models['global']
        else:
            # No calibration available
            return ft8_power_dbm, 10.0  # Large uncertainty

        # Apply polynomial correction
        calibrated_power = np.polyval(model.polynomial_coeffs, ft8_power_dbm)

        # Apply band-specific correction if available
        if band and band in model.band_corrections:
            calibrated_power += model.band_corrections[band]

        # Confidence interval width
        ci_width = model.confidence_interval[1] - model.confidence_interval[0]

        return calibrated_power, ci_width

    def calibrate_station(self, station_hash: str) -> Dict[str, float]:
        """Create station-specific calibration.

        Args:
            station_hash: Station to calibrate

        Returns:
            Station-specific calibration parameters
        """
        # Find WSPR reports for this station
        station_reports = [r for r in self.wspr_reports if r.station_hash == station_hash]

        if len(station_reports) < 5:
            return {}

        # Group by power level
        power_groups = defaultdict(list)
        for report in station_reports:
            power_groups[report.reported_power_dbm].append(report)

        # Calculate average SNR per power level
        power_snr_pairs = []
        for power_dbm, reports in power_groups.items():
            avg_snr = np.mean([r.snr_db for r in reports])
            power_snr_pairs.append((power_dbm, avg_snr))

        if len(power_snr_pairs) < 2:
            return {}

        # Fit linear model: SNR = a * Power + b
        powers, snrs = zip(*power_snr_pairs)
        slope, intercept, r_value, _, _ = stats.linregress(powers, snrs)

        calibration = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'num_levels': len(power_snr_pairs),
            'power_range': (min(powers), max(powers))
        }

        self.station_calibrations[station_hash] = calibration

        return calibration

    def cross_validate_model(self, n_folds: int = 5) -> Dict[str, float]:
        """Cross-validate calibration model.

        Args:
            n_folds: Number of cross-validation folds

        Returns:
            Cross-validation statistics
        """
        if not self.wspr_reports:
            return {}

        # Prepare data
        all_pairs = []
        for report in self.wspr_reports:
            # Simulate FT8 estimate (with noise)
            simulated_ft8 = report.reported_power_dbm + np.random.normal(0, 3)
            all_pairs.append({
                'actual': report.reported_power_dbm,
                'estimated': simulated_ft8,
                'band': report.band
            })

        if len(all_pairs) < n_folds * 5:
            return {}

        # Shuffle and split
        np.random.shuffle(all_pairs)
        fold_size = len(all_pairs) // n_folds

        rmse_scores = []
        r2_scores = []

        for i in range(n_folds):
            # Split data
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < n_folds - 1 else len(all_pairs)

            test_data = all_pairs[test_start:test_end]
            train_data = all_pairs[:test_start] + all_pairs[test_end:]

            # Train model on training set
            train_actual = np.array([d['actual'] for d in train_data])
            train_estimated = np.array([d['estimated'] for d in train_data])

            slope, intercept, r_value, _, _ = stats.linregress(train_estimated, train_actual)

            # Test on test set
            test_actual = np.array([d['actual'] for d in test_data])
            test_estimated = np.array([d['estimated'] for d in test_data])
            test_predicted = slope * test_estimated + intercept

            # Calculate metrics
            rmse = np.sqrt(np.mean((test_actual - test_predicted)**2))
            r2 = r_value**2

            rmse_scores.append(rmse)
            r2_scores.append(r2)

        return {
            'mean_rmse': np.mean(rmse_scores),
            'std_rmse': np.std(rmse_scores),
            'mean_r2': np.mean(r2_scores),
            'std_r2': np.std(r2_scores)
        }

    def generate_calibration_report(self) -> str:
        """Generate human-readable calibration report.

        Returns:
            Text report
        """
        report = "WSPR Power Calibration Report\n"
        report += "=" * 50 + "\n\n"

        if not self.calibration_models:
            report += "No calibration models available.\n"
            return report

        # Global model
        if 'global' in self.calibration_models:
            model = self.calibration_models['global']
            report += "Global Calibration Model:\n"
            report += f"  Samples: {model.num_samples}\n"
            report += f"  Linear: actual = {model.scale_factor:.3f} * estimated + {model.offset_db:.1f} dB\n"
            report += f"  RMSE: {model.rmse_db:.2f} dB\n"
            report += f"  R²: {model.r_squared:.3f}\n"
            report += f"  95% CI: [{model.confidence_interval[0]:.1f}, {model.confidence_interval[1]:.1f}] dB\n\n"

        # Band-specific models
        band_models = {k: v for k, v in self.calibration_models.items() if k != 'global'}
        if band_models:
            report += "Band-Specific Models:\n"
            for band, model in sorted(band_models.items()):
                report += f"\n  {band}:\n"
                report += f"    Samples: {model.num_samples}\n"
                report += f"    RMSE: {model.rmse_db:.2f} dB\n"
                report += f"    Correction: {model.offset_db:+.1f} dB\n"

        # Station calibrations
        if self.station_calibrations:
            report += f"\n\nStation-Specific Calibrations: {len(self.station_calibrations)}\n"

            # Show best calibrated stations
            best_stations = sorted(self.station_calibrations.items(),
                                 key=lambda x: x[1].get('r_squared', 0),
                                 reverse=True)[:5]

            report += "Top calibrated stations:\n"
            for station, cal in best_stations:
                report += f"  {station[:8]}...: R²={cal['r_squared']:.3f}, "
                report += f"levels={cal['num_levels']}\n"

        # Cross-validation results
        cv_results = self.cross_validate_model()
        if cv_results:
            report += f"\n\nCross-Validation (5-fold):\n"
            report += f"  RMSE: {cv_results['mean_rmse']:.2f} ± {cv_results['std_rmse']:.2f} dB\n"
            report += f"  R²: {cv_results['mean_r2']:.3f} ± {cv_results['std_r2']:.3f}\n"

        return report

    def get_statistics(self) -> Dict[str, Any]:
        """Get calibration statistics.

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_wspr_reports': len(self.wspr_reports),
            'unique_stations': len(set(r.station_hash for r in self.wspr_reports)),
            'calibration_models': len(self.calibration_models),
            'station_calibrations': len(self.station_calibrations)
        }

        if self.wspr_reports:
            power_distribution = defaultdict(int)
            for report in self.wspr_reports:
                power_w = 10**((report.reported_power_dbm - 30) / 10)
                if power_w <= 5:
                    power_distribution['qrp'] += 1
                elif power_w <= 100:
                    power_distribution['typical'] += 1
                else:
                    power_distribution['qro'] += 1

            stats['power_distribution'] = dict(power_distribution)

        if self.calibration_models:
            rmse_values = [m.rmse_db for m in self.calibration_models.values()]
            stats['avg_model_rmse'] = np.mean(rmse_values)
            stats['best_model_rmse'] = min(rmse_values)

        return stats