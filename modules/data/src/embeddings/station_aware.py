"""Station-aware embedding generation for propagation modeling.

T075: Generate embeddings that incorporate station fingerprints for improved
propagation prediction while maintaining privacy.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class StationEmbedding:
    """Station-specific embedding for propagation modeling."""

    station_hash: str
    embedding_vector: np.ndarray  # Dense vector representation
    timestamp: datetime

    # Components of the embedding
    technical_features: np.ndarray  # Technical characteristics
    temporal_features: np.ndarray   # Activity patterns
    geographic_features: np.ndarray # Grid-based location
    behavioral_features: np.ndarray # Operating patterns

    # Metadata
    embedding_version: str = "1.0.0"
    dimensionality: int = 128


class StationAwareEmbeddingGenerator:
    """Generates privacy-preserving embeddings from station fingerprints."""

    def __init__(self, embedding_dim: int = 128):
        """Initialize embedding generator.

        Args:
            embedding_dim: Dimensionality of output embeddings
        """
        self.embedding_dim = embedding_dim
        self.embeddings_cache: Dict[str, StationEmbedding] = {}

        # Component dimensions (must sum to embedding_dim)
        self.technical_dim = 32
        self.temporal_dim = 32
        self.geographic_dim = 32
        self.behavioral_dim = 32

        # Feature normalization parameters (learned from data)
        self.feature_stats = {
            'snr_mean': 0.0,
            'snr_std': 15.0,
            'freq_stability_mean': 0.0,
            'freq_stability_std': 10.0,
            'duty_cycle_mean': 20.0,
            'duty_cycle_std': 15.0
        }

    def generate_embedding(self, fingerprint: Dict[str, Any]) -> StationEmbedding:
        """Generate embedding from station fingerprint.

        Args:
            fingerprint: Station fingerprint dictionary

        Returns:
            StationEmbedding object
        """
        station_hash = fingerprint['station_hash']

        # Check cache
        if station_hash in self.embeddings_cache:
            cached = self.embeddings_cache[station_hash]
            # Update if fingerprint is newer
            if fingerprint.get('last_seen', datetime.now()) <= cached.timestamp:
                return cached

        # Generate component embeddings
        technical = self._encode_technical_features(fingerprint)
        temporal = self._encode_temporal_features(fingerprint)
        geographic = self._encode_geographic_features(fingerprint)
        behavioral = self._encode_behavioral_features(fingerprint)

        # Concatenate into full embedding
        embedding_vector = np.concatenate([
            technical, temporal, geographic, behavioral
        ])

        # L2 normalize for cosine similarity
        embedding_vector = embedding_vector / (np.linalg.norm(embedding_vector) + 1e-8)

        embedding = StationEmbedding(
            station_hash=station_hash,
            embedding_vector=embedding_vector,
            timestamp=datetime.now(),
            technical_features=technical,
            temporal_features=temporal,
            geographic_features=geographic,
            behavioral_features=behavioral,
            dimensionality=self.embedding_dim
        )

        # Cache the embedding
        self.embeddings_cache[station_hash] = embedding

        return embedding

    def _encode_technical_features(self, fingerprint: Dict) -> np.ndarray:
        """Encode technical characteristics.

        Args:
            fingerprint: Station fingerprint

        Returns:
            Technical feature vector
        """
        features = np.zeros(self.technical_dim)

        # SNR characteristics (normalized)
        avg_snr = fingerprint.get('avg_snr_db', 0)
        snr_norm = (avg_snr - self.feature_stats['snr_mean']) / self.feature_stats['snr_std']
        features[0] = np.tanh(snr_norm)  # Bounded [-1, 1]

        # SNR variance
        features[1] = np.tanh(fingerprint.get('snr_variance', 0) / 10)

        # Frequency stability
        freq_stability = fingerprint.get('frequency_stability_ppm', 0)
        features[2] = np.tanh(freq_stability / self.feature_stats['freq_stability_std'])

        # Frequency drift
        features[3] = np.tanh(fingerprint.get('frequency_drift_hz_per_min', 0) / 5)

        # Power characteristics
        power_dbm = fingerprint.get('typical_power_dbm', 30)
        features[4] = (power_dbm - 30) / 20  # Normalize around 30 dBm

        # Phase noise
        if 'phase_noise_db' in fingerprint:
            features[5] = np.tanh(fingerprint['phase_noise_db'] / 60)

        # IMD3
        if 'imd3_db' in fingerprint:
            features[6] = np.tanh(fingerprint['imd3_db'] / 40)

        # Modulation quality
        if 'modulation_quality' in fingerprint:
            features[7] = fingerprint['modulation_quality']

        # Band usage pattern (one-hot encoding for primary bands)
        bands = fingerprint.get('primary_bands', [])
        band_map = {'160m': 8, '80m': 9, '40m': 10, '30m': 11, '20m': 12,
                   '17m': 13, '15m': 14, '12m': 15, '10m': 16, '6m': 17}
        for band in bands[:5]:  # Limit to 5 bands
            if band in band_map:
                features[band_map[band]] = 1.0

        return features

    def _encode_temporal_features(self, fingerprint: Dict) -> np.ndarray:
        """Encode temporal activity patterns.

        Args:
            fingerprint: Station fingerprint

        Returns:
            Temporal feature vector
        """
        features = np.zeros(self.temporal_dim)

        # Activity hours (24-hour circular encoding)
        active_hours = fingerprint.get('active_hours_utc', [])
        for hour in active_hours:
            # Circular encoding to preserve 23->0 continuity
            features[hour % 24] = 1.0

        # Day of week pattern
        active_days = fingerprint.get('active_days', [])
        for day in active_days:
            features[24 + day] = 1.0  # Offset by 24 for hour features

        # Duty cycle
        duty_cycle = fingerprint.get('duty_cycle', 0)
        features[31] = duty_cycle / 100.0  # Last position

        return features

    def _encode_geographic_features(self, fingerprint: Dict) -> np.ndarray:
        """Encode geographic characteristics from grid squares.

        Args:
            fingerprint: Station fingerprint

        Returns:
            Geographic feature vector
        """
        features = np.zeros(self.geographic_dim)

        primary_grid = fingerprint.get('primary_grid', 'JJ00')  # Default to null island

        if len(primary_grid) >= 4:
            # Encode field (first two letters)
            field_lat = ord(primary_grid[1]) - ord('A')  # 0-17
            field_lon = ord(primary_grid[0]) - ord('A')  # 0-17

            # Normalize to [-1, 1]
            features[0] = (field_lon - 8.5) / 8.5
            features[1] = (field_lat - 8.5) / 8.5

            # Encode square (two digits)
            square_lon = int(primary_grid[2])  # 0-9
            square_lat = int(primary_grid[3])  # 0-9

            features[2] = (square_lon - 4.5) / 4.5
            features[3] = (square_lat - 4.5) / 4.5

            # Encode subsquare if available
            if len(primary_grid) >= 6:
                sub_lon = ord(primary_grid[4].upper()) - ord('A')  # 0-23
                sub_lat = ord(primary_grid[5].upper()) - ord('A')  # 0-23

                features[4] = (sub_lon - 11.5) / 11.5
                features[5] = (sub_lat - 11.5) / 11.5

        # Grid square diversity (how many different grids observed)
        grid_count = len(fingerprint.get('grid_squares', []))
        features[6] = np.tanh(grid_count / 10)  # Normalize with tanh

        # Maximum observed distance
        max_distance = fingerprint.get('max_distance_km', 0)
        features[7] = np.tanh(max_distance / 10000)  # Normalize for 10,000 km

        # Median distance
        median_distance = fingerprint.get('median_distance_km', 0)
        features[8] = np.tanh(median_distance / 5000)

        # Geographic consistency (placeholder for future lat/lon encoding)
        # Reserve features[9:31] for future geographic encoding

        return features

    def _encode_behavioral_features(self, fingerprint: Dict) -> np.ndarray:
        """Encode operating behavior patterns.

        Args:
            fingerprint: Station fingerprint

        Returns:
            Behavioral feature vector
        """
        features = np.zeros(self.behavioral_dim)

        # Message type distribution
        message_types = fingerprint.get('message_types', {})
        total_messages = sum(message_types.values()) if message_types else 1

        # Normalize message type frequencies
        type_indices = {'CQ': 0, 'QSO': 1, 'BEACON': 2, 'GRID': 3, 'OTHER': 4}
        for msg_type, count in message_types.items():
            if msg_type in type_indices:
                features[type_indices[msg_type]] = count / total_messages

        # QSO characteristics
        qso_duration = fingerprint.get('qso_duration_avg_min', 0)
        features[5] = np.tanh(qso_duration / 30)  # Normalize for 30-minute QSOs

        response_time = fingerprint.get('response_time_avg_sec', 0)
        features[6] = np.tanh(response_time / 60)  # Normalize for 60-second response

        # Station persistence (how long active)
        if 'first_seen' in fingerprint and 'last_seen' in fingerprint:
            first = fingerprint['first_seen']
            last = fingerprint['last_seen']
            if isinstance(first, str):
                first = datetime.fromisoformat(first)
            if isinstance(last, str):
                last = datetime.fromisoformat(last)

            active_days = (last - first).days
            features[7] = np.tanh(active_days / 365)  # Normalize for one year

        # Observation count (activity level)
        obs_count = fingerprint.get('total_observations', 0)
        features[8] = np.tanh(obs_count / 1000)

        # Reserved for future behavioral metrics
        # features[9:31] available

        return features

    def compute_similarity(self, embedding1: StationEmbedding,
                         embedding2: StationEmbedding) -> float:
        """Compute cosine similarity between two station embeddings.

        Args:
            embedding1, embedding2: Station embeddings to compare

        Returns:
            Similarity score [0, 1]
        """
        # Cosine similarity (vectors are already L2 normalized)
        similarity = np.dot(embedding1.embedding_vector, embedding2.embedding_vector)

        # Convert from [-1, 1] to [0, 1]
        return (similarity + 1) / 2

    def compute_component_similarities(self, embedding1: StationEmbedding,
                                      embedding2: StationEmbedding) -> Dict[str, float]:
        """Compute similarity for each embedding component.

        Args:
            embedding1, embedding2: Station embeddings to compare

        Returns:
            Dictionary of component similarities
        """
        def cosine_sim(v1, v2):
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return np.dot(v1, v2) / (norm1 * norm2)

        return {
            'technical': cosine_sim(embedding1.technical_features,
                                  embedding2.technical_features),
            'temporal': cosine_sim(embedding1.temporal_features,
                                 embedding2.temporal_features),
            'geographic': cosine_sim(embedding1.geographic_features,
                                   embedding2.geographic_features),
            'behavioral': cosine_sim(embedding1.behavioral_features,
                                   embedding2.behavioral_features),
            'overall': self.compute_similarity(embedding1, embedding2)
        }

    def find_similar_stations(self, station_hash: str,
                             threshold: float = 0.8,
                             max_results: int = 10) -> List[Tuple[str, float]]:
        """Find stations with similar embeddings.

        Args:
            station_hash: Reference station
            threshold: Minimum similarity score
            max_results: Maximum number of results

        Returns:
            List of (station_hash, similarity) tuples
        """
        if station_hash not in self.embeddings_cache:
            return []

        reference = self.embeddings_cache[station_hash]
        similar = []

        for other_hash, other_embedding in self.embeddings_cache.items():
            if other_hash == station_hash:
                continue

            similarity = self.compute_similarity(reference, other_embedding)
            if similarity >= threshold:
                similar.append((other_hash, similarity))

        # Sort by similarity and limit results
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar[:max_results]

    def cluster_stations(self, min_cluster_size: int = 5) -> Dict[int, List[str]]:
        """Cluster stations based on embedding similarity.

        Args:
            min_cluster_size: Minimum stations per cluster

        Returns:
            Dictionary mapping cluster ID to station hashes
        """
        if len(self.embeddings_cache) < min_cluster_size:
            return {0: list(self.embeddings_cache.keys())}

        # Extract embedding vectors
        station_hashes = list(self.embeddings_cache.keys())
        vectors = np.array([self.embeddings_cache[h].embedding_vector
                           for h in station_hashes])

        # Simple k-means clustering
        from sklearn.cluster import KMeans

        # Determine optimal number of clusters
        n_clusters = min(10, len(vectors) // min_cluster_size)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(vectors)

        # Group stations by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(station_hashes[i])

        # Filter out small clusters
        return {k: v for k, v in clusters.items() if len(v) >= min_cluster_size}

    def update_feature_statistics(self, fingerprints: List[Dict]):
        """Update normalization statistics from fingerprint data.

        Args:
            fingerprints: List of station fingerprints
        """
        if not fingerprints:
            return

        # Collect feature values
        snr_values = [fp.get('avg_snr_db', 0) for fp in fingerprints]
        freq_stability = [fp.get('frequency_stability_ppm', 0) for fp in fingerprints]
        duty_cycles = [fp.get('duty_cycle', 0) for fp in fingerprints]

        # Update statistics
        if snr_values:
            self.feature_stats['snr_mean'] = np.mean(snr_values)
            self.feature_stats['snr_std'] = np.std(snr_values) + 1e-8

        if freq_stability:
            self.feature_stats['freq_stability_mean'] = np.mean(freq_stability)
            self.feature_stats['freq_stability_std'] = np.std(freq_stability) + 1e-8

        if duty_cycles:
            self.feature_stats['duty_cycle_mean'] = np.mean(duty_cycles)
            self.feature_stats['duty_cycle_std'] = np.std(duty_cycles) + 1e-8

        logger.info(f"Updated feature statistics from {len(fingerprints)} fingerprints")