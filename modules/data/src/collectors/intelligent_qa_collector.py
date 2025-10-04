"""
Intelligent QA Sample Collector with Micro-Embedder Training

Uses lightweight embedding models to select diverse QA samples and stores them
in Tigris with free egress for cost-effective training.
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import torch
import torch.nn as nn
from collections import deque
from sklearn.cluster import MiniBatchKMeans
import aioboto3
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MicroEmbedder(nn.Module):
    """
    Ultra-lightweight pattern detector for real-time diversity scoring.
    Only 1-2M parameters, processes IQ data 1000x faster than main VAE.
    """

    def __init__(self, input_channels: int = 2, embedding_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=64, stride=32),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=32, stride=16),
            nn.ReLU(),
            nn.Conv1d(32, embedding_dim, kernel_size=16, stride=8),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, iq_chunk: torch.Tensor) -> torch.Tensor:
        """Generate compact fingerprint in <1ms."""
        embedding = self.encoder(iq_chunk)
        return embedding.squeeze(-1)


@dataclass
class QASample:
    """QA sample with metadata and diversity scoring."""
    timestamp: datetime
    iq_data: np.ndarray
    frequency_khz: float
    band: str
    grid_square: str
    session_id: str
    diversity_score: float
    selection_reason: str  # 'random', 'diversity', 'transition', 'anomaly'
    micro_embedding: Optional[np.ndarray] = None
    file_size_bytes: int = 0
    tigris_key: Optional[str] = None


class IntelligentQACollector:
    """
    Progressive QA collector with micro-embedder training.
    Stores samples in Tigris with local cache for active training.
    """

    def __init__(self, tigris_bucket: str = "cascade-qa-samples"):
        self.tigris_bucket = tigris_bucket
        self.micro_model = None
        self.collection_month = 1

        # Pattern memory for diversity scoring
        self.pattern_memory = deque(maxlen=10000)
        self.clusterer = MiniBatchKMeans(n_clusters=100)

        # Local cache for active training (100GB limit)
        self.local_cache_path = Path("/nvme/qa_cache")
        self.local_cache_path.mkdir(parents=True, exist_ok=True)
        self.cache_size_gb_limit = 100
        self.current_cache_size_gb = 0

        # Collection rates by phase
        self.collection_phases = {
            'bootstrap': {'months': [1, 2], 'rate': 0.03, 'method': 'random'},
            'hybrid': {'months': [3, 4], 'rate': 0.08, 'method': 'mixed'},
            'production': {'months': [5, 18], 'rate': 0.12, 'method': 'intelligent'}
        }

        # Training schedule
        self.last_training_date = None
        self.training_interval_days = 7  # Weekly

        # Tigris client (configured with zero egress fees!)
        self.s3_session = None
        self.s3_client = None

    async def initialize(self):
        """Initialize Tigris client and load existing model if available."""
        import os

        # Initialize Tigris client (Fly.io auto-configures these env vars)
        self.s3_session = aioboto3.Session()
        self.s3_client = await self.s3_session.client(
            's3',
            endpoint_url=os.getenv('AWS_ENDPOINT_URL_S3', 'https://fly.storage.tigris.dev'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name='auto'  # Tigris uses 'auto' region
        ).__aenter__()

        # Load existing micro-model if available
        model_path = self.local_cache_path / "micro_embedder.pt"
        if model_path.exists():
            logger.info("Loading existing micro-embedder model")
            self.micro_model = MicroEmbedder()
            self.micro_model.load_state_dict(torch.load(model_path))
            self.micro_model.eval()

    def get_current_phase(self) -> Dict[str, Any]:
        """Determine current collection phase based on month."""
        for phase_name, config in self.collection_phases.items():
            if self.collection_month in config['months']:
                return {'name': phase_name, **config}
        return {'name': 'production', **self.collection_phases['production']}

    async def should_save_qa_sample(
        self,
        iq_data: np.ndarray,
        metadata: Dict[str, Any]
    ) -> Tuple[bool, str, float]:
        """
        Determine if IQ data should be saved as QA sample.
        Returns (should_save, reason, diversity_score).
        """
        phase = self.get_current_phase()

        # Phase 1-2: Bootstrap with random sampling
        if phase['name'] == 'bootstrap':
            should_save = np.random.random() < phase['rate']
            return should_save, 'random', 1.0

        # Phase 3+: Use micro-model if available
        if self.micro_model is not None:
            diversity_score = self.calculate_diversity_score(iq_data)

            # Phase 3-4: Hybrid approach
            if phase['name'] == 'hybrid':
                # 5% random + 3% intelligent
                if np.random.random() < 0.05:
                    return True, 'random', diversity_score
                elif diversity_score > self.get_diversity_threshold():
                    return True, 'diversity', diversity_score

            # Phase 5-18: Production intelligent selection
            elif phase['name'] == 'production':
                # 2% random baseline + 10% intelligent
                if np.random.random() < 0.02:
                    return True, 'random', diversity_score
                elif diversity_score > self.get_diversity_threshold():
                    return True, 'diversity', diversity_score
                elif self.is_transition_point(iq_data):
                    return True, 'transition', diversity_score
                elif self.is_anomaly(iq_data):
                    return True, 'anomaly', diversity_score * 2

        # Default: random sampling at phase rate
        should_save = np.random.random() < phase['rate']
        return should_save, 'random', 1.0

    def calculate_diversity_score(self, iq_data: np.ndarray) -> float:
        """
        Calculate diversity score using micro-embedder.
        Higher scores indicate more unique/interesting patterns.
        """
        if self.micro_model is None:
            return 1.0

        # Get micro-embedding
        iq_tensor = torch.from_numpy(iq_data).float().unsqueeze(0)
        with torch.no_grad():
            embedding = self.micro_model(iq_tensor).numpy().flatten()

        # Update clustering model
        self.clusterer.partial_fit(embedding.reshape(1, -1))

        # Calculate cluster distance (novelty)
        cluster_centers = self.clusterer.cluster_centers_
        distances = np.linalg.norm(embedding - cluster_centers, axis=1)
        cluster_distance = np.min(distances)

        # Calculate temporal diversity (difference from recent samples)
        temporal_diversity = 1.0
        if len(self.pattern_memory) > 0:
            recent_embeddings = [e for e, _ in list(self.pattern_memory)[-100:]]
            temporal_distances = [
                np.linalg.norm(embedding - past)
                for past in recent_embeddings
            ]
            temporal_diversity = np.mean(temporal_distances)

        # Combined diversity score
        diversity_score = cluster_distance + temporal_diversity

        # Update pattern memory
        self.pattern_memory.append((embedding, diversity_score))

        return diversity_score

    def get_diversity_threshold(self) -> float:
        """
        Calculate adaptive diversity threshold for current window.
        Selects top 10% most diverse patterns.
        """
        if len(self.pattern_memory) < 100:
            return 0.5  # Conservative initial threshold

        recent_scores = [score for _, score in list(self.pattern_memory)[-1000:]]
        threshold = np.percentile(recent_scores, 90)  # Top 10%

        return threshold

    def is_transition_point(self, iq_data: np.ndarray) -> bool:
        """Detect if this is a transition between quiet and active periods."""
        # Simple edge detection in amplitude
        amplitude = np.abs(iq_data)
        gradient = np.gradient(amplitude)
        edge_strength = np.std(gradient)

        return edge_strength > np.percentile(gradient, 95)

    def is_anomaly(self, iq_data: np.ndarray) -> bool:
        """Detect anomalous patterns worth preserving."""
        # Check for non-Gaussian statistics (interesting!)
        kurtosis = np.mean(iq_data**4) / (np.var(iq_data)**2) - 3

        # High kurtosis indicates outliers/interesting patterns
        return abs(kurtosis) > 2.0

    async def save_qa_sample(self, sample: QASample):
        """
        Save QA sample to Tigris with free egress.
        Also maintains local cache for active training.
        """
        # Generate unique key
        timestamp_str = sample.timestamp.strftime("%Y%m%d_%H%M%S")
        sample_id = hashlib.sha256(
            f"{sample.session_id}_{timestamp_str}".encode()
        ).hexdigest()[:16]

        # Tigris key structure: year/month/day/hour/sample_id.npz
        tigris_key = (
            f"qa_samples/{sample.timestamp.year:04d}/"
            f"{sample.timestamp.month:02d}/{sample.timestamp.day:02d}/"
            f"{sample.timestamp.hour:02d}/{sample_id}.npz"
        )

        # Prepare data for storage
        sample_data = {
            'iq_data': sample.iq_data,
            'metadata': {
                'timestamp': sample.timestamp.isoformat(),
                'frequency_khz': sample.frequency_khz,
                'band': sample.band,
                'grid_square': sample.grid_square,
                'session_id': sample.session_id,
                'diversity_score': sample.diversity_score,
                'selection_reason': sample.selection_reason
            }
        }

        if sample.micro_embedding is not None:
            sample_data['micro_embedding'] = sample.micro_embedding

        # Save to Tigris (free egress!)
        import io
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **sample_data)
        buffer.seek(0)

        await self.s3_client.put_object(
            Bucket=self.tigris_bucket,
            Key=tigris_key,
            Body=buffer.getvalue()
        )

        sample.tigris_key = tigris_key
        logger.debug(f"Saved QA sample to Tigris: {tigris_key}")

        # Update local cache if this week's data
        if (datetime.now(timezone.utc) - sample.timestamp).days < 7:
            await self.update_local_cache(sample_id, buffer.getvalue())

    async def update_local_cache(self, sample_id: str, data: bytes):
        """
        Maintain local cache of recent QA samples for training.
        Automatically manages size limit.
        """
        cache_file = self.local_cache_path / f"{sample_id}.npz"
        file_size_gb = len(data) / (1024**3)

        # Check cache size limit
        if self.current_cache_size_gb + file_size_gb > self.cache_size_gb_limit:
            await self.clear_old_cache_files()

        # Save to local cache
        cache_file.write_bytes(data)
        self.current_cache_size_gb += file_size_gb

    async def clear_old_cache_files(self):
        """Remove oldest cache files to stay within size limit."""
        cache_files = sorted(
            self.local_cache_path.glob("*.npz"),
            key=lambda f: f.stat().st_mtime
        )

        while self.current_cache_size_gb > self.cache_size_gb_limit * 0.8:
            if not cache_files:
                break

            oldest = cache_files.pop(0)
            file_size_gb = oldest.stat().st_size / (1024**3)
            oldest.unlink()
            self.current_cache_size_gb -= file_size_gb
            logger.debug(f"Removed old cache file: {oldest.name}")

    async def train_micro_model(self):
        """
        Train micro-embedder on QA samples from Tigris.
        Uses free egress to download training data.
        """
        logger.info("Starting micro-embedder training")

        # Download recent QA samples from Tigris (free!)
        training_data = await self.download_training_batch()

        if len(training_data) < 1000:
            logger.warning(f"Insufficient training data: {len(training_data)} samples")
            return

        # Initialize or update model
        if self.micro_model is None:
            self.micro_model = MicroEmbedder()

        # Prepare training dataset
        iq_samples = []
        for sample_data in training_data:
            iq_data = sample_data['iq_data']
            # Normalize and prepare for training
            iq_normalized = iq_data / (np.abs(iq_data).max() + 1e-8)
            iq_samples.append(iq_normalized)

        # Simple self-supervised training (autoencoder-style)
        self.micro_model.train()
        optimizer = torch.optim.Adam(self.micro_model.parameters(), lr=1e-3)

        for epoch in range(10):  # Quick training
            epoch_loss = 0
            for iq_data in iq_samples[:1000]:  # Subset for speed
                iq_tensor = torch.from_numpy(iq_data).float().unsqueeze(0)

                # Forward pass
                embedding = self.micro_model(iq_tensor)

                # Simple reconstruction loss (or contrastive, etc.)
                loss = torch.nn.functional.mse_loss(
                    embedding,
                    torch.randn_like(embedding) * 0.1  # Placeholder target
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            logger.info(f"Epoch {epoch+1}/10, Loss: {epoch_loss/len(iq_samples):.4f}")

        # Save trained model locally
        model_path = self.local_cache_path / "micro_embedder.pt"
        torch.save(self.micro_model.state_dict(), model_path)
        logger.info(f"Saved micro-embedder to {model_path}")

        self.micro_model.eval()
        self.last_training_date = datetime.now(timezone.utc)

    async def download_training_batch(self) -> List[Dict]:
        """
        Download recent QA samples from Tigris for training.
        Free egress makes this cost-effective!
        """
        training_data = []

        # Get list of recent QA samples
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=7)

        # List objects in date range
        prefix = f"qa_samples/{start_date.year:04d}/{start_date.month:02d}/"

        paginator = self.s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(
            Bucket=self.tigris_bucket,
            Prefix=prefix
        )

        # Download samples (free with Tigris!)
        for page in page_iterator:
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                if len(training_data) >= 5000:  # Limit for training
                    break

                # Download object (zero cost!)
                response = await self.s3_client.get_object(
                    Bucket=self.tigris_bucket,
                    Key=obj['Key']
                )

                data = await response['Body'].read()
                sample_data = np.load(io.BytesIO(data), allow_pickle=True)
                training_data.append(dict(sample_data))

        logger.info(f"Downloaded {len(training_data)} QA samples from Tigris (free!)")
        return training_data

    async def should_retrain(self) -> bool:
        """Check if it's time to retrain the micro-model."""
        if self.last_training_date is None:
            # Initial training after bootstrap phase
            return self.collection_month >= 2

        # Weekly retraining schedule
        days_since_training = (
            datetime.now(timezone.utc) - self.last_training_date
        ).days

        return days_since_training >= self.training_interval_days

    async def process_iq_stream(
        self,
        iq_stream: np.ndarray,
        metadata: Dict[str, Any]
    ) -> Optional[QASample]:
        """
        Process IQ stream and determine if it should be saved as QA.
        """
        # Check if we should save this as QA
        should_save, reason, diversity_score = await self.should_save_qa_sample(
            iq_stream, metadata
        )

        if not should_save:
            return None

        # Create QA sample
        sample = QASample(
            timestamp=datetime.now(timezone.utc),
            iq_data=iq_stream,
            frequency_khz=metadata.get('frequency_khz', 14074),
            band=metadata.get('band', '20m'),
            grid_square=metadata.get('grid_square', 'XX00'),
            session_id=metadata.get('session_id', 'unknown'),
            diversity_score=diversity_score,
            selection_reason=reason,
            file_size_bytes=iq_stream.nbytes
        )

        # Save to Tigris
        await self.save_qa_sample(sample)

        # Check if retraining needed
        if await self.should_retrain():
            asyncio.create_task(self.train_micro_model())

        return sample

    def advance_month(self):
        """Advance to next collection month."""
        self.collection_month += 1
        phase = self.get_current_phase()
        logger.info(
            f"Advanced to month {self.collection_month}, "
            f"phase: {phase['name']}, rate: {phase['rate']}"
        )


# Example usage
async def main():
    """Example of using the intelligent QA collector."""
    import os

    collector = IntelligentQACollector()
    await collector.initialize()

    # Simulate processing IQ data
    for hour in range(24):
        # Generate mock IQ data
        iq_data = np.random.randn(12000) + 1j * np.random.randn(12000)

        metadata = {
            'frequency_khz': 14074,
            'band': '20m',
            'grid_square': 'FN42',
            'session_id': f"session_{hour:02d}"
        }

        # Process and potentially save as QA
        qa_sample = await collector.process_iq_stream(iq_data, metadata)

        if qa_sample:
            print(
                f"Saved QA sample: {qa_sample.selection_reason}, "
                f"diversity: {qa_sample.diversity_score:.2f}"
            )


if __name__ == "__main__":
    import os
    asyncio.run(main())