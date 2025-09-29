# Embedding Analytics and Storage Strategies

Channel embeddings are not merely compressed representations of radio data - they form a rich analytical space that reveals propagation patterns, guides training strategies, and enables scientific discovery. This document explores how to store, analyze, and leverage embeddings for optimal CASCADE training.

## Selective Dual Storage Strategy

Rather than choosing between database and filesystem storage, CASCADE should use selective dual storage that maximizes analytical value while minimizing costs. This approach stores ALL embeddings in files (cheap, complete) while selectively storing high-value embeddings in the database (expensive, queryable).

### Cost Analysis

For 50 million embeddings (approximately 48GB of raw data), the storage costs vary significantly:

**PostgreSQL Database Storage**
- Base storage: 48 GB
- Indexing overhead: +30% for efficient queries
- TOAST compression: -20% for array storage
- Total size: ~53 GB
- AWS RDS cost: $0.115/GB/month = $6.10/month
- Advantages: Full SQL queries, ACID compliance, concurrent access

**Filesystem Storage (HDF5/NumPy)**
- Raw NumPy arrays: 48 GB
- HDF5 with compression: 34 GB (30% reduction)
- AWS EBS cost: $0.023/GB/month = $0.78/month
- Advantages: Direct memory mapping, bulk I/O performance

**Selective Dual Storage (Recommended)**
- PostgreSQL for high-value embeddings (~10%): 10-20 GB
- HDF5 for ALL embeddings (100%): 34 GB compressed
- Total cost: ~$11/month ($10 DB + $0.78 files)
- Maximum value: Complete archive + advanced analytics on interesting data

### What to Store Where

The selective dual storage strategy optimizes both cost and analytical capability:

```python
class SelectiveDualStorage:
    """
    Complete archive in files, high-value subset in database
    """

    def __init__(self):
        # PostgreSQL stores high-value embeddings with full vectors
        self.db = psycopg2.connect("dbname=cascade")

        # HDF5 stores ALL embeddings
        self.h5file = h5py.File('embeddings.h5', 'a')

    def store_embedding(self, embedding, metadata):
        # ALWAYS store in HDF5 (complete archive)
        embedding_id = str(uuid.uuid4())
        self.h5file.create_dataset(
            f'embeddings/{embedding_id}',
            data=embedding,
            compression='gzip',
            compression_opts=4
        )

        # SELECTIVELY store in PostgreSQL (high-value only)
        if self.is_high_value(metadata):
            with self.db.cursor() as cur:
                cur.execute("""
                    INSERT INTO high_value_embeddings
                    (id, vector, timestamp, k_index, snr,
                     x_ray_class, propagation_mode, is_anomaly)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    embedding_id,
                    embedding.tolist(),  # Full vector for analytics
                    metadata['timestamp'],
                    metadata.get('k_index'),
                    metadata.get('snr'),
                    metadata.get('x_ray_class'),
                    metadata.get('propagation_mode'),
                    metadata.get('is_anomaly', False)
                ))

    def is_high_value(self, metadata):
        """
        Criteria for database storage
        """
        # Always store rare/interesting events
        if any([
            metadata.get('k_index', 0) >= 5,           # Geomagnetic storms
            metadata.get('x_ray_class') in ['M', 'X'], # Solar flares
            metadata.get('snr', 0) < -20,              # Extreme weak signals
            metadata.get('propagation_mode') in ['Aurora', 'TEP', 'MS'],
            metadata.get('is_anomaly', False),
            metadata.get('is_transition', False),
        ]):
            return True

        # 1% random sample for statistical validity
        if random.random() < 0.01:
            return True

        return False

    def query_similar(self, query_embedding, conditions, k=100):
        # Coarse filter using PostgreSQL (fast)
        with self.db.cursor() as cur:
            cur.execute("""
                SELECT id, h5_path
                FROM embedding_metadata
                WHERE k_index = %s
                AND embedding_preview <-> %s < 0.5
                ORDER BY embedding_preview <-> %s
                LIMIT %s
            """, (
                conditions['k_index'],
                query_embedding[:8],
                query_embedding[:8],
                k * 3  # Over-retrieve for fine ranking
            ))
            candidates = cur.fetchall()

        # Fine ranking with full embeddings from HDF5
        full_embeddings = []
        for id, h5_path in candidates:
            full_embeddings.append(self.h5file[h5_path][:])

        # Exact distance computation
        distances = cdist([query_embedding], full_embeddings)[0]
        top_k_indices = np.argsort(distances)[:k]

        return [(candidates[i][0], distances[i])
                for i in top_k_indices]
```

## Propagation Discovery Through Clustering

Embeddings naturally cluster into propagation modes, revealing both known and potentially unknown phenomena.

### Unsupervised Propagation Mode Discovery

The embedding space self-organizes into distinct regions corresponding to different propagation mechanisms:

```python
def discover_propagation_modes(embeddings, metadata):
    """
    Identify natural clusters in propagation space
    """
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA

    # Reduce dimensions for initial clustering
    pca = PCA(n_components=10)
    reduced = pca.fit_transform(embeddings)

    # Density-based clustering finds natural groupings
    clusters = DBSCAN(eps=0.3, min_samples=50).fit(reduced)

    # Analyze each discovered cluster
    cluster_profiles = {}
    for cluster_id in set(clusters.labels_):
        if cluster_id == -1:  # Outliers
            continue

        mask = clusters.labels_ == cluster_id
        cluster_data = metadata[mask]

        # Profile the cluster characteristics
        profile = {
            'size': mask.sum(),
            'avg_snr': cluster_data['snr'].mean(),
            'distance_distribution': cluster_data['distance_km'].describe(),
            'time_preference': cluster_data['hour'].mode(),
            'k_index_correlation': cluster_data['k_index'].corr(mask),
            'seasonal_pattern': cluster_data['month'].value_counts(),
            'geographic_bias': cluster_data['path_type'].value_counts()
        }

        # Attempt to identify the propagation mode
        if profile['k_index_correlation'] > 0.7:
            profile['likely_mode'] = 'Auroral'
        elif profile['time_preference'] in [10, 11, 12, 13, 14]:
            profile['likely_mode'] = 'F2 Layer'
        elif profile['seasonal_pattern'].idxmax() in [5, 6, 7]:
            profile['likely_mode'] = 'Sporadic-E'
        else:
            profile['likely_mode'] = 'Unknown - Investigate!'

        cluster_profiles[cluster_id] = profile

    return clusters, cluster_profiles
```

### Anomaly Detection for Rare Propagation

Embeddings far from cluster centers often represent scientifically interesting propagation:

```python
def find_rare_propagation_events(embeddings, metadata):
    """
    Identify unusual propagation worthy of investigation
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    # Ensemble approach for robust anomaly detection
    iso_forest = IsolationForest(contamination=0.01)
    lof = LocalOutlierFactor(contamination=0.01)

    iso_scores = iso_forest.fit_predict(embeddings)
    lof_scores = lof.fit_predict(embeddings)

    # Events flagged by both methods are truly unusual
    rare_mask = (iso_scores == -1) & (lof_scores == -1)
    rare_embeddings = embeddings[rare_mask]
    rare_metadata = metadata[rare_mask]

    # Categorize the anomalies
    anomaly_types = []
    for idx in np.where(rare_mask)[0]:
        event = metadata.iloc[idx]

        # Ultra-long distance with high SNR
        if event['distance_km'] > 15000 and event['snr'] > 0:
            anomaly_types.append('Super-propagation')

        # Wrong time for the distance (e.g., long path at noon)
        elif event['hour'] in [11, 12, 13] and event['distance_km'] > 10000:
            anomaly_types.append('Anomalous daytime DX')

        # Extremely rapid fading
        elif 'fading_rate' in event and event['fading_rate'] > 10:
            anomaly_types.append('Rapid flutter fading')

        else:
            anomaly_types.append('Unknown anomaly')

    # These rare events get priority in training
    training_weights = np.ones(len(embeddings))
    training_weights[rare_mask] = 10.0  # 10x weight for anomalies

    return rare_embeddings, anomaly_types, training_weights
```

## Temporal Evolution Analysis

Embeddings change over time in ways that reveal propagation dynamics and guide training focus.

### Propagation Transition Detection

Sudden changes in embedding space indicate propagation mode transitions that challenge decoders:

```python
def analyze_temporal_evolution(embeddings_sequence):
    """
    Track how embeddings evolve and find critical transitions
    """
    # Compute embedding velocity (rate of change)
    embedding_velocity = np.diff(embeddings_sequence, axis=0)
    velocity_magnitude = np.linalg.norm(embedding_velocity, axis=1)

    # Find sudden transitions
    transition_threshold = np.percentile(velocity_magnitude, 95)
    transitions = np.where(velocity_magnitude > transition_threshold)[0]

    # Characterize each transition
    transition_analysis = []
    for t in transitions:
        before = embeddings_sequence[max(0, t-5):t].mean(axis=0)
        after = embeddings_sequence[t+1:min(len(embeddings_sequence), t+6)].mean(axis=0)

        # What changed?
        change_vector = after - before

        # Map to interpretable features (learned from labeled data)
        transition_type = classify_transition(change_vector)

        transition_analysis.append({
            'time_index': t,
            'magnitude': velocity_magnitude[t],
            'type': transition_type,  # 'F2_to_Es', 'sunrise', etc.
            'training_priority': 'high'  # Focus training here
        })

    return transition_analysis
```

### Diurnal and Seasonal Patterns

Embeddings reveal cyclic patterns that inform when to collect more data:

```python
def extract_cyclic_patterns(embeddings_with_time):
    """
    Identify gaps in temporal coverage
    """
    from scipy import signal
    import pandas as pd

    df = pd.DataFrame({
        'embedding_mean': embeddings_with_time.mean(axis=1),
        'timestamp': metadata['timestamp']
    })

    # Hourly aggregation
    hourly = df.groupby(df.timestamp.dt.hour)['embedding_mean'].agg(['mean', 'std', 'count'])

    # Find undersampled hours
    undersampled_hours = hourly[hourly['count'] < hourly['count'].median() * 0.5].index

    # Seasonal analysis
    monthly = df.groupby(df.timestamp.dt.month)['embedding_mean'].agg(['mean', 'std', 'count'])
    undersampled_months = monthly[monthly['count'] < monthly['count'].mean() * 0.5].index

    # Fourier analysis for periodic patterns
    fft = np.fft.fft(df['embedding_mean'].values)
    frequencies = np.fft.fftfreq(len(fft), d=1/24)  # Hourly sampling

    # Dominant periods
    power = np.abs(fft)**2
    dominant_periods = 1 / frequencies[np.argsort(power)[-5:]]  # Top 5 periods

    return {
        'undersampled_hours': undersampled_hours,
        'undersampled_months': undersampled_months,
        'dominant_periods_hours': dominant_periods,
        'collection_guidance': f"Focus on hours {undersampled_hours.tolist()} and months {undersampled_months.tolist()}"
    }
```

## Geographic Pattern Analysis

Embeddings encode path-dependent propagation characteristics that vary by geography.

### Path Clustering

Different geographic paths exhibit distinct embedding signatures:

```python
def analyze_geographic_patterns(embeddings, paths):
    """
    Discover how geography affects propagation
    """
    from sklearn.manifold import TSNE
    import networkx as nx

    # Group by path
    path_embeddings = {}
    for path_id, indices in paths.items():
        path_embeddings[path_id] = {
            'mean': embeddings[indices].mean(axis=0),
            'std': embeddings[indices].std(axis=0),
            'count': len(indices)
        }

    # Build path similarity network
    G = nx.Graph()
    for path1, data1 in path_embeddings.items():
        for path2, data2 in path_embeddings.items():
            if path1 < path2:  # Avoid duplicates
                similarity = 1 / (1 + np.linalg.norm(data1['mean'] - data2['mean']))
                if similarity > 0.7:  # Threshold for similar paths
                    G.add_edge(path1, path2, weight=similarity)

    # Find communities of similar paths
    communities = nx.community.louvain_communities(G)

    # Characterize each community
    community_profiles = []
    for community in communities:
        paths_in_community = list(community)

        # Common characteristics
        profile = analyze_path_characteristics(paths_in_community)

        # Often reveals patterns like:
        # - All polar paths cluster together
        # - Transequatorial paths form distinct group
        # - Mountain diffraction paths cluster
        # - Over-water vs over-land separation

        community_profiles.append(profile)

    return community_profiles
```

## Training Strategy Optimization

Embedding analysis directly informs CASCADE training strategies.

### Curriculum Learning from Embeddings

Train progressively from typical to challenging conditions:

```python
def design_curriculum(embeddings, performance_scores):
    """
    Order training examples from easy to hard
    """
    from sklearn.cluster import KMeans

    # Find cluster centers (typical examples)
    kmeans = KMeans(n_clusters=50)
    kmeans.fit(embeddings)

    # Distance to nearest center = difficulty
    difficulties = []
    for embedding in embeddings:
        distances_to_centers = [
            np.linalg.norm(embedding - center)
            for center in kmeans.cluster_centers_
        ]
        difficulty = min(distances_to_centers)
        difficulties.append(difficulty)

    # Curriculum stages
    difficulties = np.array(difficulties)
    curriculum = {
        'stage_1_easy': np.where(difficulties < np.percentile(difficulties, 33))[0],
        'stage_2_medium': np.where(
            (difficulties >= np.percentile(difficulties, 33)) &
            (difficulties < np.percentile(difficulties, 67))
        )[0],
        'stage_3_hard': np.where(difficulties >= np.percentile(difficulties, 67))[0]
    }

    # Additional focus on transitions
    transition_indices = find_embedding_transitions(embeddings)
    curriculum['stage_4_transitions'] = transition_indices

    return curriculum
```

### Diversity-Based Batch Selection

Select training batches that maximize coverage of embedding space:

```python
def select_diverse_batch(embeddings, batch_size=1000):
    """
    Maximum diversity batch selection
    """
    selected = []
    remaining = list(range(len(embeddings)))

    # Start with embedding furthest from mean
    mean_embedding = embeddings.mean(axis=0)
    distances_from_mean = [
        np.linalg.norm(emb - mean_embedding)
        for emb in embeddings
    ]
    first_idx = np.argmax(distances_from_mean)
    selected.append(first_idx)
    remaining.remove(first_idx)

    # Iteratively add most distant from selected set
    while len(selected) < batch_size and remaining:
        max_min_distance = -1
        best_candidate = None

        for candidate in remaining:
            # Minimum distance to any selected point
            min_distance = min([
                np.linalg.norm(embeddings[candidate] - embeddings[s])
                for s in selected
            ])

            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_candidate = candidate

        selected.append(best_candidate)
        remaining.remove(best_candidate)

    return selected
```

### Performance Correlation Analysis

Identify which embedding dimensions predict CASCADE performance:

```python
def analyze_performance_correlations(embeddings, decode_success):
    """
    Find embedding features that predict success/failure
    """
    correlations = []

    # Per-dimension correlation with success
    for dim in range(embeddings.shape[1]):
        corr = np.corrcoef(embeddings[:, dim], decode_success)[0, 1]
        correlations.append({
            'dimension': dim,
            'correlation': corr,
            'abs_correlation': abs(corr)
        })

    # Sort by importance
    correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)

    # Top predictive dimensions
    critical_dims = [c['dimension'] for c in correlations[:10]]

    # Build predictive model
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(embeddings[:, critical_dims], decode_success)

    # Feature importance from random forest
    feature_importance = rf.feature_importances_

    # Training focus: emphasize examples with extreme values in critical dimensions
    focus_mask = np.zeros(len(embeddings), dtype=bool)
    for dim in critical_dims[:3]:  # Top 3 most important
        dim_values = embeddings[:, dim]
        # Focus on extremes
        focus_mask |= (dim_values < np.percentile(dim_values, 10))
        focus_mask |= (dim_values > np.percentile(dim_values, 90))

    return {
        'critical_dimensions': critical_dims,
        'feature_importance': feature_importance,
        'focus_indices': np.where(focus_mask)[0],
        'prediction_model': rf
    }
```

## Storage Implementation Guidelines

### PostgreSQL Schema for Hybrid Storage

```sql
-- Metadata table with embedding previews for fast search
CREATE TABLE embedding_metadata (
    embedding_id UUID PRIMARY KEY,
    h5_path VARCHAR(255) NOT NULL,  -- Path in HDF5 file

    -- Searchable metadata
    timestamp TIMESTAMP NOT NULL,
    frequency_hz INTEGER NOT NULL,
    k_index INTEGER,
    solar_flux INTEGER,
    snr_db FLOAT,
    distance_km FLOAT,
    propagation_mode VARCHAR(20),

    -- Embedding preview for coarse similarity
    embedding_preview FLOAT[16],  -- First 16 dimensions
    embedding_norm FLOAT,  -- L2 norm for distance calculations

    -- Cluster assignments from analysis
    cluster_id INTEGER,
    anomaly_score FLOAT,
    difficulty_score FLOAT,

    -- Performance tracking
    used_in_training BOOLEAN DEFAULT FALSE,
    training_weight FLOAT DEFAULT 1.0,
    decode_success_rate FLOAT,

    -- Indexes for efficient queries
    INDEX idx_temporal (timestamp, frequency_hz),
    INDEX idx_conditions (k_index, solar_flux),
    INDEX idx_cluster (cluster_id),
    INDEX idx_anomaly (anomaly_score) WHERE anomaly_score > 0.9,
    INDEX idx_training (used_in_training, training_weight)
);

-- Install pgvector for similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Add vector column for similarity searches
ALTER TABLE embedding_metadata
ADD COLUMN embedding_preview_vec vector(16);

-- Create index for fast similarity search
CREATE INDEX embedding_preview_idx
ON embedding_metadata
USING ivfflat (embedding_preview_vec vector_cosine_ops)
WITH (lists = 100);

-- Materialized view for training batch generation
CREATE MATERIALIZED VIEW training_candidates AS
SELECT
    e.embedding_id,
    e.h5_path,
    e.embedding_preview_vec,
    e.difficulty_score,
    e.anomaly_score,
    e.cluster_id,
    e.training_weight,
    e.decode_success_rate
FROM embedding_metadata e
WHERE e.training_weight > 0
ORDER BY
    e.anomaly_score DESC,  -- Prioritize rare events
    e.difficulty_score DESC  -- Then difficult examples
WITH DATA;
```

### HDF5 Organization

```python
def organize_hdf5_storage():
    """
    Efficient HDF5 structure for embeddings
    """
    with h5py.File('embeddings.h5', 'w') as f:
        # Hierarchical organization by time and frequency
        # /2025/01/15/3573khz/embeddings
        # /2025/01/15/3573khz/metadata

        # Create groups for each day
        for date in dates:
            date_group = f.create_group(date.strftime('%Y/%m/%d'))

            for frequency in CASCADE_FREQUENCIES:
                freq_group = date_group.create_group(f'{frequency}khz')

                # Store embeddings as chunked dataset for efficient access
                embeddings_dataset = freq_group.create_dataset(
                    'embeddings',
                    shape=(0, embedding_dim),
                    maxshape=(None, embedding_dim),
                    chunks=(1000, embedding_dim),  # 1000 embeddings per chunk
                    compression='gzip',
                    compression_opts=4
                )

                # Store associated metadata
                metadata_dataset = freq_group.create_dataset(
                    'metadata',
                    shape=(0,),
                    maxshape=(None,),
                    dtype=metadata_dtype,
                    chunks=(1000,),
                    compression='gzip'
                )

        # Add attributes for quick reference
        f.attrs['total_embeddings'] = 0
        f.attrs['embedding_dimension'] = embedding_dim
        f.attrs['creation_date'] = datetime.now().isoformat()
        f.attrs['CASCADE_frequencies'] = CASCADE_FREQUENCIES
```

## Conclusions

Channel embeddings are far more than compressed data storage - they are a window into the physics of radio propagation and a powerful tool for optimizing CASCADE training. The hybrid storage approach combining PostgreSQL's query capabilities with HDF5's bulk storage efficiency provides the ideal platform for both operational training and scientific discovery.

Key insights from embedding analytics:

1. **Propagation modes naturally cluster** in embedding space, potentially revealing unknown phenomena
2. **Anomalies indicate rare events** that deserve extra training weight
3. **Temporal transitions** mark challenging conditions requiring focused training
4. **Geographic patterns** emerge that inform path-specific model adaptation
5. **Curriculum learning** can be derived directly from embedding structure
6. **Performance prediction** from embedding features guides training focus

The embedding space is not just a means to an end - it's a rich analytical framework that continuously improves our understanding of HF propagation and CASCADE's training needs.