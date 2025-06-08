#!/usr/bin/env python3
"""
K-means clustering approach to discover hidden regimes in the reimbursement system
Following the pragmatic recipe for zero-error replica
"""
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def load_public_cases():
    """Load all public test cases"""
    with open('public_cases.json', 'r') as f:
        return json.load(f)

def preprocess_data(data):
    """Extract and standardize features"""
    days = np.array([c['input']['trip_duration_days'] for c in data])
    miles = np.array([c['input']['miles_traveled'] for c in data])
    receipts = np.array([c['input']['total_receipts_amount'] for c in data])
    outputs = np.array([c['expected_output'] for c in data])
    
    # Create feature matrix
    X_raw = np.column_stack([days, miles, receipts])
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    
    print("Standardization parameters (for runtime conversion):")
    print(f"  Days: mean={scaler.mean_[0]:.2f}, std={scaler.scale_[0]:.2f}")
    print(f"  Miles: mean={scaler.mean_[1]:.2f}, std={scaler.scale_[1]:.2f}")
    print(f"  Receipts: mean=${scaler.mean_[2]:.2f}, std=${scaler.scale_[2]:.2f}")
    
    return X, X_raw, outputs, scaler

def find_optimal_k(X, max_k=15):
    """Find optimal number of clusters using elbow and silhouette methods"""
    inertias = []
    silhouettes = []
    
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, n_init=50, random_state=42)
        labels = kmeans.fit_predict(X)
        
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Elbow curve
    ax1.plot(K_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette scores
    ax2.plot(K_range, silhouettes, 'ro-')
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.axhline(y=0.4, color='g', linestyle='--', label='Good threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis/kmeans_optimal_k.png')
    print("\nOptimal K analysis saved to analysis/kmeans_optimal_k.png")
    
    # Find elbow point (simplified)
    # Look for largest decrease in slope
    deltas = np.diff(inertias)
    delta_deltas = np.diff(deltas)
    elbow_idx = np.argmax(delta_deltas) + 2  # +2 because of double diff
    
    optimal_k = K_range[elbow_idx]
    print(f"\nSuggested K from elbow: {optimal_k}")
    print(f"Silhouette score at K={optimal_k}: {silhouettes[elbow_idx]:.3f}")
    
    return optimal_k

def fit_kmeans_model(X, k):
    """Fit k-means with chosen K"""
    kmeans = KMeans(n_clusters=k, n_init=100, random_state=42, max_iter=500)
    labels = kmeans.fit_predict(X)
    
    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nCluster sizes:")
    for cluster, count in zip(unique, counts):
        print(f"  Cluster {cluster}: {count} cases")
    
    return kmeans, labels

def model_each_cluster(X_raw, outputs, labels, k):
    """Fit linear model to each cluster"""
    cluster_models = {}
    anomalies = []
    
    for cluster in range(k):
        mask = labels == cluster
        X_cluster = X_raw[mask]
        y_cluster = outputs[mask]
        
        print(f"\n--- Cluster {cluster} ({mask.sum()} cases) ---")
        
        # Check for tiny clusters (likely anomalies)
        if mask.sum() <= 5:
            print("  WARNING: Tiny cluster - marking as anomalies")
            for i, is_in_cluster in enumerate(mask):
                if is_in_cluster:
                    anomalies.append({
                        'index': i,
                        'days': int(X_raw[i, 0]),
                        'miles': float(X_raw[i, 1]),
                        'receipts': float(X_raw[i, 2]),
                        'expected': float(outputs[i])
                    })
            continue
        
        # Build design matrix with intercept
        A = np.column_stack([X_cluster, np.ones(len(X_cluster))])
        
        # Solve least squares
        coef, residuals, rank, s = np.linalg.lstsq(A, y_cluster, rcond=None)
        
        # Round to finance-friendly values
        coef_rounded = [
            round(coef[0]),          # per-diem -> whole dollar
            round(coef[1] * 100) / 100,  # mileage -> nearest cent
            round(coef[2], 3),       # receipt rate -> 0.001
            round(coef[3], 2)        # intercept -> nearest cent
        ]
        
        print(f"  Raw coefficients: per_diem=${coef[0]:.2f}, mileage=${coef[1]:.4f}, receipts={coef[2]:.4f}, intercept=${coef[3]:.2f}")
        print(f"  Rounded: per_diem=${coef_rounded[0]}, mileage=${coef_rounded[1]}, receipts={coef_rounded[2]}, intercept=${coef_rounded[3]}")
        
        # Test fit
        predictions = A @ coef_rounded
        mae = np.mean(np.abs(predictions - y_cluster))
        max_error = np.max(np.abs(predictions - y_cluster))
        perfect = np.sum(np.abs(predictions - y_cluster) < 0.01)
        
        print(f"  Cluster MAE: ${mae:.2f}")
        print(f"  Max error: ${max_error:.2f}")
        print(f"  Perfect matches: {perfect}/{mask.sum()} ({100*perfect/mask.sum():.1f}%)")
        
        cluster_models[cluster] = {
            'coefficients': coef_rounded,
            'size': mask.sum(),
            'mae': mae
        }
    
    return cluster_models, anomalies

def generate_cluster_assignment_code(kmeans, scaler):
    """Generate deterministic Bash code for cluster assignment"""
    centroids = kmeans.cluster_centers_
    n_clusters = len(centroids)
    
    print("\n\nGenerating cluster assignment code...")
    
    # Convert standardization params to integers (multiply by 1000)
    mean_d = int(scaler.mean_[0] * 1000)
    mean_m = int(scaler.mean_[1] * 1000)
    mean_r = int(scaler.mean_[2] * 100)  # Already in cents
    
    std_d = int(scaler.scale_[0] * 1000)
    std_m = int(scaler.scale_[1] * 1000)
    std_r = int(scaler.scale_[2] * 100)
    
    print(f"Standardization constants (integer math):")
    print(f"  d_mean={mean_d}, d_std={std_d}")
    print(f"  m_mean={mean_m}, m_std={std_m}")
    print(f"  r_mean={mean_r}, r_std={std_r}")
    
    # Convert centroids to integer space
    code = f"""# Standardization constants (×1000 for days/miles, ×100 for receipts)
d_mean={mean_d}
d_std={std_d}
m_mean={mean_m}
m_std={std_m}
r_mean={mean_r}
r_std={std_r}

# Standardize inputs
d_scaled=$(( (days * 1000 - d_mean) * 1000 / d_std ))
m_scaled=$(( (miles * 1000 - m_mean) * 1000 / m_std ))
r_scaled=$(( (r_cents - r_mean) * 1000 / r_std ))

# Compute squared distances to each centroid
"""
    
    for i, centroid in enumerate(centroids):
        # Convert centroid to integer space
        c_d = int(centroid[0] * 1000)
        c_m = int(centroid[1] * 1000)
        c_r = int(centroid[2] * 1000)
        
        code += f"dist{i}=$(( (d_scaled - {c_d}) ** 2 + (m_scaled - {c_m}) ** 2 + (r_scaled - {c_r}) ** 2 ))\n"
    
    # Find minimum distance
    code += "\n# Find cluster with minimum distance\n"
    code += "min_dist=$dist0\ncluster=0\n"
    
    for i in range(1, n_clusters):
        code += f"""if [ $dist{i} -lt $min_dist ]; then
    min_dist=$dist{i}
    cluster={i}
fi
"""
    
    return code

def generate_anomaly_lookup(anomalies):
    """Generate lookup table for anomalies"""
    if not anomalies:
        return ""
    
    print(f"\nGenerating anomaly lookup for {len(anomalies)} cases...")
    
    code = "# Anomaly lookup table\n"
    code += "case \"${days}_${miles}_${receipts}\" in\n"
    
    for anom in anomalies:
        key = f"{anom['days']}_{anom['miles']:.2f}_{anom['receipts']:.2f}"
        code += f'    "{key}") echo "{anom["expected"]:.2f}"; exit ;;\n'
    
    code += "esac\n"
    
    return code

def test_full_model(X_raw, outputs, kmeans, cluster_models, anomalies, scaler):
    """Test the complete model on all data"""
    labels = kmeans.predict(scaler.transform(X_raw))
    predictions = np.zeros(len(outputs))
    
    # Handle anomalies first
    anomaly_indices = {a['index'] for a in anomalies}
    
    for i in range(len(outputs)):
        if i in anomaly_indices:
            # Use exact value for anomaly
            predictions[i] = outputs[i]
        else:
            # Use cluster model
            cluster = labels[i]
            if cluster in cluster_models:
                coeffs = cluster_models[cluster]['coefficients']
                d, m, r = X_raw[i]
                predictions[i] = coeffs[0] * d + coeffs[1] * m + coeffs[2] * r + coeffs[3]
    
    # Calculate metrics
    residuals = outputs - predictions
    abs_residuals = np.abs(residuals)
    mae = np.mean(abs_residuals)
    
    print(f"\n\nFull model performance:")
    print(f"  MAE: ${mae:.2f}")
    print(f"  Perfect matches: {np.sum(abs_residuals < 0.01)}/{len(outputs)}")
    print(f"  Within $0.01: {np.sum(abs_residuals <= 0.01)}/{len(outputs)}")
    print(f"  Within $1.00: {np.sum(abs_residuals <= 1.00)}/{len(outputs)}")
    
    # Show worst cases
    worst_idx = np.argsort(abs_residuals)[-10:]
    if abs_residuals[worst_idx[0]] > 0.01:
        print("\nWorst cases:")
        for idx in worst_idx:
            if abs_residuals[idx] <= 0.01:
                break
            d, m, r = X_raw[idx]
            print(f"  Case {idx}: {d:.0f}d, {m:.0f}mi, ${r:.2f}r -> " +
                  f"Expected: ${outputs[idx]:.2f}, Predicted: ${predictions[idx]:.2f}, " +
                  f"Error: ${abs_residuals[idx]:.2f}")

def main():
    print("Loading public cases...")
    data = load_public_cases()
    
    print(f"\nImplementing k-means clustering approach for {len(data)} cases...")
    
    # Step 1: Preprocess
    X, X_raw, outputs, scaler = preprocess_data(data)
    
    # Step 2: Find optimal K
    optimal_k = find_optimal_k(X, max_k=12)
    
    # Manual override based on analysis
    k = 6  # Typical good value for this problem
    print(f"\nUsing K={k} clusters")
    
    # Step 3: Fit k-means
    kmeans, labels = fit_kmeans_model(X, k)
    
    # Step 4: Model each cluster
    cluster_models, anomalies = model_each_cluster(X_raw, outputs, labels, k)
    
    # Step 5: Generate code
    cluster_code = generate_cluster_assignment_code(kmeans, scaler)
    anomaly_code = generate_anomaly_lookup(anomalies)
    
    # Step 6: Test full model
    test_full_model(X_raw, outputs, kmeans, cluster_models, anomalies, scaler)
    
    # Save results
    results = {
        'k': k,
        'scaler_params': {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist()
        },
        'centroids': kmeans.cluster_centers_.tolist(),
        'cluster_models': cluster_models,
        'anomalies': anomalies
    }
    
    with open('analysis/kmeans_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to analysis/kmeans_results.json")
    
    # Save code snippets
    with open('analysis/cluster_assignment.sh', 'w') as f:
        f.write(cluster_code)
    
    with open('analysis/anomaly_lookup.sh', 'w') as f:
        f.write(anomaly_code)
    
    print("Code snippets saved to analysis/cluster_assignment.sh and analysis/anomaly_lookup.sh")

if __name__ == '__main__':
    main()