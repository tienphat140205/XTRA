import cupy as cp
import numpy as np
import os
from pathlib import Path
import sys
from cuml.cluster import KMeans as cuKMeans
from cuml.preprocessing import normalize as cuNormalize

ROOT_DIR = os.path.abspath(os.getcwd())  # current dir (XTRA)

def perform_svd_reduction(embeddings, dimensions):
    n_samples, n_features = embeddings.shape
    effective_dims = min(dimensions, n_samples, n_features)
    if effective_dims < 1:
        return truncate_embeddings(embeddings, dimensions)
    u, s, _ = cp.linalg.svd(embeddings, full_matrices=False)
    computed_dims = min(effective_dims, len(s))
    u_comp = u[:, :computed_dims]
    s_comp = s[:computed_dims]
    u_svd_result = u_comp
    svd_lr_result = u_comp * s_comp
    cp.cuda.stream.get_current_stream().synchronize()
    return u_svd_result, svd_lr_result

def truncate_embeddings(embeddings, dimensions):
    n_samples, n_features = embeddings.shape
    safe_dim = min(n_features, max(dimensions, 1) if dimensions > 0 else 1)
    if n_features < safe_dim:
        safe_dim = n_features
    if safe_dim == 0 and n_features > 0:
        safe_dim = 1
    if safe_dim == 0:
        empty_cp = cp.empty((n_samples, 0), dtype=embeddings.dtype)
        print(f"Returning truncated embeddings of shape: {empty_cp.shape}", flush=True)
        return empty_cp, empty_cp
    truncated = embeddings[:, :safe_dim]
    print(f"Returning truncated embeddings of shape: {truncated.shape}", flush=True)
    return truncated, truncated

def run_kmeans(embeddings, num_clusters, n_samples):
    effective_k = min(num_clusters, n_samples)
    print(f"Running KMeans with K={effective_k}, Samples={n_samples}, Features={embeddings.shape[1]}...", flush=True)
    kmeans_model = cuKMeans(n_clusters=effective_k, random_state=0, max_iter=300, output_type='cupy')
    kmeans_model.fit(embeddings)
    cp.cuda.stream.get_current_stream().synchronize()
    print("KMeans finished.", flush=True)
    return kmeans_model

def process_dataset(dataset_name, lang1, lang2, svd_dimensions, kmeans_clusters):
    base_path = os.path.join(ROOT_DIR, "data")

    embed_path_lang1 = Path(f"{base_path}/{dataset_name}/doc_embeddings_{lang1}_train.npy")
    embed_path_lang2 = Path(f"{base_path}/{dataset_name}/doc_embeddings_{lang2}_train.npy")
    save_dir = Path(f"{base_path}/{dataset_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading embeddings for {lang1} from {embed_path_lang1}", flush=True)
    embed_lang1_np = np.load(embed_path_lang1).astype(np.float32)
    print(f"Loading embeddings for {lang2} from {embed_path_lang2}", flush=True)
    embed_lang2_np = np.load(embed_path_lang2).astype(np.float32)

    n_features_orig = embed_lang1_np.shape[1]
    print(f"Original feature dimension: {n_features_orig}", flush=True)

    n_samples_lang1 = embed_lang1_np.shape[0]
    n_samples_lang2 = embed_lang2_np.shape[0]
    print(f"Samples loaded: {lang1}={n_samples_lang1}, {lang2}={n_samples_lang2}", flush=True)

    lang_index_np = np.concatenate([
        np.zeros(n_samples_lang1, dtype=np.int32),
        np.ones(n_samples_lang2, dtype=np.int32)
    ])
    embeddings_np = np.concatenate([embed_lang1_np, embed_lang2_np], axis=0)
    n_total_samples = embeddings_np.shape[0]
    print(f"Total samples concatenated: {n_total_samples}", flush=True)
    del embed_lang1_np, embed_lang2_np

    # GPU
    embeddings_cp = cp.asarray(embeddings_np)
    lang_index_cp = cp.asarray(lang_index_np)
    del embeddings_np, lang_index_np

    # SVD only
    print(f"Performing SVD reduction to {svd_dimensions} dimensions...", flush=True)
    embed_usvd_cp, _ = perform_svd_reduction(embeddings_cp, dimensions=svd_dimensions)
    print(f"Shape after u-SVD: {embed_usvd_cp.shape}", flush=True)
    del embeddings_cp

    chosen_method_name = "u-SVD (Cosine via L2Norm)"
    embeddings_reduced_cp = embed_usvd_cp

    # Normalize
    print("Normalizing embeddings (L2)...", flush=True)
    embeddings_normalized_cp = cuNormalize(embeddings_reduced_cp, norm='l2', axis=1)
    del embeddings_reduced_cp

    # Split by language
    lang1_mask_cp = (lang_index_cp == 0)
    lang2_mask_cp = (lang_index_cp == 1)

    embeddings_lang1_norm_cp = embeddings_normalized_cp[lang1_mask_cp]
    embeddings_lang2_norm_cp = embeddings_normalized_cp[lang2_mask_cp]

    n_samples_lang1_eff = embeddings_lang1_norm_cp.shape[0]
    n_samples_lang2_eff = embeddings_lang2_norm_cp.shape[0]

    print(f"Shape of normalized {lang1}: {embeddings_lang1_norm_cp.shape}", flush=True)
    print(f"Shape of normalized {lang2}: {embeddings_lang2_norm_cp.shape}", flush=True)

    del embeddings_normalized_cp, lang_index_cp

    # KMeans on lang2 (assumed 'en')
    print(f"Running K-Means on {lang2} only (Samples: {n_samples_lang2_eff})...", flush=True)
    kmeans_lang2_model = run_kmeans(embeddings_lang2_norm_cp, kmeans_clusters, n_samples_lang2_eff)

    labels_lang2_cp = kmeans_lang2_model.labels_
    centroids_lang2_cp = kmeans_lang2_model.cluster_centers_
    effective_k = kmeans_lang2_model.n_clusters
    print(f"K-Means on {lang2} completed. Found {effective_k} clusters.", flush=True)
    del embeddings_lang2_norm_cp

    # Assign lang1 to nearest lang2 centroid by cosine
    print(f"Assigning {lang1} to nearest {lang2} centroid (cosine)...", flush=True)
    cosine_similarities_cp = cp.dot(embeddings_lang1_norm_cp, centroids_lang2_cp.T)
    labels_lang1_cp = cp.argmax(cosine_similarities_cp, axis=1)
    print(f"Assignment of {lang1} finished.", flush=True)
    del embeddings_lang1_norm_cp, centroids_lang2_cp, cosine_similarities_cp

    # To CPU
    labels_lang1_np = cp.asnumpy(labels_lang1_cp)
    labels_lang2_np = cp.asnumpy(labels_lang2_cp)
    del labels_lang1_cp, labels_lang2_cp
    cp.get_default_memory_pool().free_all_blocks()

    # Save
    save_path_lang1 = save_dir / f"cluster_labels_{lang1}_cosine2.npy"
    save_path_lang2 = save_dir / f"cluster_labels_{lang2}_cosine2.npy"
    np.save(save_path_lang1, labels_lang1_np)
    np.save(save_path_lang2, labels_lang2_np)
    print(f"Saved {lang1} labels to: {save_path_lang1}", flush=True)
    print(f"Saved {lang2} labels to: {save_path_lang2}", flush=True)

    # Stats
    print(f"\nDataset: {dataset_name}", flush=True)
    print(f"Method: K-Means on {lang2} ({chosen_method_name}), assign {lang1} by cosine.", flush=True)
    print(f"Effective clusters: {effective_k}", flush=True)
    print("Docs per cluster:")
    for k_idx in range(effective_k):
        lang1_count_in_k = np.sum(labels_lang1_np == k_idx)
        lang2_count_in_k = np.sum(labels_lang2_np == k_idx)
        print(f"  Cluster {k_idx}: {lang1_count_in_k} ({lang1}), {lang2_count_in_k} ({lang2})", flush=True)

def main():
    print("Start: KMeans on 'en', assign other language by cosine\n", flush=True)
    SVD_DIM = 100
    KMEANS_CLUSTERS = 50

    datasets_to_process = [
        ('Amazon_Review', 'cn', 'en'),
        ('ECNews', 'cn', 'en'),
        ('Rakuten_Amazon', 'ja', 'en')
    ]

    for i, (dataset, lang1, lang2) in enumerate(datasets_to_process):
        if lang2 != 'en':
            print(f"Warning: expects second language 'en'. Skipping {dataset} ({lang1}, {lang2})", flush=True)
            continue
        print(f"--- Processing: {dataset} ({lang1} vs {lang2}) ---", flush=True)
        process_dataset(
            dataset_name=dataset,
            lang1=lang1,
            lang2=lang2,
            svd_dimensions=SVD_DIM,
            kmeans_clusters=KMEANS_CLUSTERS
        )
        if i < len(datasets_to_process) - 1:
            print("\n", flush=True)
        sys.stdout.flush()

    print("\n--- Done ---", flush=True)

if __name__ == "__main__":
    main()
