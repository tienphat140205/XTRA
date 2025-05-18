import cupy as cp
import numpy as np
import os
from pathlib import Path
import sys
from cuml.cluster import KMeans as cuKMeans
from cuml.preprocessing import normalize as cuNormalize

ROOT_DIR = os.path.abspath(os.getcwd())  # This will use the current directory (XTRA)

def apply_language_independent_reduction(embeddings, language_indices, dims_to_remove):
    """
    Apply Language Independent Reduction (LIR) to remove language-specific dimensions.
    
    Args:
        embeddings: CuPy array of document embeddings
        language_indices: CuPy array of language indices
        dims_to_remove: Number of dimensions to remove
        
    Returns:
        CuPy array of processed embeddings
    """
    if dims_to_remove <= 0:
        return embeddings.copy()
        
    n_samples, n_features = embeddings.shape
    processed_embeddings = embeddings.copy()
    unique_languages = cp.unique(language_indices)
    
    for lang_id in unique_languages:
        lang_mask = (language_indices == lang_id)
        lang_subset = processed_embeddings[lang_mask]
        n_samples_lang = lang_subset.shape[0]

        if n_samples_lang == 0:
            continue
            
        effective_dims = min(dims_to_remove, n_samples_lang, n_features)
        if effective_dims < 1:
            continue

        try:
            _, _, Vt = cp.linalg.svd(lang_subset, full_matrices=False)
            V = Vt.T
            actual_svd_dims = V.shape[1]
            final_dims = min(effective_dims, actual_svd_dims)
            
            if final_dims < 1:
                continue

            c_L = V[:, :final_dims]
            projection_matrix = c_L @ c_L.T
            identity_matrix = cp.identity(n_features, dtype=embeddings.dtype)
            removal_matrix = identity_matrix - projection_matrix
            processed_embeddings[lang_mask] = (removal_matrix @ lang_subset.T).T

        except cp.linalg.LinAlgError:
            lang_id_cpu = cp.asnumpy(lang_id)
            print(f"LinAlgError in LIR for language {lang_id_cpu}. Skipping LIR for this language.", flush=True)
        except Exception as e:
            lang_id_cpu = cp.asnumpy(lang_id)
            print(f"Unknown error in LIR for language {lang_id_cpu}: {e}. Skipping LIR for this language.", flush=True)
            
    cp.cuda.stream.get_current_stream().synchronize()
    return processed_embeddings


def perform_svd_reduction(embeddings, dimensions):
    """
    Perform dimensionality reduction using SVD.
    
    Args:
        embeddings: CuPy array of embeddings
        dimensions: Target number of dimensions
        
    Returns:
        Tuple of (u_svd_result, svd_lr_result)
    """
    n_samples, n_features = embeddings.shape
    effective_dims = min(dimensions, n_samples, n_features)

    if effective_dims < 1:
        print(f"Warning: Effective SVD dimension ({effective_dims}) < 1. Truncating to safe dimension.", flush=True)
        return truncate_embeddings(embeddings, dimensions)

    try:
        u, s, _ = cp.linalg.svd(embeddings, full_matrices=False)
        computed_dims = min(effective_dims, len(s))
        
        if computed_dims < 1:
            raise ValueError("SVD resulted in < 1 singular value")

        u_comp = u[:, :computed_dims]
        s_comp = s[:computed_dims]

        u_svd_result = u_comp
        svd_lr_result = u_comp * s_comp
        cp.cuda.stream.get_current_stream().synchronize()
        return u_svd_result, svd_lr_result

    except Exception as e:
        print(f"Error during SVD reduction: {e}. Falling back to truncation.", flush=True)
        return truncate_embeddings(embeddings, dimensions)


def truncate_embeddings(embeddings, dimensions):
    """
    Truncate embeddings to specified dimensions as fallback for SVD.
    
    Args:
        embeddings: CuPy array of embeddings
        dimensions: Target number of dimensions
        
    Returns:
        Tuple of (truncated embeddings, truncated embeddings)
    """
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
    """
    Run K-means clustering on embeddings.
    
    Args:
        embeddings: CuPy array of normalized embeddings
        num_clusters: Number of clusters (k)
        n_samples: Number of samples
        
    Returns:
        KMeans model or None if clustering fails
    """
    effective_k = min(num_clusters, n_samples)
    
    if effective_k < 1 or embeddings.shape[1] < 1 or n_samples < effective_k:
        print(f"Error: Cannot run KMeans (K={num_clusters}, EffectiveK={effective_k}, "
              f"ActualSamples={n_samples}, Features={embeddings.shape[1]}). Skipping K-Means.", flush=True)
        return None
        
    try:
        kmeans_model = cuKMeans(n_clusters=effective_k, random_state=0, max_iter=300, output_type='cupy')
        print(f"Running KMeans with K={effective_k}, ActualSamples={n_samples}, "
              f"Features={embeddings.shape[1]}...", flush=True)
        kmeans_model.fit(embeddings)
        cp.cuda.stream.get_current_stream().synchronize()
        print("KMeans finished.", flush=True)
        return kmeans_model
    except Exception as e:
        print(f"Error during K-Means fitting: {e}", flush=True)
        return None


def process_dataset(dataset_name, lang1, lang2, svd_dimensions, lir_dimensions, kmeans_clusters):
    """
    Process a dataset by applying LIR, SVD reduction, and K-means clustering.
    
    Args:
        dataset_name: Name of the dataset
        lang1: First language code
        lang2: Second language code
        svd_dimensions: Number of dimensions to keep in SVD
        lir_dimensions: Number of dimensions to remove in LIR
        kmeans_clusters: Number of clusters for K-means
    """
    base_path = os.path.join(ROOT_DIR, "data")

    embed_path_lang1 = Path(f"{base_path}/{dataset_name}/doc_embeddings_{lang1}_train.npy")
    embed_path_lang2 = Path(f"{base_path}/{dataset_name}/doc_embeddings_{lang2}_train.npy")
    save_dir = Path(f"{base_path}/{dataset_name}")

    # Check if files exist
    if not embed_path_lang1.exists() or not embed_path_lang2.exists():
        print(f"File does not exist: {embed_path_lang1 if not embed_path_lang1.exists() else embed_path_lang2}", flush=True)
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    try:
        print(f"Loading embeddings for {lang1} from {embed_path_lang1}", flush=True)
        embed_lang1_np = np.load(embed_path_lang1).astype(np.float32)
        print(f"Loading embeddings for {lang2} from {embed_path_lang2}", flush=True)
        embed_lang2_np = np.load(embed_path_lang2).astype(np.float32)
    except Exception as e:
        print(f"Error loading embedding files: {e}", flush=True)
        return

    # Validate embeddings
    if embed_lang1_np.size == 0 or embed_lang2_np.size == 0:
        print(f"Error: Empty embeddings for {dataset_name}. Lang1 size: {embed_lang1_np.size}, "
              f"Lang2 size: {embed_lang2_np.size}", flush=True)
        return
    if embed_lang1_np.shape[1] != embed_lang2_np.shape[1]:
        print(f"Error: Dimension mismatch for {dataset_name}. Lang1 shape: {embed_lang1_np.shape}, "
              f"Lang2 shape: {embed_lang2_np.shape}", flush=True)
        return

    n_features_orig = embed_lang1_np.shape[1]
    print(f"Original feature dimension: {n_features_orig}", flush=True)

    # Prepare data
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

    # Move data to GPU
    try:
        embeddings_cp = cp.asarray(embeddings_np)
        lang_index_cp = cp.asarray(lang_index_np)
        del embeddings_np, lang_index_np
    except Exception as e:
        print(f"Error moving data to GPU: {e}", flush=True)
        cp.get_default_memory_pool().free_all_blocks()
        return

    # Apply Language Independent Reduction (LIR)
    print(f"Applying LIR with dimensions to remove={lir_dimensions}...", flush=True)
    embeddings_lir_cp = apply_language_independent_reduction(
        embeddings_cp, lang_index_cp, dims_to_remove=lir_dimensions
    )
    print(f"Shape after LIR: {embeddings_lir_cp.shape}", flush=True)
    del embeddings_cp

    # Perform SVD reduction
    print(f"Performing SVD reduction to {svd_dimensions} dimensions...", flush=True)
    embed_usvd_cp, _ = perform_svd_reduction(embeddings_lir_cp, dimensions=svd_dimensions)
    print(f"Shape after u-SVD: {embed_usvd_cp.shape}", flush=True)
    del embeddings_lir_cp

    # Validate SVD results
    if embed_usvd_cp.size == 0 or embed_usvd_cp.shape[1] == 0:
        print(f"Error: Invalid or empty SVD (u-SVD) result for {dataset_name}. "
              f"Shape: {embed_usvd_cp.shape}", flush=True)
        cp.get_default_memory_pool().free_all_blocks()
        return

    chosen_method_name = "u-SVD (Cosine via L2Norm)"
    embeddings_reduced_cp = embed_usvd_cp

    # Normalize embeddings
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

    print(f"Shape of normalized {lang1} embeddings (selected by index 0): {embeddings_lang1_norm_cp.shape}", flush=True)
    print(f"Shape of normalized {lang2} embeddings (selected by index 1): {embeddings_lang2_norm_cp.shape}", flush=True)

    del embeddings_normalized_cp, lang_index_cp

    # Run K-means on second language embeddings
    print(f"Running K-Means on {lang2} embeddings only (Samples: {n_samples_lang2_eff})...", flush=True)
    kmeans_lang2_model = run_kmeans(embeddings_lang2_norm_cp, kmeans_clusters, n_samples_lang2_eff)

    if kmeans_lang2_model is None:
        print(f"Could not perform K-Means for {lang2} in dataset {dataset_name}. Skipping.", flush=True)
        del embeddings_lang1_norm_cp, embeddings_lang2_norm_cp
        cp.get_default_memory_pool().free_all_blocks()
        return

    labels_lang2_cp = kmeans_lang2_model.labels_
    centroids_lang2_cp = kmeans_lang2_model.cluster_centers_
    effective_k = kmeans_lang2_model.n_clusters
    print(f"K-Means on {lang2} completed. Found {effective_k} clusters.", flush=True)
    del embeddings_lang2_norm_cp

    # Assign first language embeddings to nearest second language cluster
    print(f"Assigning {lang1} embeddings (Samples: {n_samples_lang1_eff}) to the nearest {lang2} "
          f"cluster centroid (based on Cosine Similarity)...", flush=True)
    cosine_similarities_cp = cp.dot(embeddings_lang1_norm_cp, centroids_lang2_cp.T)
    labels_lang1_cp = cp.argmax(cosine_similarities_cp, axis=1)
    print(f"Assignment of {lang1} embeddings finished.", flush=True)
    del embeddings_lang1_norm_cp, centroids_lang2_cp, cosine_similarities_cp

    # Move results back to CPU
    try:
        labels_lang1_np = cp.asnumpy(labels_lang1_cp)
        labels_lang2_np = cp.asnumpy(labels_lang2_cp)
        del labels_lang1_cp, labels_lang2_cp
    except Exception as e:
        print(f"Error moving results from GPU to CPU: {e}", flush=True)
        cp.get_default_memory_pool().free_all_blocks()
        return
    finally:
        cp.get_default_memory_pool().free_all_blocks()

    # Save results
    save_path_lang1 = save_dir / f"cluster_labels_{lang1}_cosine.npy"
    save_path_lang2 = save_dir / f"cluster_labels_{lang2}_cosine.npy"

    try:
        np.save(save_path_lang1, labels_lang1_np)
        np.save(save_path_lang2, labels_lang2_np)
        print(f"Saved {lang1} cluster labels (assigned) to: {save_path_lang1}", flush=True)
        print(f"Saved {lang2} cluster labels (from K-Means) to: {save_path_lang2}", flush=True)
    except Exception as e:
        print(f"Error saving cluster label files: {e}", flush=True)
        return

    # Print statistics
    print(f"\nDataset: {dataset_name}", flush=True)
    print(f"Method: K-Means on {lang2} ({chosen_method_name}), Assign {lang1} based on "
          f"Cosine Similarity with {lang2} cluster centers.", flush=True)
    print(f"Effective number of clusters (from {lang2} K-Means): {effective_k}", flush=True)
    print("Statistics of document counts per language in each cluster:")
    for k_idx in range(effective_k):
        lang1_count_in_k = np.sum(labels_lang1_np == k_idx)
        lang2_count_in_k = np.sum(labels_lang2_np == k_idx)
        print(f"  Cluster {k_idx}: {lang1_count_in_k} documents from {lang1} (assigned), "
              f"{lang2_count_in_k} documents from {lang2} (clustered)", flush=True)


def main():
    """Main function to process datasets."""
    print("Starting clustering process with new method (KMeans on 'en'):", flush=True)
    print("1. K-Means on 'en' embeddings (after LIR, SVD, Normalize).", flush=True)
    print("2. Assign other language embeddings ('cn'/'ja') to nearest 'en' cluster by Cosine Similarity.\n", flush=True)

    SVD_DIM = 100
    LIR_DIMS_TO_REMOVE = 0
    KMEANS_CLUSTERS = 50

    datasets_to_process = [
        ('Amazon_Review', 'cn', 'en'),
        ('ECNews', 'cn', 'en'),
        ('Rakuten_Amazon', 'ja', 'en')
    ]

    for i, (dataset, lang1, lang2) in enumerate(datasets_to_process):
        if lang2 != 'en':
            print(f"Warning: Current code assumes second language ({lang2}) is 'en' for K-Means. "
                  f"Skipping dataset {dataset} ({lang1}, {lang2})", flush=True)
            continue
        print(f"--- Processing dataset: {dataset} ({lang1} vs {lang2}) ---", flush=True)
        process_dataset(
            dataset_name=dataset,
            lang1=lang1,
            lang2=lang2,
            svd_dimensions=SVD_DIM,
            lir_dimensions=LIR_DIMS_TO_REMOVE,
            kmeans_clusters=KMEANS_CLUSTERS
        )
        if i < len(datasets_to_process) - 1:
            print("\n", flush=True)
        sys.stdout.flush()

    print("\n--- Clustering process completed ---", flush=True)


if __name__ == "__main__":
    main()