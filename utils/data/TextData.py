import os
import numpy as np
import scipy
import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

from utils.data import file_utils


class BilingualTextDataset(Dataset):    
    def __init__(self, bow_en, bow_cn, doc_embeddings_en=None, doc_embeddings_cn=None,
                 clusterinfo_en=None, clusterinfo_cn=None):

        self.bow_en = bow_en
        self.bow_cn = bow_cn
        self.doc_embeddings_en = doc_embeddings_en
        self.doc_embeddings_cn = doc_embeddings_cn
        self.clusterinfo_en = clusterinfo_en
        self.clusterinfo_cn = clusterinfo_cn
        self.bow_size_en = len(self.bow_en)
        self.bow_size_cn = len(self.bow_cn)

    def __len__(self):
        return max(self.bow_size_en, self.bow_size_cn)

    def __getitem__(self, index):
        en_idx = index % self.bow_size_en
        cn_idx = index % self.bow_size_cn
        
        return_dict = {
            'bow_en': self.bow_en[en_idx],
            'bow_cn': self.bow_cn[cn_idx],
            'doc_embedding_en': self.doc_embeddings_en[en_idx] if self.doc_embeddings_en is not None else None,
            'doc_embedding_cn': self.doc_embeddings_cn[cn_idx] if self.doc_embeddings_cn is not None else None,
            'cluster_en': self.clusterinfo_en[en_idx] if self.clusterinfo_en is not None else None,
            'cluster_cn': self.clusterinfo_cn[cn_idx] if self.clusterinfo_cn is not None else None,
        }

        return return_dict


class DatasetHandler:
    
    def __init__(self, dataset, batch_size, lang1, lang2, n_topics=50, device=0):

        data_dir = f'./data/{dataset}'
        self.device = device
        self.batch_size = batch_size
        self.data_dir = data_dir

        # Merge split files if necessary
        self._merge_split_embeddings(data_dir, lang1, lang2)

        # Load data for both languages
        self.train_texts_en, self.test_texts_en, self.train_bow_matrix_en, self.test_bow_matrix_en, \
        self.vocab_en = self.read_data(data_dir, lang=lang1)
        
        self.train_texts_cn, self.test_texts_cn, self.train_bow_matrix_cn, self.test_bow_matrix_cn, \
        self.vocab_cn = self.read_data(data_dir, lang=lang2)

        # Set dimensions
        self.train_size_en = len(self.train_texts_en)
        self.train_size_cn = len(self.train_texts_cn)
        self.vocab_size_en = len(self.vocab_en)
        self.vocab_size_cn = len(self.vocab_cn)

        self.doc_embeddings_en = np.load(os.path.join(data_dir, f'doc_embeddings_{lang1}_train.npy'))
        self.doc_embeddings_cn = np.load(os.path.join(data_dir, f'doc_embeddings_{lang2}_train.npy'))
        
        # Load cluster information
        self.clusterinfo_en = np.load(os.path.join(data_dir, f'cluster_labels_{lang1}_cosine.npy'))
        self.clusterinfo_cn = np.load(os.path.join(data_dir, f'cluster_labels_{lang2}_cosine.npy'))
        
        self.beta_en = self.calculate_beta(
            self.train_bow_matrix_en, self.clusterinfo_en, self.vocab_size_en, num_topics=n_topics
        )
        self.beta_cn = self.calculate_beta(
            self.train_bow_matrix_cn, self.clusterinfo_cn, self.vocab_size_cn, num_topics=n_topics
        )

        self.mu_prior, self.var_prior = self.calculate_cluster_based_prior(num_topics=n_topics)
        
        # Move data to CUDA if available
        self.doc_embeddings_en, self.doc_embeddings_cn = self.move_to_cuda(self.doc_embeddings_en, self.doc_embeddings_cn)
        self.clusterinfo_en, self.clusterinfo_cn = self.move_to_cuda(self.clusterinfo_en, self.clusterinfo_cn)       
        self.train_bow_matrix_en, self.test_bow_matrix_en = self.move_to_cuda(
            self.train_bow_matrix_en, self.test_bow_matrix_en)
        self.train_bow_matrix_cn, self.test_bow_matrix_cn = self.move_to_cuda(
            self.train_bow_matrix_cn, self.test_bow_matrix_cn)

        # Create data loaders
        self.train_loader = DataLoader(
            BilingualTextDataset(
                self.train_bow_matrix_en, self.train_bow_matrix_cn,
                self.doc_embeddings_en, self.doc_embeddings_cn,
                self.clusterinfo_en, self.clusterinfo_cn
            ), 
            batch_size=batch_size, 
            shuffle=True
        )
        
        self.test_loader = DataLoader(
            BilingualTextDataset(
                self.test_bow_matrix_en, self.test_bow_matrix_cn,
            ),
            batch_size=batch_size, 
            shuffle=False
        )

    def _merge_split_embeddings(self, data_dir, lang1, lang2):
        """Merge split doc_embeddings files into single file"""
        
        for lang in [lang1, lang2]:
            base_name = f'doc_embeddings_{lang}_train'
            part1_path = os.path.join(data_dir, f'{base_name}_part1.npy')
            part2_path = os.path.join(data_dir, f'{base_name}_part2.npy')
            merged_path = os.path.join(data_dir, f'{base_name}.npy')
            
            # If merged file already exists, skip
            if os.path.exists(merged_path):
                print(f"✓ Merged file already exists: {merged_path}")
                continue
            
            # If both parts exist, proceed with merging
            if os.path.exists(part1_path) and os.path.exists(part2_path):
                print(f"Merging {lang}...")
                part1 = np.load(part1_path)
                part2 = np.load(part2_path)
                
                print(f"  Part1 shape: {part1.shape}, Part2 shape: {part2.shape}")
                
                # Check compatibility
                if part1.ndim != part2.ndim:
                    raise ValueError(f"Part1 and Part2 have different dimensions: {part1.ndim} vs {part2.ndim}")
                
                # Merge along axis 0 (rows/samples)
                merged = np.concatenate([part1, part2], axis=0)
                np.save(merged_path, merged)
                
                # Validation
                expected_total = part1.shape[0] + part2.shape[0]
                if merged.shape[0] != expected_total:
                    raise ValueError(
                        f"Merge incorrect! Expected {expected_total} samples, "
                        f"but got {merged.shape[0]}"
                    )
                
                print(f"✓ Successfully merged: {merged_path}")
                print(f"  Part1: {part1.shape} + Part2: {part2.shape} = Merged: {merged.shape}")
            
            # If merged file does not exist and no parts, error
            elif not os.path.exists(merged_path):
                raise FileNotFoundError(
                    f"Embeddings file not found for {lang}. "
                    f"Need {part1_path} and {part2_path} or {merged_path}"
                )
        
    def move_to_cuda(self, *arrays):
        results = []
        for arr in arrays:
            if isinstance(arr, np.ndarray):
                tensor = torch.tensor(arr).float()
            elif isinstance(arr, torch.Tensor):
                tensor = arr.float()
            else:
                raise TypeError("Input must be a numpy array or torch tensor")
            
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{self.device}")
                tensor = tensor.to(device)
            results.append(tensor)
        
        return results if len(results) > 1 else results[0]

    def read_data(self, data_dir, lang):
        train_texts = file_utils.read_texts(os.path.join(data_dir, f'train_texts_{lang}.txt'))
        test_texts = file_utils.read_texts(os.path.join(data_dir, f'test_texts_{lang}.txt'))
        vocab = file_utils.read_texts(os.path.join(data_dir, f'vocab_{lang}'))

        train_bow_matrix = scipy.sparse.load_npz(
            os.path.join(data_dir, f'train_bow_matrix_{lang}.npz')).toarray()
        test_bow_matrix = scipy.sparse.load_npz(
            os.path.join(data_dir, f'test_bow_matrix_{lang}.npz')).toarray()

        return train_texts, test_texts, train_bow_matrix, test_bow_matrix, vocab

    def calculate_beta(self, bow_np, cluster_np, vocab_size, num_topics, epsilon=1e-8):
        beta = np.zeros((num_topics, vocab_size), dtype=np.float32)

        for doc_idx, cluster_id in enumerate(cluster_np):
            if cluster_id < num_topics:
                beta[cluster_id] += bow_np[doc_idx]

        row_sums = beta.sum(axis=1, keepdims=True)
        beta = beta / np.maximum(row_sums, epsilon)

        doc_freq = np.count_nonzero(beta > 0, axis=0)
        idf = np.log((num_topics + 1) / (doc_freq + 1)) + 1

        beta = beta * idf
        row_sums = beta.sum(axis=1, keepdims=True)
        beta = beta / np.maximum(row_sums, epsilon) + epsilon
        return beta
    
    def calculate_cluster_based_prior(self, num_topics, smoothing_count=0, epsilon=1e-6):
        labels_l1 = self.clusterinfo_en
        labels_l2 = self.clusterinfo_cn
        
        all_labels = np.concatenate((labels_l1, labels_l2))
        
        counts_array = np.zeros(num_topics, dtype=np.float32)
        for label in all_labels:
            if 0 <= label < num_topics: 
                counts_array[int(label)] += 1
        
        smoothed_counts = counts_array + smoothing_count
        total_docs = smoothed_counts.sum()
            
        avg_count = total_docs / num_topics
        a_new = (smoothed_counts / (avg_count + epsilon)) + epsilon
        a_new = np.maximum(a_new, epsilon)
        a_new = a_new.reshape(1, -1)
        
        log_a_new = np.log(a_new)
        mu_prior = (log_a_new.T - np.mean(log_a_new, 1, keepdims=True)).T
        
        term1 = (1.0 / a_new) * (1 - (2.0 / num_topics))
        term2 = (1.0 / (num_topics * num_topics)) * np.sum(1.0 / a_new, 1, keepdims=True)
        var_prior = term1 + term2
        
        mu_prior = mu_prior.squeeze()
        var_prior = var_prior.squeeze()
        return mu_prior.astype(np.float32), var_prior.astype(np.float32)
