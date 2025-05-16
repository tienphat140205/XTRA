from datetime import datetime
import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import wandb


def get_current_datetime():
    # Get the current date and time
    current_datetime = datetime.now()

    # Convert it to a string
    datetime_string = current_datetime.strftime(
        "%Y-%m-%d_%H-%M-%S")  # Format as YYYY-MM-DD HH:MM:SS
    return datetime_string


def create_folder_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Folder created:", folder_path)
    else:
        print("Folder already exists:", folder_path)


def tsne_viz(word_embedding, topic_embedding, save_path, viz_group=False, logwandb=False):
    tsne = TSNE(n_components=2, random_state=0,
                perplexity=5 if viz_group else 30)
    word_c = np.ones(word_embedding.shape[0])
    topic_c = np.zeros(topic_embedding.shape[0])
    wt_c = np.concatenate([word_c, topic_c], axis=0)
    word_and_topic_emb = np.concatenate(
        [word_embedding, topic_embedding], axis=0)
    wt_tsne = tsne.fit_transform(word_and_topic_emb)

    plt.figure(figsize=(10, 5))
    plt.scatter(wt_tsne[:, 0], wt_tsne[:, 1], c=wt_c)
    for i, _ in enumerate(topic_c):
        plt.annotate(
            str(i), (wt_tsne[word_c.shape[0] + i, 0], wt_tsne[word_c.shape[0] + i, 1]))
    plt.title('Word and Topic Embeddings')
    plt.savefig(save_path)
    plt.close()
    if logwandb:
        wandb.log({"Word and Topic Embedding": wandb.Image(save_path)})


def tsne_group_viz(word_embedding, topic_embedding, group_embeddings, save_path_1, save_path_2, viz_group=False):
    tsne = TSNE(n_components=2, random_state=0,
                perplexity=5 if viz_group else 30)
    word_c = np.ones(word_embedding.shape[0])
    topic_c = np.zeros(topic_embedding.shape[0])
    group_c = np.ones(group_embeddings.shape[0]) * 2
    wt_c = np.concatenate([word_c, topic_c, group_c], axis=0)
    word_and_topic_emb = np.concatenate(
        [word_embedding, topic_embedding, group_embeddings], axis=0)
    wt_tsne = tsne.fit_transform(word_and_topic_emb)

    plt.figure(figsize=(10, 5))
    plt.scatter(wt_tsne[:, 0], wt_tsne[:, 1], c=wt_c)
    for i, _ in enumerate(topic_c):
        plt.annotate(
            str(i), (wt_tsne[word_c.shape[0] + i, 0], wt_tsne[word_c.shape[0] + i, 1]))
    plt.title('Word and Topic Embeddings')
    plt.savefig(save_path_1)
    
    tsne = TSNE(n_components=2, random_state=0,
                perplexity=5 if viz_group else 30)
    topic_c = np.zeros(topic_embedding.shape[0])
    group_c = np.ones(group_embeddings.shape[0])
    wt_c = np.concatenate([topic_c, group_c], axis=0)
    topic_and_group_emb = np.concatenate(
        [topic_embedding, group_embeddings], axis=0)
    wt_tsne = tsne.fit_transform(topic_and_group_emb)

    plt.figure(figsize=(10, 5))
    plt.scatter(wt_tsne[:, 0], wt_tsne[:, 1], c=wt_c)
    for i, _ in enumerate(topic_c):
        plt.annotate(
            str(i), (wt_tsne[i, 0], wt_tsne[i, 1]))
    plt.title('Topic and Group Embeddings')
    plt.savefig(save_path_2)


def eval_viz_group(n_groups, n_topics_per_group, topic_embeddings, dir, logger):
    group_distance = np.zeros((n_groups, n_groups))
    # group_disance[i, j] = average distance between topics in group i and topics in group j

    for i in range(n_groups):
        for j in range(n_groups):
            sum_distance = 0.
            for k in range(n_topics_per_group):
                for l in range(n_topics_per_group):
                    sum_distance += np.linalg.norm(
                        topic_embeddings[i * n_topics_per_group + k] - topic_embeddings[j * n_topics_per_group + l])
            if i == j:
                group_distance[i, j] = sum_distance / \
                    (n_topics_per_group*(n_topics_per_group-1))
            else:
                group_distance[i, j] = sum_distance / \
                    (n_topics_per_group*n_topics_per_group)

    logger.info(f"Group distance:")
    for i in range(len(group_distance)):
        logger.info(f"{group_distance[i]}")

    create_folder_if_not_exist(os.path.join(dir, 'pairwise_group_tsne'))

    # pairwise group tsne visualization
    for i in range(n_groups):
        for j in range(n_groups):
            if i == j:
                continue
            else:
                emb_list_i = np.arange(
                    i*n_topics_per_group, (i+1)*n_topics_per_group)
                emb_list_j = np.arange(
                    j*n_topics_per_group, (j+1)*n_topics_per_group)
                tsne_viz(topic_embeddings[emb_list_i], topic_embeddings[emb_list_j],
                         os.path.join(dir, 'pairwise_group_tsne', f'{i}_{j}.png'), viz_group=True)

    print(group_distance[:5, :5])
    np.fill_diagonal(group_distance, np.inf)
    min_index = np.unravel_index(
        np.argmin(group_distance), group_distance.shape)
    print("argmin_group_distance: ", min_index)
    print("min_group_distance: ", group_distance.min(),
          group_distance[min_index])
    logger.info(f"argmin_group_distance: {min_index}")
    logger.info(
        f"min_group_distance: {group_distance.min()} {group_distance[min_index]}")