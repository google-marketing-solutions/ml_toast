# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Topic Clusterer implementation for K-Means on Cosine Distance."""

import logging
from typing import Any, Optional, Sequence, Tuple


import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow.compat.v2 as tf
import tensorflow_text
import tqdm

from ml_toast import topic_clusterer


def get_highest_cosine_similarity_indices(a: tf.Tensor,
                                          b: tf.Tensor,
                                          limit_cosine_similarity: int = 0
                                         ) -> np.ndarray:
  """Returns the elements in b with the highest cosine similarity to a.

  `limit_cosine_similarity` sets a lower bound limit on the cosine similarity
  for an element to be returned (and returns -1 for these values).

  Args:
    a: Tensor of vectors.
    b: Tensor of vectors.
    limit_cosine_similarity: Integer between 0 and 1.

  Returns:
    The elements from b with the highest cosine similarity to elements in a.
  """
  similarity = tf.reduce_sum(a[:, tf.newaxis] * b, axis=-1)

  similarity = tf.math.divide(
      similarity,
      tf.norm(a[:, tf.newaxis], axis=-1) * tf.norm(b, axis=-1))

  indices = tf.math.argmax(similarity, axis=1).numpy()
  if limit_cosine_similarity > 0:
    max_cosine_similarity = tf.math.reduce_max(similarity, axis=1).numpy()
    indices[max_cosine_similarity < limit_cosine_similarity] = -1

  return indices


class TopicClustererKmeans(topic_clusterer.TopicClusterer):
  """Class to perform topic clustering using K-Means.

  Attributes:
    data_id: Identifier of the data to be clustered. Useful for logging when
      running the module with different inputs.
    model: Model object loaded from TensorFlow Hub.
    stop_words: List of custom words to use as stop words.
    clusters: List of clusters to use to identify the optimal value of K.
    optimal_k: The optimal value of K from the given `clusters`.
  """

  def __init__(self, data_id: str, model: Any, stop_words: Sequence[str],
               clusters: Sequence[int]) -> None:
    """Initializer."""
    super().__init__()

    self.data_id = data_id
    self.model = model
    self.stop_words = stop_words
    self.clusters = clusters

    self.optimal_k = None

  def modelling_pipeline(self, documents: pd.DataFrame) -> pd.Series:
    """Runs the clustering modelling pipeline with K-Means.

    Args:
      documents: Pandas dataframe containing the corpus to cluster.

    Returns:
      A Pandas series containing the cluster assignments for the input
      documents.
    """
    logging.info('%s - Starting K-Means clustering...', self.data_id)

    embeddings = self.model(documents)

    if len(self.clusters) == 1:
      self.optimal_k = self.clusters[0]
    else:
      num_clusters = [k for k in self.clusters if len(documents) > k]
      silhouette_scores = [
          self.calculate_silhouette_score(embeddings, num_clusters=k)
          for k in num_clusters
      ]
      silhouette_scores = dict(zip(num_clusters, silhouette_scores))
      self.optimal_k = max(silhouette_scores, key=silhouette_scores.get)

      logging.info(
          '%s - Optimal number of clusters is %d with silhoutte score %f.',
          self.data_id, self.optimal_k, silhouette_scores[self.optimal_k])

    cluster_indices, cluster_centers = self.generate_clusters(
        embeddings, self.optimal_k)
    candidate_cluster_names = self.recommend_topics_by_repetition(
        documents=documents.iloc[:, 0])
    indices = get_highest_cosine_similarity_indices(
        a=cluster_centers, b=self.model(candidate_cluster_names))
    cluster_names = dict(
        zip(
            np.arange(len(cluster_centers)),
            [candidate_cluster_names[i] for i in list(indices)]))

    logging.info('%s - Finished K-Means clustering.', self.data_id)
    return pd.Series(list(cluster_indices)).map(cluster_names)

  def calculate_silhouette_score(self, embeddings: tf.Tensor,
                                 num_clusters: int) -> float:
    """Calculates the silhouette score of the clustering model.

    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (a) and the mean nearest-cluster distance (b) for each sample.
    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters. Negative values generally indicate that a sample has
    been assigned to the wrong cluster, as a different cluster is more similar.

    Args:
      embeddings: Tensor containing document embeddings.
      num_clusters: The value of K (number of clusters).

    Returns:
      The silhouette score.
    """
    cluster_indices, _ = self.generate_clusters(
        embeddings, num_clusters, predict=False)

    score = metrics.silhouette_score(
        X=embeddings.numpy(), labels=list(cluster_indices))

    logging.debug('%s - %d clusters yields %f silhouette score', self.data_id,
                  num_clusters, score)
    return score

  def generate_clusters(
      self,
      embeddings: tf.Tensor,
      num_clusters: int,
      predict: bool = True,
      max_train_iterations: int = 10,
      random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generates clusters of input vectors using K-Means on cosine distance.

    Trains K-Means for `max_train_iterations` and stops training once the score
    improves by less than an arbitrary value of 5%.
    Note: Due to the use of cosine distance, there is no need to apply
    dimensionality reduction before predicting.

    Args:
      embeddings: Tensor containing document embeddings.
      num_clusters: The number of clusters to generate.
      predict: Whether to train only or also predict. Defaults to the latter.
      max_train_iterations: Maximum number of iterations for training K-Means.
        Defaults to 10.
      random_seed: Optional seed for K-Means. Defaults to None.

    Returns:
      A list of cluster assignments for every input document.
    """
    kmeans = tf.compat.v1.estimator.experimental.KMeans(
        num_clusters=num_clusters,
        use_mini_batch=False,
        seed=random_seed,
        distance_metric=(
            tf.compat.v1.estimator.experimental.KMeans.COSINE_DISTANCE))

    def input_fn():
      return tf.compat.v1.train.limit_epochs(
          # First convert to numpy due to v1 & eager incompatibility
          tf.convert_to_tensor(embeddings.numpy(), dtype=tf.float32),
          num_epochs=1)

    previous_centers = None
    score = 0

    pbar = tqdm.tqdm(
        desc=''.join([
            f'{self.data_id} - ',
            'Predicting' if predict else 'Training',
            f' with K={num_clusters}',
        ]),
        total=max_train_iterations)
    for i in range(max_train_iterations):
      kmeans.train(input_fn)
      cluster_centers = kmeans.cluster_centers()

      if previous_centers is not None:
        previous_centers = cluster_centers
      new_score = kmeans.score(input_fn)
      logging.debug('%s - K=%d - Iteration %d: Sum of squared distances: %.0f',
                    self.data_id, num_clusters, i, new_score)
      if np.divide(score, new_score) > 1.05 or score == 0:
        score = new_score
        pbar.update(1)
      else:
        pbar.update(max_train_iterations - i)
        break
    pbar.close()

    return kmeans.predict_cluster_index(input_fn), cluster_centers

  def recommend_topics_by_repetition(
      self,
      documents: pd.Series,
      max_recommendations: int = 150) -> Sequence[str]:
    """Recommends a list of topics for a given set of docs based on repetition.

    Args:
      documents: Pandas series with documents to recommend topics for.
      max_recommendations: Maximum size of the recommendations to return.

    Returns:
      The recommended list of topics.
    """
    candidate_clusters = pd.Series(' '.join(documents).split()).value_counts()

    if self.stop_words:
      word_filter = True
      for word in self.stop_words:
        word_filter &= ~candidate_clusters.index.str.contains(word)
      candidate_clusters = candidate_clusters[word_filter]

    return candidate_clusters[0:max_recommendations].index.to_list()
