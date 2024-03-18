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

"""Topic Clusterer implementation for HDBSCAN + UMAP dimensionality reduction.
"""

import functools
import logging
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import hdbscan
import hyperopt
import numpy as np
import pandas as pd
from sklearn import feature_extraction
from tensorflow import keras
import tensorflow.compat.v2 as tf
import tensorflow_text
import tqdm
import umap

from ml_toast import topic_clusterer

_MIN_SAMPLE_SIZE = 100


def evaluate_hdbscan_clusters(
    clusters: hdbscan.HDBSCAN,
    confidence_threshold: float = 0.05) -> Tuple[int, float]:
  """Evaluates the proposed clusters by calculating a performance score.

  This method is used as the loss function during hyperparameter tuning of
  UMAP and HDBSCAN, where the objective is to minimize the loss. The loss is
  calulcated as the number of data points with a relatively low (defaults to
  5%) cluster assignment confidence.

  Args:
    clusters: The HDBSCAN model reference.
    confidence_threshold: Cluster assignment confidence threshold. Defaults to
      5%.

  Returns:
    A tuple containing the number of unique clusters and the loss value.
  """
  unique_label_count = len(np.unique(clusters.labels_))
  label_count = len(clusters.labels_)
  loss = np.count_nonzero(
      clusters.probabilities_ < confidence_threshold) / label_count

  return unique_label_count, loss


def get_salient_terms(
    vectorizer: feature_extraction.text.TfidfVectorizer,
    candidate_cluster_names: np.ndarray,
    cluster_documents: pd.Series,
    max_num_terms: int = 3) -> Tuple[Sequence[str], Sequence[float]]:
  """Retrieves the top n salient terms for the given cluster documents.

  Args:
    vectorizer: TF-IDF Vectorizer instance.
    candidate_cluster_names: Top TF-IDF values for a specific label.
    cluster_documents: The entire input corpus for a specific label.
    max_num_terms: Maximum number of salient terms to return. Defaults to 3.

  Returns:
    A tuple of n salient terms and their saliency score.
  """
  tfidf = vectorizer.transform(cluster_documents)
  tfidf = np.squeeze(np.asarray(np.mean(tfidf, axis=0)))
  indices = np.argsort(-tfidf)[:max_num_terms]
  indices = [i for i in indices if tfidf[i]]
  terms = [candidate_cluster_names[i] for i in indices]
  weights = [tfidf[i] for i in indices]

  return terms, weights


class TopicClustererHdbscan(topic_clusterer.TopicClusterer):
  """Class to perform topic clustering using UMAP + HDBSCAN.

  Attributes:
    data_id: Identifier of the data to be clustered. Useful for logging when
      running the module with different inputs.
    model: Model object loaded from TensorFlow Hub.
    stop_words: List of custom words to use as stop words.
    default_clustering_params: Default hyperparameters for tuning UMAP and
      HDBSCAN.
    hyperopt_params: Hyperopt pyll graphs for tuning UMAP and HDBSCAN
      hyperparameters.
    desired_num_clusters_range: Desired generated number of clusters. Defaults
      to 10 <= num_clusters <= 150.
    cluster_opt_thresholds: Thresholds for optimizing the generated clusters.
      Defaults to - Assign unclustered data points to their predicted clusters
      if the prediction score > 0.4, and - Recluster data points with a
      prediction score > 0.8 in order to make more consistent cluster
      interpretations. No optimization will happen if this parameter is None.
  """

  def __init__(
      self,
      data_id: str,
      model: Any,
      stop_words: Sequence[str],
      default_clustering_params: Mapping[str, int],
      hyperopt_params: Mapping[str, Any],
      desired_num_clusters_range: Tuple[int, int] = (10, 150)
  ) -> None:
    """Initializer."""
    super().__init__()

    self.data_id = data_id
    self.model = model
    self.stop_words = stop_words
    self.default_clustering_params = default_clustering_params
    self.hyperopt_params = hyperopt_params
    self.desired_num_clusters_range = desired_num_clusters_range

    self.cluster_opt_thresholds = {
        k: self.default_clustering_params[k]
        for k in ('threshold_unclustered', 'threshold_recluster')
    } or None

  def modelling_pipeline(self, documents: pd.DataFrame) -> pd.Series:
    """Runs the clustering modelling pipeline with UMAP + HDBSCAN.

    Args:
      documents: Pandas dataframe containing the corpus to cluster.

    Returns:
      A Pandas series containing the cluster assignments for the input
      documents.
    """
    logging.info('%s - Starting UMAP + HDBSCAN clustering...', self.data_id)

    if len(documents) < _MIN_SAMPLE_SIZE:
      logging.info(
          '%s - Skipping UMAP + HDBSCAN clustering due to small input size: %d',
          self.data_id,
          len(documents),
      )
      return pd.Series([])

    embeddings = self.model(documents)

    default_clusters = self.generate_clusters(
        embeddings=embeddings, clustering_params=self.default_clustering_params)
    logging.info('%s - Generated %d clusters using default params: %s.',
                 self.data_id, len(np.unique(default_clusters.labels_)),
                 self.default_clustering_params)
    optimal_clusters = None

    if self.hyperopt_params:
      logging.info(
          '%s - Performing hyperparameter tuning for UMAP / HDBSCAN. '
          'This may take a while...', self.data_id)
      optimal_clustering_params, _ = (
          self.get_optimal_clustering_params(embeddings))
      optimal_clusters = self.generate_clusters(
          embeddings=embeddings, clustering_params=optimal_clustering_params)
      logging.info('%s - Generated %d clusters using optimal params: %s.',
                   self.data_id, len(np.unique(optimal_clusters.labels_)),
                   optimal_clustering_params)

    clusters = optimal_clusters or default_clusters
    cluster_topics = self.recommend_topics_by_saliency(
        documents=documents.iloc[:, 0], cluster_labels=clusters.labels_)

    logging.info('%s - Finished UMAP + HDBSCAN clustering.', self.data_id)
    return pd.Series(clusters.labels_).map(cluster_topics)

  def generate_clusters(self,
                        embeddings: tf.Tensor,
                        clustering_params: Mapping[str, int],
                        optimize_clusters: bool = True) -> hdbscan.HDBSCAN:
    """Generates clusters of input vectors using UMAP + HDBSCAN.

    UMAP helps counteract the 'Curse of Dimensionality' before generating
    clusters with HDBSCAN. Hyperparameters for both modules are adjusted to
    yield optimal results. For UMAP, the `'cosine'` metric is preferred over the
    default `'euclidean'` as the magnitude of the underlying embeddings should
    not affect the projection (rather only the distance), while `min_dist` is
    reduced from the default `0.1` to `0.0` to support the objective of
    highlighting local structure. For HDBSCAN, the `'leaf'` cluster selection
    method is used instead of the default `'eom'` to further support identifying
    smaller homogeneous clusters.

    Args:
      embeddings: Tensor containing document embeddings.
      clustering_params: Hyperparameters for UMAP and HDBSCAN.
      optimize_clusters: Whether to optimize the generated clusters or not.
        Defaults to True.

    Returns:
      The resulting HDBSCAN model reference.
    """
    umap_projection = (
        umap.UMAP(
            n_neighbors=clustering_params['umap_n_neighbors'],
            n_components=clustering_params['umap_n_components'],
            metric='cosine',
            min_dist=0.0,
            random_state=clustering_params['umap_random_state']).fit_transform(
                embeddings))

    clusters = (
        hdbscan.HDBSCAN(
            min_cluster_size=clustering_params['hdbscan_min_cluster_size'],
            min_samples=clustering_params['hdbscan_min_samples'],
            cluster_selection_method='leaf').fit(umap_projection))

    if optimize_clusters and self.cluster_opt_thresholds:
      self.optimize_generated_clusters(embeddings, clusters.labels_)

    return clusters

  def optimize_generated_clusters(self, embeddings: tf.Tensor,
                                  cluster_labels: np.ndarray) -> None:
    """Optimizes the generated HDBSCAN clusters based on the given thresholds.

    Fits a linear regression model from embeddings to their generated HDBSCAN
    non-noise cluster labels (label != -1), applies softmax normalization to
    surface cluster membership probability, and uses backpropagation to minimize
    the categorical cross-entropy between predictions and actual cluster
    assignments. Unclustered input is then assigned to predicted clusters based
    on predefined thresholds in `self.cluster_opt_thresholds`, while embeddings
    with a very high membership probability are reassigned (to their predicted
    cluster instead of their original HDBSCAN cluster) to make interpretations
    of clusters more consistent.

    Args:
      embeddings: Tensor containing document embeddings.
      cluster_labels: HDBSCAN cluster assignments of the given embeddings.
    """
    logging.info('%s - Optimizing HDBSCAN clusters...', self.data_id)

    linear_transformation_model = keras.Sequential([
        keras.layers.Dense(
            units=cluster_labels.max() + 1, activation='softmax')
    ])
    linear_transformation_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'])

    clustered_sample_indices = cluster_labels >= 0
    input_data = np.stack(embeddings[clustered_sample_indices])
    target_data = cluster_labels[clustered_sample_indices]

    linear_transformation_model.fit(
        input_data, target_data, epochs=50, verbose=0)

    batch_size = 1000
    cluster_predictions = []
    prediction_scores = []

    for i in tqdm.trange(0, len(embeddings), batch_size):
      batch_input = np.stack(embeddings[i:i + batch_size])
      predict_batch = linear_transformation_model.predict(
          batch_input, verbose=0)
      clusters_batch = np.argmax(predict_batch, axis=1)
      cluster_predictions.extend(clusters_batch)
      prediction_scores.extend(predict_batch[np.arange(len(clusters_batch)),
                                             clusters_batch])

    prediction_scores = np.asarray(prediction_scores)
    cluster_predictions = np.asarray(cluster_predictions)

    reclustered = ((cluster_labels < 0) &
                   (prediction_scores >
                    self.cluster_opt_thresholds['threshold_unclustered'])) | (
                        prediction_scores >
                        self.cluster_opt_thresholds['threshold_recluster'])

    unassigned_count = len(cluster_labels[cluster_labels == -1])
    cluster_labels[reclustered] = cluster_predictions[reclustered]
    logging.info('%s - Reduced number of unclustered points from %d to %d',
                 self.data_id, unassigned_count,
                 len(cluster_labels[cluster_labels == -1]))

  def optimization_objective(
      self, embeddings: tf.Tensor, label_count_min_threshold: int,
      label_count_max_threshold: int,
      params: Mapping[str, int]) -> Mapping[str, Union[int, float, str]]:
    """Represents the optimization objective for hyperparameter tuning.

    An arbitrary 10-20% penalty is added onto the calculated loss if the number
    of generated clusters falls outside the desired cluster number range.

    Args:
      embeddings: Tensor containing document embeddings.
      label_count_min_threshold: Minimum accepted number of clusters.
      label_count_max_threshold: Maximum accepted number of clusters.
      params: Dict containing hyperparameters for UMAP and HDBSCAN.

    Returns:
      A dict with the calculated loss (including penalty), unique label count
      and hyperopt 'OK' status for this hyperparameter tuning trial.
    """
    clusters = self.generate_clusters(
        embeddings=embeddings,
        clustering_params=params,
        optimize_clusters=False)

    label_count, cost = evaluate_hdbscan_clusters(clusters)

    penalty = 0
    if label_count < label_count_min_threshold:
      penalty += 0.1
    if label_count > label_count_max_threshold:
      penalty += 0.1

    return {
        'loss': cost + penalty,
        'label_count': label_count,
        'status': hyperopt.STATUS_OK,
    }

  def get_optimal_clustering_params(
      self,
      embeddings: tf.Tensor,
      max_evaluations: int = 50) -> Tuple[Mapping[str, int], hyperopt.Trials]:
    """Performs bayesian search to find the optimal hyperparameters to use.

    Args:
      embeddings: Tensor containing document embeddings.
      max_evaluations: Maximum number of evaluations to make during
        hyperparameter tuning. Defaults to 50.

    Returns:
      A tuple of the optimal hyperparameters and hyperopt's `Trials` object.
    """
    trials = hyperopt.Trials()
    fmin_objective = functools.partial(self.optimization_objective, embeddings,
                                       self.desired_num_clusters_range[0],
                                       self.desired_num_clusters_range[1])

    best = hyperopt.fmin(
        fmin_objective,
        space=self.hyperopt_params,
        algo=hyperopt.tpe.suggest,
        max_evals=max_evaluations,
        trials=trials)

    return hyperopt.space_eval(self.hyperopt_params, best), trials

  def recommend_topics_by_saliency(
      self,
      documents: pd.Series,
      cluster_labels: np.ndarray,
      vectorizer: Optional[feature_extraction.text.TfidfVectorizer] = None
  ) -> Mapping[str, Sequence[str]]:
    """Recommends a list of topics for the input docs based on TF-IDF value.

    The `max_df` parameter is reduced from the default `1.0` to `0.8` to
    automatically detect and filter stop words based on intra-corpus document
    frequency of terms, while the `ngram_range` parameter is changed to `(1, 2)`
    to capture both unigrams and bigrams.

    Args:
      documents: Pandas series with documents to recommend topics for.
      cluster_labels: Cluster assignments of the given documents.
      vectorizer: TF-IDF vectorizer to use. Defaults to None which results in
        initializing a new instance with the `max_df`, `ngram_range` and
        `stop_words params as specified in the description.

    Returns:
      A dict of input cluster label identifier to list of recommended topics.
    """
    vectorizer = vectorizer or feature_extraction.text.TfidfVectorizer(
        max_df=0.8, ngram_range=(1, 2), stop_words=self.stop_words)
    vectorizer.fit(documents)
    candidate_cluster_names = vectorizer.get_feature_names_out()

    recommended_topics = {}
    for cluster_label in np.unique(cluster_labels):
      if cluster_label < 0:
        continue
      cluster_documents = documents[cluster_labels == cluster_label]
      if cluster_documents.empty:
        continue
      terms, _ = get_salient_terms(vectorizer, candidate_cluster_names,
                                   cluster_documents)
      prune = []
      for term in terms:
        if ' ' in term:
          for t in term.split(' '):
            if t in terms:
              prune.append(t)
      terms = sorted(list(set(terms) - set(prune)))
      logging.debug('%s - %s: %s (%d)', self.data_id, cluster_label, terms,
                    cluster_documents.shape[0])

      recommended_topics[cluster_label] = terms

    return recommended_topics
