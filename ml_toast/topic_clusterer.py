# Copyright 2023 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Topic Clusterer Interface."""

import pandas as pd
import typing_extensions


class TopicClusterer(typing_extensions.Protocol):
  """Interface for topic clustering using a specific algorithm."""

  def modelling_pipeline(self, documents: pd.DataFrame) -> pd.Series:
    """Runs the clustering modelling pipeline.

    Args:
      documents: Pandas dataframe containing the corpus to cluster.

    Returns:
      Pandas series containing the cluster assignments for the input documents.
    """
    raise NotImplementedError()  # pragma nocover
