# Copyright 2018 The Google AI Language Team Authors and DMLC.
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
"""BERT embedding datasets."""
from mxnet.gluon.data import Dataset

__all__ = ['BertEmbeddingDataset']

class BertEmbeddingDataset(Dataset):
    """Dataset for BERT Embedding

    Parameters
    ----------
    sentences : List[str].
        Sentences for embeddings.
    transform : BERTDatasetTransform, default None.
        transformer for BERT input format
    """

    def __init__(self, sentences, transform=None):
        """Dataset for BERT Embedding

        Parameters
        ----------
        sentences : List[str].
            Sentences for embeddings.
        transform : BERTDatasetTransform, default None.
            transformer for BERT input format
        """
        self.sentences = sentences
        self.transform = transform

    def __getitem__(self, idx):
        sentence = (self.sentences[idx], 0)
        if self.transform:
            return self.transform(sentence)
        else:
            return sentence

    def __len__(self):
        return len(self.sentences)
