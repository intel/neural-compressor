# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)

finetuned_model = "distilbert-base-uncased-finetuned-sst-2-english"


class MyDataLoader(object):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(finetuned_model)
        self.sequence = "Shanghai is a beautiful city!"
        self.encoded_input = self.tokenizer(
            self.sequence,
            return_tensors='pt'
        )
        self.label = 1  # negative sentence: 0; positive sentence: 1
        self.batch_size = 1

    def __iter__(self):
        yield self.encoded_input, self.label


my_nlp_model = AutoModelForSequenceClassification.from_pretrained(
    finetuned_model,
)

my_nlp_dataloader = MyDataLoader()

output = my_nlp_model(**my_nlp_dataloader.encoded_input)
