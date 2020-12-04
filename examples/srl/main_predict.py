# Copyright 2020 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import yaml
import torch
from forte.pipeline import Pipeline
from forte.data.readers.ontonotes_reader import OntonotesReader
from forte.predictor import Predictor
from forte.data.extractor.extractor import LinkExtractor
from ft.onto.base_ontology import Sentence, Token, PredicateLink


reader = OntonotesReader()


pl = Pipeline()
pl.set_reader(reader)
pl.initialize()

config ={
        "entry_type": PredicateLink,
        "attribute": "arg_type",
        "based_on": Token,
        "vocab_method": "indexing",
        }

extractor = LinkExtractor(config)

for pack in pl.process_dataset("/Users/jiaqiangruan/CMU/Capstone/forte/data_samples/ontonotes/00/"):
    print("====== pack ======")
    for instance in pack.get(Sentence):
        extractor.extract(pack, instance)
