#  Copyright 2020 The Forte Authors. All Rights Reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#       http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from typing import Tuple

from examples.ner_new.ner_evaluator import NerEvaluator
from examples.ner_new.ner_predictor import NerPredictor
from examples.ner_new.ner_trainer import NerTrainer
from forte.processors.base.base_processor import BaseProcessor

from forte.evaluation.base.base_evaluator import Evaluator
from forte.trainer.base.base_trainer import BaseTrainer


class ModelSelector():
    def __init__(self):
        pass

    def search_trainer(self, type: str) -> BaseTrainer:
        pass

    def search_evaluator(self, type: str) -> Evaluator:
        pass

    def search_predictor(self, type: str) -> BaseProcessor:
        pass

    def search(self, type: str) -> Tuple[BaseTrainer, Evaluator, BaseProcessor]:
        if type == "CoNNNLNER":
            ner_trainer = NerTrainer()
            ner_predictor = NerPredictor()
            ner_evaluator = NerEvaluator()
            return ner_trainer, ner_evaluator, ner_predictor
        else:
            raise NotImplementedError

    def register(self, type: str,
                 trainer: BaseTrainer,
                 evaluator: Evaluator,
                 predictor: BaseProcessor):
        pass