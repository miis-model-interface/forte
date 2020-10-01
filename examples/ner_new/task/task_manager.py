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
from typing import Dict, Tuple

from examples.ner_new.converter.converter_container import ConverterContainer
from forte.processors.base.base_processor import BaseProcessor
from forte.evaluation.base.base_evaluator import Evaluator
from forte.trainer.base.base_trainer import BaseTrainer
from examples.ner_new.train_pipeline import TrainPipeline


class TaskManager():
    def __init__(self):
        pass

    def build_train_pipeline(self, request: Dict) -> TrainPipeline:
        # TODO: add check

        # Search trainer, predictor, evaluator.
        trainer, evaluator, predictor = self.search_model(request['model_type'])

        # Config trainer predictor, evaluator.
        # TODO

        # Build data pack request.
        datapack_request = self.generate_datapack_request(request)

        # Build converters.
        converter_container_p2t, converter_container_t2p = \
            self.generate_converters(request)

        # Build pipeline.
        train_pipe = TrainPipeline()

        return train_pipe

    def search_model(self, type: str) \
            -> Tuple[BaseTrainer, Evaluator, BaseProcessor]:
        pass

    def generate_datapack_request(self, request: Dict) -> Dict:
        pass

    def generate_converters(self, request: Dict) \
        -> Tuple[ConverterContainer, ConverterContainer]:
        pass