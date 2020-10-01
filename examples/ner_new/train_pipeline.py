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
from typing import Dict

from examples.ner_new.converter.converter_container import ConverterContainer
from forte.common.configuration import Config
from forte.data.readers.base_reader import BaseReader
from forte.evaluation.base.base_evaluator import Evaluator
from forte.processors.base.base_processor import BaseProcessor
from forte.trainer.base.base_trainer import BaseTrainer


class TrainPipeline():
    def __init__(self):
        # TODO: define attributes here
        pass

    def set_trainer(self, trainer: BaseTrainer):
        self.trainer = trainer

    def set_evaluator(self, evaluator: Evaluator):
        self.evaluator = evaluator

    def set_predictor(self, predictor: BaseProcessor):
        self.predictor = predictor

    def set_reader(self, reader: BaseReader):
        self.reader = reader

    def set_config(self, config: Config):
        self.config = config

    def set_datapack_request(self, request: Dict):
        self.datapack_request = request

    def set_preprocessor(self, preprocessor: BaseProcessor):
        # TODO: do we need this here?
        self.preprocessor = preprocessor

    def set_converter_container_p2t(self, converter_container: ConverterContainer):
        self.converter_container_p2t = converter_container

    def set_converter_container_t2p(self, converter_container: ConverterContainer):
        self.converter_container_t2p = converter_container

    def run(self):
        # TODO: run
        # TODO: prepare
        pass