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
from typing import Dict, Tuple, List

from examples.ner_new.task.converter_selector import ConverterSelector
from forte.data.ontology.core import Entry

from forte.data.readers.base_reader import BaseReader

from examples.ner_new.converter.converter_container import ConverterContainer
from examples.ner_new.task.model_selector import ModelSelector
from forte.processors.base.base_processor import BaseProcessor
from forte.evaluation.base.base_evaluator import Evaluator
from forte.trainer.base.base_trainer import BaseTrainer
from examples.ner_new.train_pipeline import TrainPipeline


class TaskManager():
    def __init__(self):
        # TODO: how to initialize attributes here instead of using set method?
        self.model_selector = ModelSelector()
        self.converter_selector = ConverterSelector()

    def set_reader(self, reader: BaseReader):
        self.reader = reader

    def build_train_pipeline(self, request: Dict) -> TrainPipeline:
        # Example request format:
        # request = {
        #     "data": {
        #         Token: {
        #             "ner": ["dense_converter"],
        #             "text": ["dense_converter", "char_converter"]
        #         }
        #     },
        #     "context_type": Sentence
        #     "model_type": "CoNNNLNER",
        #     "config": <Config>
        # }

        # TODO: add check

        # Search trainer, predictor, evaluator.
        trainer, evaluator, predictor = self.search_model(request["model_type"])

        # Config trainer predictor, evaluator.
        # TODO

        # Build data pack request.
        datapack_request = self.generate_datapack_request(request)

        # Build converters.
        converter_container_p2t, converter_container_t2p = \
            self.generate_converters(request)

        # Build pipeline.
        train_pipe = TrainPipeline()
        train_pipe.set_reader(self.reader)
        train_pipe.set_config(request["config"])
        train_pipe.set_datapack_request(datapack_request)
        train_pipe.set_trainer(trainer)
        train_pipe.set_evaluator(evaluator)
        train_pipe.set_predictor(predictor)
        train_pipe.set_converter_container_p2t(converter_container_p2t)
        train_pipe.set_converter_container_p2t(converter_container_t2p)

        return train_pipe

    def search_model(self, type: str) \
            -> Tuple[BaseTrainer, Evaluator, BaseProcessor]:
        return self.model_selector.search(type)

    def generate_datapack_request(self, request: Dict) -> Dict:
        # Example output
        # {
        #     "context_type": Sentence,
        #     "request": {
        #         Token: ["ner"],
        #         Sentence: [],  # span by default
        #     }
        # }
        datapack_request = {"context_type" : request["context_type"]}
        datapack_request["request"] = {}
        datapack_request["request"][request["context_type"]] = []

        for entry, labels in request["data"].items():
            # TODO: how to convert string to type?
            # text label is automatically requested by default
            label_list: List = [l for l in labels.keys() if l != "text"]
            datapack_request["request"][entry] = label_list

        return datapack_request

    def generate_converters(self, request: Dict) \
        -> Tuple[ConverterContainer, ConverterContainer]:
        converter_map_p2t, converter_map_tp2 = {}, {}

        for entry, labels in request["data"].items():
            converter_map_p2t[entry] = {}
            converter_map_tp2[entry] = {}
            for label, converter_types in labels.items():
                converter_map_p2t[entry][label] = []
                converter_map_tp2[entry][label] = []
                for converter_type in converter_types:
                    p2t, t2p = self.converter_selector.search(converter_type)

                    converter_map_p2t[entry][label].append(p2t)
                    converter_map_tp2[entry][label].append(t2p)

        converter_container_p2t: ConverterContainer \
            = ConverterContainer(converter_map_p2t)
        converter_container_t2p: ConverterContainer \
            = ConverterContainer(converter_map_tp2)

        return converter_container_p2t, converter_container_t2p