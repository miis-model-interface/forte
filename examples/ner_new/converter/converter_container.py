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


class ConverterContainer():
    def __init__(self, converter_request: Dict):
        # converter_map format:
        # {
        #    "Token": {
        #        "ner": [DenseConverter],
        #        "text": [DenseConverter, CharConverter]
        #    }
        # }
        # TODO
        self.converter_map: Dict = {}
        pass

    def consume(self, instance: Dict) -> Dict:
        # Consume should output tensor container, the format is:
        # {
        #    "Token": {
        #        "ner": {"dense": [<tensor>]},
        #        "text": {"dense": [<tensor>], "char": [<tensor>]},
        #    }
        # }
        tensor_container = {}

        for annotation_name in self.converter_map.keys():
            annotation = instance[annotation_name]
            tensor_container[annotation_name] = {}

            for label_name in self.converter_map[annotation_name].keys():
                # Now we focus on a single label from a certain annotation
                tensor_container[annotation_name][label_name] = {}

                label = annotation[label_name]

                for converter in \
                        self.converter_map[annotation_name][label_name]:
                    tensor_container[annotation_name][label_name][
                        converter.name] = converter.convert(label)

        return tensor_container
