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
import yaml

from examples.ner_new.task.task_manager import TaskManager
from forte.data.readers.conll03_reader import CoNLL03Reader

from forte.common.configuration import Config
from ft.onto.base_ontology import Token, Sentence


def main():
    # Define config.
    config_data = yaml.safe_load(open("config_data.yml", "r"))
    config_model = yaml.safe_load(open("config_model.yml", "r"))
    config_preprocess = yaml.safe_load(open("config_preprocessor.yml", "r"))

    config = Config({}, default_hparams=None)
    config.add_hparam('config_data', config_data)
    config.add_hparam('config_model', config_model)
    config.add_hparam('preprocessor', config_preprocess)
    config.add_hparam('reader', {})
    config.add_hparam('evaluator', {})

    # Define reader.
    reader = CoNLL03Reader()

    # Define request
    request = {
        "data": {
            Token: {
                "ner": ["dense_converter"],
                "text": ["dense_converter", 'char_converter']
            }
        },
        "context_type": Sentence,
        "model_type": "CoNNNLNER",
        "config": Config
    }

    task_manager = TaskManager()
    task_manager.set_reader(reader)

    train_pipe = task_manager.build_train_pipeline(request)

    train_pipe.run()


if __name__ == '__main__':
    main()