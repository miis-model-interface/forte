#  Copyright 2020 The Forte Authors. All Rights Reserved.
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
import logging
from typing import Iterator, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from texar.torch.data import Batch
from tqdm import tqdm
import yaml

from examples.ner_new.ner_evaluator import CoNLLNEREvaluator
from forte.models.ner.model_factory import BiRecurrentConvCRF
from forte.pipeline import Pipeline
from forte.predictor import Predictor
from forte.data.types import DATA_INPUT, DATA_OUTPUT
from forte.common.configuration import Config
from forte.data.extractor.attribute_extractor \
    import AttributeExtractor
from forte.data.extractor.base_extractor \
    import BaseExtractor
from forte.train_preprocessor import TrainPreprocessor
from forte.data.readers.imdb_reader import IMDBReader
from ft.onto.base_ontology import Sentence, Token

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("cpu")


def construct_word_embedding_table(embed_dict, extractor: BaseExtractor):
    embedding_dim = list(embed_dict.values())[0].shape[-1]

    scale = np.sqrt(3.0 / embedding_dim)
    table = np.empty(
        [extractor.size(), embedding_dim], dtype=np.float32
    )
    oov = 0
    for word, index in extractor.items():
        if word in embed_dict:
            embedding = embed_dict[word]
        elif word.lower() in embed_dict:
            embedding = embed_dict[word.lower()]
        else:
            embedding = np.random.uniform(
                -scale, scale, [1, embedding_dim]
            ).astype(np.float32)
            oov += 1
        table[index, :] = embedding
    return torch.from_numpy(table)



class MyConv(nn.Module):
    def __init__(self, embmatrix):
        super(MyConv, self).__init__()
        self.embedding = nn.Embedding(*embmatrix.size())
        self.embedding.weight.data.copy_(embmatrix)
        self.embedding.weight.requires_grad = True

        self.conv1 = nn.Conv1d(300, 100, 3)
        self.conv2 = nn.Conv1d(300, 100, 4)
        self.conv3 = nn.Conv1d(300, 100, 5)
        self.pool1 = nn.MaxPool1d(58)
        self.pool2 = nn.MaxPool1d(57)
        self.pool3 = nn.MaxPool1d(56)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(300, 16)

    def forward(self, x):  # for a sentence (300,60)
        # x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x1 = self.pool1(F.relu(self.conv1(x)))
        x2 = self.pool2(F.relu(self.conv2(x)))
        x3 = self.pool3(F.relu(self.conv3(x)))
        x = torch.cat((x1, x2, x3), dim=1).squeeze()
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        return x

def create_model(schemes: Dict[str, Dict[str, BaseExtractor]],
                 config: Config):
    text_extractor: BaseExtractor = schemes["text_tag"]["extractor"]
    #attribute_extractor: BaseExtractor = schemes["label_tag"]["extractor"]

    # embedding_dict = \
    #     load_glove_embedding(config.config_preprocessor.embedding_path)
    #
    # for word in embedding_dict:
    #     if not text_extractor.has_key(word):
    #         text_extractor.add(word)

    # TODO: temporarily make fake pretrained emb for debugging
    embedding_dict = {}
    fake_tensor = torch.tensor([0.0 for i in range(100)])
    for word, index in text_extractor.items():
        embedding_dict[word] = fake_tensor

    word_embedding_table = \
        construct_word_embedding_table(embedding_dict, text_extractor)

    model: nn.Module = \
        MyConv(word_embedding_table)

    return model


def train(model: nn.Module, optim: Optimizer, batch: Batch):
    word = batch["text_tag"]["tensor"]
    labels = batch["label_tag"]["tensor"]
    print("word in train func: ", word)
    print("labels in train func", labels)
    optim.zero_grad()
    word = word.to(device)
    labels = labels.to(device)

    outputs = model(word)

    loss = criterion(outputs, labels)

    loss.backward()
    optim.step()

    batch_train_err = loss.item() * batch.batch_size

    return batch_train_err


# def predict_forward_fn(model: nn.Module, batch: Dict) -> Dict:
#     word = batch["text_tag"]["tensor"]
#     char = batch["char_tag"]["tensor"]
#     word_masks = batch["text_tag"]["mask"][0]
#
#     output = model.decode(input_word=word,
#                           input_char=char,
#                           mask=word_masks)
#     output = output.numpy()
#     return {'ner_tag': output}
#

# All the configs
config_data = yaml.safe_load(open("config_data.yml", "r"))
config_model = yaml.safe_load(open("config_model.yml", "r"))
config_preprocess = \
    yaml.safe_load(open("config_preprocessor.yml", "r"))

config = Config({}, default_hparams=None)
config.add_hparam('config_data', config_data)
config.add_hparam('config_model', config_model)
config.add_hparam('preprocessor', config_preprocess)


tp_request = {
    "scope": Sentence,
    "schemes": {
        "text_tag": {
            "entry_type": Token,
            "get_attribute_fn": lambda x: x.text,
            "vocab_method": "indexing",
            "type": DATA_INPUT,
            "extractor": AttributeExtractor
        },
        "label_tag": {
            "entry_type": Sentence,
            "get_attribute_fn": lambda x: x.sentiment[x.text],
            "vocab_method": "indexing",
            "type": DATA_OUTPUT,
            "extractor": AttributeExtractor
        }
    }
}

# All not specified dataset parameters are set by default in Texar.
# Default settings can be found here:
# https://texar-pytorch.readthedocs.io/en/latest/code/data.html#texar.torch.data.DatasetBase.default_hparams
tp_config = {
    "preprocess": {
        "pack_dir": config.config_data.train_path,
        "device": device.type
    },
    "dataset": {
        "batch_size": config.config_data.batch_size_tokens
    }
}

imdb_train_reader = IMDBReader(cache_in_memory=True)
#imdb_val_reader = IMDBReader(cache_in_memory=True)

train_preprocessor = TrainPreprocessor(train_reader=imdb_train_reader,
                                       request=tp_request,
                                       config=tp_config)

model: nn.Module = \
    create_model(schemes=train_preprocessor.feature_resource["schemes"],
                 config=config)
model.to(device)

criterion = nn.CrossEntropyLoss()

optim: Optimizer = SGD(model.parameters(),
                       lr=config.config_model.learning_rate,
                       momentum=config.config_model.momentum,
                       nesterov=True)

# predictor = Predictor(batch_size=train_preprocessor.config.dataset.batch_size,
#                       model=model,
#                       predict_forward_fn=predict_forward_fn,
#                       feature_resource=train_preprocessor.feature_resource,
#                       cross_pack=False)
#evaluator = CoNLLNEREvaluator()

# val_pl = Pipeline()
# val_pl.set_reader(ner_val_reader)
#val_pl.add(predictor)
#val_pl.add(evaluator)

epoch = 0
train_err: float = 0.0
train_total: float = 0.0
train_sentence_len_sum: int = 0


logger.info("Start training.")
print("Total epochs: ", config.config_data.num_epochs)
while epoch < config.config_data.num_epochs:
    epoch += 1

    # TODO: For training, we need to do shuffle batch across all data packs.
    #       This is not naturally supported by Forte pipeline which assumes
    #       processing one data pack at a time.
    #       Any way to make use of Forte pipeline for training as well?
    # Get iterator of preprocessed batch of train data
    train_batch_iter: Iterator[Batch] = \
        train_preprocessor.get_train_batch_iterator()

    batchcount = 1 # Need remove later
    for batch in tqdm(train_batch_iter):
        print("batchcount", batchcount)
        batch_train_err = train(model, optim, batch)

        train_err += batch_train_err
        train_total += batch.batch_size

        batchcount += 1 # Need remove later

    batchcount = 0 # Need remove later

    logger.info(f"{epoch}th Epoch training, "
                f"total number of examples: {train_total}, "
                f"loss: {(train_err / train_total):0.3f}")

    # train_err: float = 0.0
    # train_total: float = 0.0
    # train_sentence_len_sum: int = 0
    #
    # val_pl.run(config.config_data.val_path)
    #
    # logger.info(f"{epoch}th Epoch evaluating, "
    #             f"val result: {evaluator.get_result()}")

# Save training result to disk
# train_pipeline.save_state(config.config_data.train_state_path)
# torch.save(ner_model_processor, config.config_model.model_path)