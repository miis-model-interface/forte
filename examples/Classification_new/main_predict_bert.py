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
"""This file predict the ner tag for conll03 dataset."""

import yaml
import torch
from forte.pipeline import Pipeline
from forte.predictor import Predictor
from ft.onto.base_ontology import Sentence
from forte.data.readers.imdb_reader import IMDBReader

def pad_each_bach(word, max_sen_len):
    batch_size = word.shape[0]
    curr_len = word.shape[1]
    word_list = word.tolist()

    # Line 0 in word_embedding_table is padding vec
    for i in range(batch_size):
        for j in range(max_sen_len-curr_len):
            word_list[i].append(0)

    return torch.LongTensor(word_list)


def predict_forward_fn(model, batch):
    '''Use model and batch data to predict label.'''
    word = batch["text_tag"]["data"]
    logits, pred = model(pad_each_bach(word, 500))
    pred = pred.numpy()
    print(pred)
    return {"label_tag": pred}


config_predict = yaml.safe_load(open("config_predict.yml", "r"))
saved_model = torch.load(config_predict['model_path'])
train_state = torch.load(config_predict['train_state_path'])

reader = IMDBReader()
predictor = Predictor(batch_size=config_predict['batch_size'],
                model=saved_model,
                predict_forward_fn=predict_forward_fn,
                feature_resource=train_state['feature_resource'])
#evaluator = CoNLLNEREvaluator()

pl = Pipeline()
pl.set_reader(reader)
pl.add(predictor)
#pl.add(evaluator)
pl.initialize()

for pack in pl.process_dataset(config_predict['test_path']):
    print("---- pack ----")
    for instance in pack.get(Sentence):
        sentence = instance.text
        predicts = []
        for entry in pack.get(Sentence, instance):
            predicts.append(entry.speaker)
        print('---- example -----')
        print("sentence: ", sentence)
        print("predict sentiment: ", predicts)
    #print(evaluator.get_result())
