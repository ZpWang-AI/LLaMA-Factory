# Copyright 2024 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Sequence, Tuple, Union

import numpy as np
from transformers.utils import is_jieba_available, is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_rouge_available


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


if is_rouge_available():
    from rouge_chinese import Rouge


@dataclass
class ComputeMetrics:
    r"""
    Wraps the tokenizer into metric functions, used in Seq2SeqPeftTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def __call__(self, eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
        return {}
        
        from sklearn.metrics import classification_report
        import json
        r"""
        Uses the model predictions to compute metrics.
        """
        preds, gts = eval_preds
        
        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        gts = np.where(gts != IGNORE_INDEX, gts, self.tokenizer.pad_token_id)

        preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        gts = self.tokenizer.batch_decode(gts, skip_special_tokens=True)
        gts = [p.split(',')for p in gts]
        
        from itertools import chain
        label_list = sorted(set(chain(*gts)))
        if label_list[0] == '':
            del label_list[0]
        n, m = len(preds), len(label_list)
        def label_to_id(label):
            return label_list.index(label) if label in label_list else m
        
        y_pred = np.zeros((n,m+1), dtype=int)
        y_true = np.zeros((n,m+1), dtype=int)
        for p, c in enumerate(preds):
            y_pred[p, label_to_id(c)] = 1
        for p, cs in enumerate(gts):
            # for c in cs:
            #     y_true[p, label_to_id(c)] = 1
            y_true[p, label_to_id(cs[0])] = 1

        res_dic = classification_report(
            y_true=y_true, y_pred=y_pred, 
            labels=list(range(len(label_list))), 
            target_names=label_list, zero_division=0,
            output_dict=True,
        )
        return {
            'macro-f1': res_dic['macro avg']['f1-score'],
            'res_dic': json.dumps(res_dic, ensure_ascii=False, indent=4)
        }
    