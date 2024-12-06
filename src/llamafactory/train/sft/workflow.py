# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
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

from typing import TYPE_CHECKING, List, Optional

from transformers import DataCollatorForSeq2Seq

from ...data import get_dataset, split_dataset
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeMetrics
from .trainer import CustomSeq2SeqTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    
    # =====
    from utils_zp import path, load_json
    extra_setting = load_json(path(training_args.output_dir)/'extra_setting.json')
    from copy import deepcopy as dcopy
    dataset = {}
    def get_dataset_split(split):
        cur_data_args = dcopy(data_args)
        if split == 'train':
            cur_data_args.dataset = data_args.dataset
            dataset[split] = get_dataset(model_args, cur_data_args, training_args, stage="sft", **tokenizer_module)
        elif split == 'dev':
            assert data_args.dataset.endswith('train')
            cur_data_args.dataset = data_args.dataset[:-5]+'dev'
            dataset[split] = get_dataset(model_args, cur_data_args, training_args, stage="sft", **tokenizer_module)
        elif split is None:
            dataset['test'] = get_dataset(model_args, cur_data_args, training_args, stage="sft", **tokenizer_module)
        else:
            raise 'wrong split'
        if split == 'train' and data_args.streaming:
            dataset['train'] = dataset['train'].shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)
        
    if training_args.do_train:
        training_args.do_train = True
        training_args.do_eval = False
        training_args.do_predict = False
        get_dataset_split('train')
        get_dataset_split('dev')
    else:
        training_args.do_train = False
        training_args.do_eval = False
        training_args.do_predict = True
        get_dataset_split(None)
    # =====

    # print('='*200)
    # print(dataset['train'])
    # print('='*20)
    # print(dataset['dev'])
    # print('='*20)
    # print(model_args)
    # print('='*20)
    # print(data_args)
    # print('='*20)
    # print(training_args)
    # print('='*20)
    # print(finetuning_args)
    # print('='*20)
    # print(generating_args)
    # print('='*20)
    # print(callbacks)
    # print('='*20)
    # # print(split_dataset(dataset, data_args, training_args))
    # print('='*20)
    
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    
    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False if model_args.visual_inputs else training_args.remove_unused_columns

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
        train_dataset=dataset['train'] if 'train' in dataset else None,
        eval_dataset=dataset['dev'] if 'dev' in dataset else None,
        **tokenizer_module,
        # **split_dataset(dataset, data_args, training_args),
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # =====
    from utils_zp import GPUBalancer, os
    cuda_id = int(os.environ.get('CUDA_VISIBLE_DEVICES'))
    balancer = GPUBalancer(
        cuda_ids=[cuda_id], rest_mem_mb=extra_setting['rest_mem_mb'],
        keep_run=False, wait_before_start=extra_setting['wait_befor_start'],
    )
    # =====
    
    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])


    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset['test'], metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset, predict_results)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)

    # =====
    balancer.close()
    # =====