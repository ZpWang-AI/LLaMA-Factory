# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_equal_to_4_46, is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: "PreTrainedTokenizer" = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: Dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", List["torch.Tensor"]]]:
        r"""
        Fixes the loss value for transformers 4.46.0.
        https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/trainer.py#L3605
        """
        loss = super().compute_loss(model, inputs, return_outputs, **kwargs)
        if is_transformers_version_equal_to_4_46() and not getattr(self, "model_accepts_loss_kwargs", False):
            # other model should not scale the loss
            if return_outputs:
                return (loss[0] / self.args.gradient_accumulation_steps, *loss[1:])
            else:
                return loss / self.args.gradient_accumulation_steps

        return loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"] if "labels" in inputs else None
        if self.args.predict_with_generate:
            assert self.processing_class.padding_side == "left", "This method only accepts left-padded tensor."
            labels = labels.detach().clone() if labels is not None else None  # backup labels
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        # loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
        loss, generated_tokens, _ = self.prediction_step_with_score_custom(  # ignore the returned labels (may be truncated)
            model, inputs, 
            prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys,
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    # =====
    def prediction_step_with_score_custom(
        self,
        model: 'torch.nn.Module',
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
        from transformers.integrations.fsdp import is_fsdp_managed_module
        default_synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self.model)
        gen_kwargs["synced_gpus"] = gen_kwargs.get("synced_gpus", default_synced_gpus)

        generation_inputs = inputs.copy()
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in generation_inputs
            and "decoder_input_ids" in generation_inputs
            and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {
                k: v for k, v in inputs.items() if k not in ("decoder_input_ids", "decoder_attention_mask")
            }

        import contextlib
        from torch.distributed.fsdp import FullyShardedDataParallel
        summon_full_params_context = (
            FullyShardedDataParallel.summon_full_params(self.model)
            if isinstance(self.model, FullyShardedDataParallel)
            else contextlib.nullcontext()
        )

        # =====
        # -- old ---
        # with summon_full_params_context:
        #     generated_tokens = self.model.generate(**inputs, **custome_gen_kwargs)
        # -- old ---

        from utils_zp import path, auto_load, auto_dump, tensor_to_list, dcopy

        custome_gen_kwargs = dcopy(gen_kwargs)
        custome_gen_kwargs['return_dict_in_generate'] = True
        custome_gen_kwargs['output_scores'] = True
        with summon_full_params_context:
            generated_tokens = self.model.generate(**inputs, **custome_gen_kwargs)
        sequences, scores, past_key_values = (
            generated_tokens['sequences'],
            generated_tokens['scores'],
            generated_tokens['past_key_values'],
        )

        src_output_dir = path(self.args.output_dir)
        root_output_dir = src_output_dir.parent
        extra_setting = auto_load(root_output_dir / 'main_config.json')['extra_setting']
        if extra_setting['output_scores']:
            # print(self.processing_class.batch_decode(sequences[:, :prompt_len]))
            # print(self.processing_class.batch_decode(sequences[:, prompt_len:]))
            # seq_len = sequences.shape[1]
            # print(seq_len, prompt_len, seq_len-prompt_len)
            # print(len(scores))
            prompt_len = inputs["input_ids"].size(-1)
            output_str = self.processing_class.batch_decode(
                sequences[:, prompt_len:], 
                skip_special_tokens=True,
            )
            output_seq = sequences[0, prompt_len:]
            output_scores = torch.concat(scores)
            # print(output_scores.shape)
            output_scores = output_scores[range(output_scores.shape[0]), output_seq]
            # output_scores = [
            #     scores[pid][0, output_seq[pid]].item()
            #     for pid in range(output_seq.shape[0])
            # ]
            label_str = self.processing_class.batch_decode(inputs['labels'], skip_special_tokens=True)

            output_seq = tensor_to_list(output_seq)
            output_scores = tensor_to_list(output_scores)
            auto_dump(
                {
                    'output_str': output_str,
                    'output_seq': output_seq,
                    'output_scores': output_scores,
                    'label_str': label_str,
                },
                # path('/public/home/hongy/zpwang/LLaMA-Factory_zp/src/scripts/sga100', 'c.json')
                src_output_dir / 'generated_scores.jsonl',
            )
            # _scores = []
            # prompt_len = inputs["input_ids"].size(-1)
            # for pid, score in enumerate(scores):
            #     scores.append(score[0,seq[pid+prompt_len]])
            # try:
            #     scores = torch.tensor(scores).detach().cpu().tolist()
            # except:
            #     print(scores)
            #     exit()
            
            # real_label = selfprocessing_class.batch_decode(real_labels)[0]
            # real_label = real_label.replace('<|eot_id|>', '')
            # score_info = {'label': real_label, 'scores': scores}
            # auto_dump(score_info, score_output_path)
        
        generated_tokens = generated_tokens['sequences']
        # exit()
        # =====

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, generated_tokens, labels

    # def prediction_step_with_score(
    #     self,
    #     model: "torch.nn.Module",
    #     inputs: Dict[str, Union[torch.Tensor, Any]],
    #     prediction_loss_only: bool,
    #     ignore_keys: Optional[List[str]] = None,
    #     **gen_kwargs,
    # ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
    #     """
    #     Perform an evaluation step on `model` using `inputs`.

    #     Subclass and override to inject custom behavior.

    #     Args:
    #         model (`nn.Module`):
    #             The model to evaluate.
    #         inputs (`Dict[str, Union[torch.Tensor, Any]]`):
    #             The inputs and targets of the model.

    #             The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
    #             argument `labels`. Check your model's documentation for all accepted arguments.
    #         prediction_loss_only (`bool`):
    #             Whether or not to return the loss only.
    #         gen_kwargs:
    #             Additional `generate` specific kwargs.

    #     Return:
    #         Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
    #         labels (each being optional).
    #     """

    #     if not self.args.predict_with_generate or prediction_loss_only:
    #         return super().prediction_step(
    #             model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
    #         )

    #     has_labels = "labels" in inputs
    #     inputs = self._prepare_inputs(inputs)

    #     # XXX: adapt synced_gpus for fairscale as well
    #     # Priority (handled in generate):
    #     # gen_kwargs > model.generation_config > default GenerationConfig()

    #     if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
    #         gen_kwargs = self._gen_kwargs.copy()

    #     if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
    #         gen_kwargs["max_length"] = self.model.config.max_length
    #     gen_kwargs["num_beams"] = (
    #         gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
    #     )
    #     default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
    #     gen_kwargs["synced_gpus"] = (
    #         gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
    #     )

    #     # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
    #     # (otherwise, it would continue generating from the padded `decoder_input_ids`)
    #     if (
    #         "labels" in inputs
    #         and "decoder_input_ids" in inputs
    #         and inputs["labels"].shape == inputs["decoder_input_ids"].shape
    #     ):
    #         inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}

    #     # =====
    #     # from utils_zp import *
    #     # import utils_zp
    #     from utils_zp import auto_load, auto_dump, path
    #     src_output_dir = path(self.args.output_dir)
    #     root_output_dir = src_output_dir.parent
    #     extra_setting = auto_load(root_output_dir / 'main_config.json')['extra_setting']
    #     if extra_setting['output_scores']:
    #         score_output_path = src_output_dir / 'generated_scores.jsonl'
            
    #         gen_kwargs['return_dict_in_generate'] = True
    #         gen_kwargs['output_scores'] = True
    #         generated_tokens = self.model.generate(**inputs, **gen_kwargs)
    #         seq = generated_tokens['sequences']
    #         seq = seq[0]
    #         scores = []
    #         prompt_len = inputs["input_ids"].size(-1)
    #         for pid, score in enumerate(generated_tokens['scores']):
    #             scores.append(score[0,seq[pid+prompt_len]])
    #         try:
    #             scores = torch.tensor(scores).detach().cpu().tolist()
    #         except:
    #             print(scores)
    #             exit()
            
    #         real_label = selfprocessing_class.batch_decode(real_labels)[0]
    #         real_label = real_label.replace('<|eot_id|>', '')
    #         score_info = {'label': real_label, 'scores': scores}
    #         auto_dump(score_info, score_output_path)
            
    #         gen_kwargs['return_dict_in_generate'] = False
    #         gen_kwargs['output_scores'] = False
    #     # =====
        
    #     generated_tokens = self.model.generate(**inputs, **gen_kwargs)

    #     # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
    #     # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
    #     # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
    #     if self.model.generation_config._from_model_config:
    #         self.model.generation_config._from_model_config = False

    #     # Retrieves GenerationConfig from model.generation_config
    #     gen_config = self.model.generation_config
    #     # in case the batch is shorter than max length, the output should be padded
    #     if generated_tokens.shape[-1] < gen_config.max_length:
    #         generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
    #     elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
    #         generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

    #     with torch.no_grad():
    #         if has_labels:
    #             with self.compute_loss_context_manager():
    #                 outputs = model(**inputs)
    #             if self.label_smoother is not None:
    #                 loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
    #             else:
    #                 loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
    #         else:
    #             loss = None

    #     if self.args.prediction_loss_only:
    #         return loss, None, None

    #     if has_labels:
    #         labels = inputs["labels"]
    #         if labels.shape[-1] < gen_config.max_length:
    #             labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
    #         elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
    #             labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
    #     else:
    #         labels = None

    #     return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: "torch.Tensor", tgt_tensor: "torch.Tensor") -> "torch.Tensor":
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.processing_class.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.processing_class.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
