import dataclasses
import os
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Type, Union

import torch
import torch.nn as nn
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BaseImageProcessor,
    DataCollator,
    DataCollatorForLanguageModeling,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainingArguments,
    is_wandb_available,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils.deprecation import deprecate_kwarg

from trl.trainer.sft_config import SFTConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

if is_wandb_available():
    import wandb


class SFTTrainer(Trainer):
    """
    Simplified Trainer for Supervised Fine-Tuning (SFT) method.

    This class is a wrapper around the [`transformers.Trainer`] class and inherits all of its attributes and methods.
    Simplified to work with custom DataLoaders and without PEFT/Liger support.

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:
            - A string, being the *model id* of a pretrained model hosted on huggingface.co
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        args ([`SFTConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        data_collator (`DataCollator`, *optional*):
            Function to use to form a batch from a list of elements.
        train_dataset ([`~torch.utils.data.DataLoader`]):
            DataLoader to use for training.
        eval_dataset ([`~torch.utils.data.DataLoader`] or `dict[str, DataLoader]`):
            DataLoader(s) to use for evaluation.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use.
        optimizer_cls_and_kwargs (`Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]`, *optional*, defaults to `None`):
            A tuple containing the optimizer class and keyword arguments to use.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*, defaults to `None`):
            A function that preprocess the logits right before caching them at each evaluation step.
    """

    @deprecate_kwarg(
        "tokenizer", "0.16.0", "processing_class", warn_if_greater_or_equal_version=True, raise_if_both_names=True
    )
    def __init__(
            self,
            model: Union[str, nn.Module, PreTrainedModel],
            args: Optional[Union[SFTConfig, TrainingArguments]] = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset, DataLoader]] = None,
            eval_dataset: Optional[Union[Dataset, dict[str, Dataset], DataLoader, dict[
                str, Union[Dataset, IterableDataset, DataLoader]]]] = None,
            processing_class: Optional[
                Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
            ] = None,
            compute_loss_func: Optional[Callable] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None, None),
            optimizer_cls_and_kwargs: Optional[tuple[Type[torch.optim.Optimizer], dict[str, Any]]] = None,
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = SFTConfig(f"{model_name}-SFT")
        elif isinstance(args, TrainingArguments) and not isinstance(args, SFTConfig):
            dict_args = args.to_dict()
            dict_args["hub_token"] = args.hub_token  # to_dict hides the hub_token
            dict_args.pop("push_to_hub_token")
            args = SFTConfig(**dict_args)

        # Model
        if args.model_init_kwargs is not None and not isinstance(model, str):
            warnings.warn(
                "You passed model_init_kwargs to the `SFTConfig`, but your model is already instantiated. "
                "The `model_init_kwargs` will be ignored."
            )
        if isinstance(model, str):
            model = self._create_model_from_path(model, args)

        # Handle the tokenizer
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path)
            if processing_class.pad_token is None:
                processing_class.pad_token = processing_class.eos_token

        # Data collator - set to None if using DataLoader as it should handle its own collation
        if data_collator is None and not isinstance(train_dataset, DataLoader):
            data_collator = DataCollatorForLanguageModeling(tokenizer=processing_class, mlm=False)

        # Initialize the metrics
        self._metrics = defaultdict(list)

        # Initialize the Trainer
        super_init_kwargs = {}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super_init_kwargs["optimizer_cls_and_kwargs"] = optimizer_cls_and_kwargs
        else:
            if optimizer_cls_and_kwargs is not None:
                warnings.warn(
                    "The `optimizer_cls_and_kwargs` argument is only available for `transformers>=4.47.0`. "
                    "The default optimizer will be used. "
                    "Remove the `optimizer_cls_and_kwargs` or upgrade to `transformers>=4.47.0`."
                )
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **super_init_kwargs,
        )

    def _create_model_from_path(self, model_path: str, args: SFTConfig) -> PreTrainedModel:
        """Creates a model from a path or model identifier."""
        model_init_kwargs = args.model_init_kwargs or {}
        # Handle torch dtype
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
            model_init_kwargs["torch_dtype"] = torch_dtype
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `SFTConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        # Disable caching if gradient checkpointing is enabled (not supported)
        if args.gradient_checkpointing:
            model_init_kwargs["use_cache"] = False

        # Create model
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_init_kwargs)
        return model

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        (loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        # Compute token accuracy if we have labels
        if "labels" in inputs:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()

            # Get predictions
            predictions = shift_logits.argmax(dim=-1)

            # Create mask for non-padding tokens (assuming ignore_index is -100)
            mask = shift_labels != -100

            # Calculate accuracy only on non-padding tokens
            correct_predictions = (predictions == shift_labels) & mask
            total_tokens = mask.sum()
            correct_tokens = correct_predictions.sum()

            # Gather the correct_tokens and total_tokens across all processes
            correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
            total_tokens = self.accelerator.gather_for_metrics(total_tokens)

            # Compute the mean token accuracy and log it
            accuracy = (correct_tokens.sum() / total_tokens.sum()).item() if total_tokens.sum() > 0 else 0.0
            self._metrics["mean_token_accuracy"].append(accuracy)

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
            self,
            model_name: Optional[str] = None,
            dataset_name: Optional[str] = None,
            tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="SFT",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    # If user passed a DataLoader, use it directly for training
    def get_train_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, DataLoader):
            return self.train_dataset
        return super().get_train_dataloader()

    # If user passed a DataLoader (or in evaluate()), use it directly for evaluation
    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        ds = self.eval_dataset if eval_dataset is None else eval_dataset
        if isinstance(ds, DataLoader):
            return ds
        return super().get_eval_dataloader(eval_dataset)