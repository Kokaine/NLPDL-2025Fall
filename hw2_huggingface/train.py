import os
import sys
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
# import torch_npu ## Uncomment this line if you are using Ascend NPU

from datasets import Dataset, DatasetDict
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    ## TODO: import other classes you need
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    HfArgumentParser,
    set_seed,
)
import evaluate
import peft
import adapters
import swanlab
from peft import LoraConfig, TaskType, get_peft_model
from swanlab.integration.transformers import SwanLabCallback

from dataHelper import get_dataset

# os.environ["WANDB_MODE"] = "offline" ## Uncomment this line if you cannot connect to wandb server


@dataclass
class BaseArgs:
    '''Base arguments for the training script.'''

    ## TODO: define your arguments here and set default values or fields
    dataset: str = field(
        metadata={"help": "The name of the dataset to use (e.g., restaurant_sup, acl_sup, agnews_sup)."}
    )
    model_name: str = field(
        metadata={"help": "Pretrained model or model identifier from huggingface.co/models"}
    )
    sep_token: str = field(
        default="<SEP>", metadata={"help": "The separator token used in the dataset."}
    )
    peft: Optional[str] = field(
        default=None, metadata={"help": "The PEFT method to use ('lora' or 'adapter'). Leave None for full finetuning."}
    )
    max_length: int = field(
        default=256, metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Enable trust_remote_code for custom models."}
    )


@dataclass
class LoraArgs:
    '''Arguments for LoRA.'''

    ## TODO: define your arguments here and set default values or fields
    rank: int = field(default=8, metadata={"help": "LoRA attention dimension."})
    alpha: int = field(default=16, metadata={"help": "LoRA alpha."})
    dropout: float = field(default=0.1, metadata={"help": "LoRA dropout."})

@dataclass
class AdapterArgs:
    '''Arguments for Bottleneck Adapters.'''
    reduction_factor: int = field(
        default=16, metadata={"help": "Reduction factor for the bottleneck adapter."}
    )

logger = logging.getLogger(__name__)


def print_trainable_parameters(model):
    """
    Print out the number of trainable parameters in the model.
    transformers models have a `print_trainable_parameters` method, but not all models have it.
    """

    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params:,} || "
          f"all params: {all_params:,} || "
          f"trainable%: {100 * trainable_params / all_params:.2f}%")


def parse_arguments():
    '''Parse command line arguments into dataclasses.'''

    ## TODO: parse arguments
    parser = HfArgumentParser((BaseArgs, TrainingArguments, AdapterArgs, LoraArgs))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        base_args, train_args, adapter_args, lora_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        base_args, train_args, adapter_args, lora_args = parser.parse_args_into_dataclasses()
    
    if train_args.label_names is None:
        train_args.label_names = ["labels"]
    return base_args, train_args, adapter_args, lora_args


def set_random_seed(seed: int) -> None:
    '''Set random seed for training.'''

    ## TODO: set random seed
    set_seed(seed)


def set_logger() -> None:
    '''Set up the logger to print messages to stdout.'''

    ## TODO: set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_data(dataset_name: str,
              sep_token: str = "<SEP>") -> Tuple[DatasetDict, int]:
    '''Load dataset and return the number of labels.'''

    ## TODO: load dataset using `get_dataset()`
    raw_dataset = get_dataset(dataset_name, sep_token=sep_token)
    
    unique_labels = set()
    for split in raw_dataset.keys():
        unique_labels.update(raw_dataset[split]['label'])
    num_labels = len(unique_labels)
    
    logger.info(f"Loaded dataset '{dataset_name}' with {num_labels} labels.")
    return raw_dataset, num_labels


def get_model(model_name: str, num_labels: int, trust_remote_code: bool):
    '''Load model and tokenizer.'''

    ## TODO: get model and tokenizer
    config = AutoConfig.from_pretrained(
        model_name, 
        num_labels=num_labels,
        trust_remote_code=trust_remote_code
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=trust_remote_code
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        config=config,
        trust_remote_code=trust_remote_code
    )

    ## TODO: set pad_token for tokenizer and model (Why?)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model


def tokenize_data(raw_dataset, tokenizer, max_length: int = 256):
    '''Tokenize the dataset.'''

    ## TODO: tokenize dataset using tokenizer and dataset.map()
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=max_length
        )

    logger.info("Tokenizing datasets...")
    tokenized_datasets = raw_dataset.map(
        tokenize,
        batched=True
    )
    # if "label" in tokenized_datasets["train"].column_names:
    #     tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    train_dataset, eval_dataset = tokenized_datasets["train"], tokenized_datasets["test"]
    return train_dataset, eval_dataset


def get_lora_model(model, lora_args):
    '''Initialize the model with LoRA modules.'''

    ## TODO: use LoRA to wrap your model
    logger.info(f"Applying LoRA on model={model}: rank={lora_args.rank}, alpha={lora_args.alpha}")
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        inference_mode=False, 
        r=lora_args.rank, 
        lora_alpha=lora_args.alpha, 
        lora_dropout=lora_args.dropout
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def get_adapter_model(model, adapter_args):
    '''Initialize the model with Bottleneck Adapter modules.'''

    ## TODO: use Adapter to wrap your model
    logger.info(f"Applying Bottleneck Adapter: reduction_factor={adapter_args.reduction_factor}")
    
    # Initialize adapters library
    adapters.init(model)
    
    # Define adapter config (Pfeiffer is a standard bottleneck configuration)
    adapter_config = adapters.AdapterConfig.load(
        "pfeiffer", 
        reduction_factor=adapter_args.reduction_factor
    )
    
    # Add a new adapter
    adapter_name = "default_adapter"
    model.add_adapter(adapter_name, config=adapter_config)
    
    # Activate the adapter for training
    model.train_adapter(adapter_name)
    print_trainable_parameters(model)
    return model


def get_data_collator(tokenizer):
    '''Define data collator for padding.'''

    ## TODO: define data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return data_collator


def compute_metrics(pred) -> Dict[str, float]:
    '''Compute accuracy, macro F1 and micro F1.'''
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    
    if hasattr(pred, "predictions") and hasattr(pred, "label_ids"):
        logits = pred.predictions
        labels = pred.label_ids
    else:
        logits, labels = pred

    if isinstance(logits, tuple):
        logits = logits[0]
        
    predictions = np.argmax(logits, axis=1)

    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1_micro = f1_metric.compute(predictions=predictions, references=labels, average="micro")
    f1_macro = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    f1_weighted = f1_metric.compute(predictions=predictions, references=labels, average="weighted")

    ## TODO: compute metrics with `evaluate` package
    return {
        "accuracy": acc["accuracy"],
        "macro_f1": f1_macro["f1"],
        "micro_f1": f1_micro["f1"],
        "weighted_f1": f1_weighted["f1"]
    }


def get_trainer(model, args, train_dataset, eval_dataset, tokenizer, data_collator):
    '''Define Trainer for training and evaluation.'''
    
    ## TODO: define Trainer with appropriate arguments

    if hasattr(model, "adapters_config"):
        logger.info("Using Adapter Trainer.")
        from adapters import AdapterTrainer
        trainer = AdapterTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[SwanLabCallback()]
        )
    else:
        logger.info("Using standard Trainer.")
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[SwanLabCallback()]
        )
        
    return trainer


def main():
    # Parse arguments
    base_args, train_args, adapter_args, lora_args = parse_arguments()

    # Set seed before initializing model.
    set_random_seed(train_args.seed)

    # Set up logging
    set_logger()

    # load dataset
    raw_dataset, num_labels = load_data(base_args.dataset, base_args.sep_token)

    # get model and tokenizer
    tokenizer, model = get_model(base_args.model_name, num_labels, base_args.trust_remote_code)

    # tokenize dataset
    train_dataset, eval_dataset = tokenize_data(raw_dataset, tokenizer,
                                              base_args.max_length)

    # peft method
    if base_args.peft != None:
        if base_args.peft.lower() == "lora":
            model = get_lora_model(model, lora_args)
        elif base_args.peft.lower() == "adapter":
            model = get_adapter_model(model, adapter_args)
        else:
            raise ValueError("Unsupported PEFT method!")

    data_collator = get_data_collator(tokenizer)

    ## TODO: initialize wandb and set config

    swanlab.init(project=os.environ.get("SWANLAB_PROJECT", "HF_TRAINING"), 
                   name=f"{base_args.model_name}-{base_args.peft or 'full'}")
    swanlab.config.update({
        "epoch": train_args.num_train_epochs,
        "batch_size": train_args.per_device_train_batch_size,
        "lr": train_args.learning_rate,
        "peft": base_args.peft,
        "model": base_args.model_name
    })

    if train_args.do_train:
        logger.info("*** Train ***")
        ## TODO: define Trainer and start training
        trainer = get_trainer(model, train_args, train_dataset, eval_dataset, tokenizer, data_collator)
        train_result = trainer.train()
        metrics = train_result.metrics

        ## TODO: save model, state and metricss
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if train_args.do_eval:
        logger.info("*** Evaluation ***")

        ## TODO: run evaluation and get metrics
        if 'trainer' not in locals():
            trainer = get_trainer(model, train_args, train_dataset, eval_dataset, tokenizer, data_collator)

        ## TODO: log metrics
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if train_args.do_predict:
        logger.info("*** Predict ***")
        ## TODO: predict without checking, and save the results to `predict_results.txt`
        if 'trainer' not in locals():
            trainer = get_trainer(model, train_args, train_dataset, eval_dataset, tokenizer, data_collator)
        
        # Using eval_dataset for prediction demo, normally this would be a separate 'predict' split
        predictions = trainer.predict(eval_dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        output_predict_file = os.path.join(train_args.output_dir, "predict_results.txt")
        
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                writer.write("index\tprediction\n")
                for index, item in enumerate(preds):
                    writer.write(f"{index}\t{item}\n")


if __name__ == "__main__":
    main()
