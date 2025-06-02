"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import argparse
import datetime
import json
import os
import random
from io import BytesIO
from os.path import basename
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.utilities import rank_zero_only
from sconf import Config

from donut import DonutDataset
from lightning_module import DonutDataPLModule, DonutModelPLModule


class CustomCheckpointIO(CheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        del checkpoint["state_dict"]
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, storage_options=None):
        checkpoint = torch.load(path + "artifacts.ckpt")
        state_dict = torch.load(path + "pytorch_model.bin")
        checkpoint["state_dict"] = {"model." + key: value for key, value in state_dict.items()}
        return checkpoint

    def remove_checkpoint(self, path) -> None:
        return super().remove_checkpoint(path)


@rank_zero_only
def save_config_file(config, path):
    if not Path(path).exists():
        os.makedirs(path)
    save_path = Path(path) / "config.yaml"
    print(config.dumps())
    with open(save_path, "w") as f:
        f.write(config.dumps(modified_color=None, quote_str=True))
        print(f"Config is saved at {save_path}")


class ProgressBar(pl.callbacks.TQDMProgressBar):
    def __init__(self, config):
        super().__init__()
        self.enable = True
        self.config = config

    def disable(self):
        self.enable = False

    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        items["exp_name"] = f"{self.config.get('exp_name', '')}"
        items["exp_version"] = f"{self.config.get('exp_version', '')}"
        return items


def set_seed(seed):
    pytorch_lightning_version = int(pl.__version__[0])
    if pytorch_lightning_version < 2:
        pl.utilities.seed.seed_everything(seed, workers=True)
    else:
        import lightning_fabric
        lightning_fabric.utilities.seed.seed_everything(seed, workers=True)


def filter_dataset_by_metadata(dataset, filters):
    """
    Filter dataset based on metadata fields in the ground_truth column.
    
    Args:
        dataset: HuggingFace dataset
        filters: Dict with filter conditions
        
    Returns:
        Filtered dataset
    """
    if not filters:
        return dataset
    
    def meets_filter_criteria(example):
        try:
            # Parse the ground_truth JSON if it's a string
            if isinstance(example['ground_truth'], str):
                gt_data = json.loads(example['ground_truth'])
            else:
                gt_data = example['ground_truth']
            
            meta = gt_data.get('meta', {})
            
            # Check each filter condition
            for field_path, expected_value in filters.items():
                # Support nested field access with dot notation (e.g., "meta.version")
                field_parts = field_path.split('.')
                current_data = gt_data
                
                # Navigate through nested structure
                for part in field_parts:
                    if isinstance(current_data, dict) and part in current_data:
                        current_data = current_data[part]
                    else:
                        return False  # Field not found
                
                # Check if the value matches
                if isinstance(expected_value, list):
                    # If expected_value is a list, check if current_data is in the list
                    if current_data not in expected_value:
                        return False
                else:
                    # Direct comparison
                    if current_data != expected_value:
                        return False
            
            return True
            
        except (json.JSONDecodeError, KeyError, TypeError):
            return False
    
    return dataset.filter(meets_filter_criteria)


def create_filtered_dataset(dataset_name_or_path, split, filters, model_module, config, task_start_token, prompt_end_token, max_samples=None):
    """
    Create a DonutDataset with optional filtering based on metadata.
    
    Args:
        dataset_name_or_path: Path or name of the dataset
        split: Dataset split ('train' or 'validation')
        filters: Dictionary of filter conditions
        model_module: Donut model module
        config: Configuration object
        task_start_token: Task start token
        prompt_end_token: Prompt end token
        max_samples: Maximum number of samples to include (for validation size limiting)
        
    Returns:
        DonutDataset instance
    """
    # Create the base dataset
    base_dataset = DonutDataset(
        dataset_name_or_path=dataset_name_or_path,
        donut_model=model_module.model,
        max_length=config.max_length,
        split=split,
        task_start_token=task_start_token,
        prompt_end_token=prompt_end_token,
        sort_json_key=config.sort_json_key,
    )
    
    # Apply filtering if specified
    if filters:
        print(f"Applying filters to {split} split: {filters}")
        original_size = len(base_dataset.dataset)
        base_dataset.dataset = filter_dataset_by_metadata(base_dataset.dataset, filters)
        filtered_size = len(base_dataset.dataset)
        print(f"Dataset {split} split filtered from {original_size} to {filtered_size} examples")
    
    # Apply sample size limiting if specified (for validation set)
    if max_samples is not None and len(base_dataset.dataset) > max_samples:
        print(f"Limiting {split} dataset from {len(base_dataset.dataset)} to {max_samples} samples")
        # Shuffle and select random subset
        indices = list(range(len(base_dataset.dataset)))
        random.shuffle(indices)
        selected_indices = indices[:max_samples]
        base_dataset.dataset = base_dataset.dataset.select(selected_indices)
        print(f"Final {split} dataset size: {len(base_dataset.dataset)} examples")
    
    # Critical fix: Update all dataset size attributes to reflect the filtered dataset
    final_size = len(base_dataset.dataset)
    base_dataset.dataset_length = final_size
    
    # Override the __len__ method to return the correct size
    original_len = base_dataset.__len__
    base_dataset.__len__ = lambda: final_size
    
    # Update any other size-related attributes that might exist
    for attr in ['num_training_samples', 'length', 'size']:
        if hasattr(base_dataset, attr):
            setattr(base_dataset, attr, final_size)
    
    print(f"DonutDataset created with final size: {final_size}")
    
    return base_dataset


def train(config):
    set_seed(config.get("seed", 42))

    model_module = DonutModelPLModule(config)
    data_module = DonutDataPLModule(config)

    # add datasets to data_module
    datasets = {"train": [], "validation": []}
    
    # Get filtering configurations
    train_filters = config.get("train_filters", {})
    val_filters = config.get("val_filters", {})
    
    # Get validation size limit configuration
    val_size_limit_percent = config.get("val_size_limit_percent", 0.1)  # Default 10%
    
    # Get split configurations - allow forcing to use only train split
    force_train_only = config.get("force_train_split_only", False)
    
    for i, dataset_name_or_path in enumerate(config.dataset_name_or_paths):
        task_name = config.get("task_name", os.path.basename(dataset_name_or_path))
        
        # add categorical special tokens (optional)
        if task_name == "rvlcdip":
            model_module.model.decoder.add_special_tokens([
                "<advertisement/>", "<budget/>", "<email/>", "<file_folder/>", 
                "<form/>", "<handwritten/>", "<invoice/>", "<letter/>", 
                "<memo/>", "<news_article/>", "<presentation/>", "<questionnaire/>", 
                "<resume/>", "<scientific_publication/>", "<scientific_report/>", "<specification/>"
            ])
        if task_name == "docvqa":
            model_module.model.decoder.add_special_tokens(["<yes/>", "<no/>"])
        
        # Determine task start and prompt end tokens
        task_start_token = (config.task_start_tokens[i] 
                           if config.get("task_start_tokens", None) 
                           else f"<s_{task_name}>")
        prompt_end_token = ("<s_answer>" if "docvqa" in dataset_name_or_path 
                           else f"<s_{task_name}>")
        
        # Create training dataset
        if force_train_only:
            # Use train split for both training and validation with different filters
            train_dataset = create_filtered_dataset(
                dataset_name_or_path=dataset_name_or_path,
                split="train",
                filters=train_filters,
                model_module=model_module,
                config=config,
                task_start_token=task_start_token,
                prompt_end_token=prompt_end_token
            )
            datasets["train"].append(train_dataset)
            
            # Calculate validation size limit based on training dataset size
            train_size = len(train_dataset.dataset)
            max_val_size = max(1, int(train_size * val_size_limit_percent))
            print(f"Training dataset size: {train_size}, limiting validation to {max_val_size} samples ({val_size_limit_percent*100}%)")
            
            # Create validation dataset from train split with different filters and size limit
            val_dataset = create_filtered_dataset(
                dataset_name_or_path=dataset_name_or_path,
                split="train",  # Using train split for validation too
                filters=val_filters,
                model_module=model_module,
                config=config,
                task_start_token=task_start_token,
                prompt_end_token=prompt_end_token,
                max_samples=max_val_size
            )
            datasets["validation"].append(val_dataset)
            
        else:
            # Use separate train and validation splits
            train_datasets_for_size_calc = []  # Track train datasets for size calculation
            
            for split in ["train", "validation"]:
                filters = train_filters if split == "train" else val_filters
                max_samples = None  # No limit for train split initially
                
                if split == "validation":
                    # Calculate validation size limit based on total training dataset size
                    total_train_size = sum(len(ds.dataset) for ds in train_datasets_for_size_calc)
                    max_samples = max(1, int(total_train_size * val_size_limit_percent))
                    print(f"Total training dataset size: {total_train_size}, limiting validation to {max_samples} samples ({val_size_limit_percent*100}%)")
                
                dataset = create_filtered_dataset(
                    dataset_name_or_path=dataset_name_or_path,
                    split=split,
                    filters=filters,
                    model_module=model_module,
                    config=config,
                    task_start_token=task_start_token,
                    prompt_end_token=prompt_end_token,
                    max_samples=max_samples
                )
                
                datasets[split].append(dataset)
                
                if split == "train":
                    train_datasets_for_size_calc.append(dataset)

    data_module.train_datasets = datasets["train"]
    data_module.val_datasets = datasets["validation"]

    logger = TensorBoardLogger(
        save_dir=config.result_path,
        name=config.exp_name,
        version=config.exp_version,
        default_hp_metric=False,
    )

    lr_callback = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_metric",
        dirpath=Path(config.result_path) / config.exp_name / config.exp_version,
        filename="artifacts",
        save_top_k=1,
        save_last=False,
        mode="min",
    )

    bar = ProgressBar(config)

    custom_ckpt = CustomCheckpointIO()
    trainer = pl.Trainer(
        num_nodes=config.get("num_nodes", 1),
        devices=torch.cuda.device_count(),
        #strategy="ddp",
        accelerator="gpu",
        plugins=custom_ckpt,
        max_epochs=config.max_epochs,
        max_steps=config.max_steps,
        val_check_interval=config.val_check_interval,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        gradient_clip_val=config.gradient_clip_val,
        precision=16,
        num_sanity_val_steps=0,
        logger=logger,
        callbacks=[lr_callback, checkpoint_callback, bar],
    )

    trainer.fit(model_module, data_module, ckpt_path=config.get("resume_from_checkpoint_path", None))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_version", type=str, required=False)
    args, left_argv = parser.parse_known_args()

    config = Config(args.config)
    config.argv_update(left_argv)

    config.exp_name = basename(args.config).split(".")[0]
    config.exp_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if not args.exp_version else args.exp_version

    save_config_file(config, Path(config.result_path) / config.exp_name / config.exp_version)
    train(config)