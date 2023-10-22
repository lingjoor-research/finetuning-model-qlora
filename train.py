import os
import logging
import warnings
warnings.filterwarnings("ignore")

import wandb
import pandas as pd
from huggingface_hub import login
from transformers import (AutoModelForCausalLM, AutoTokenizer, 
                          BitsAndBytesConfig, TrainingArguments)
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig
from trl import SFTTrainer
import bitsandbytes as bnb

from dotenv import load_dotenv
load_dotenv()

import config

# Environment Setup
login(token=os.environ.get("HF_TOKEN"), add_to_git_credential=True)

# set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def find_linear_layers(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit): 
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def modify_dataset_2_record(record):
    input_value = record['input'] if record['input'] is not None else ""
    context_value = record['context'] if record['context'] is not None else ""
    
    return {
        'instruction': record['instruction'],
        'context': input_value + context_value[:512],
        'response': record['output'],
        'category': None  
    }

def modify_dataset_3_record(record):
    input_value = record['input'] if record['input'] is not None else ""
    context_value = record['_context'] if '_context' in record and record['_context'] is not None else ""
    combined_context = input_value + " " + context_value  

    return {
        'instruction': f"([quality] {record['quality_gain']} [/quality]) + {record['instruction']}",
        'context': combined_context,
        'response': record['output'],
        'category': None 
    }

def modify_dataset_4_record(record):
    context_value = record['input'] + record['conversations'] if (record['input'] is not None and record['conversations'] is not None) else ""
    instruction_value = record['prompt'] if record['prompt'] is not None else ""
    response_value = record['completion'] if record['completion'] is not None else ""
    
    return {
        'instruction': instruction_value,
        'context': context_value,
        'response': response_value,
        'category': None 
    }

def create_combined_text(record):
    instruction_value = record['instruction'] if record['instruction'] is not None else ""
    context_value = record['context'] if record['context'] is not None else ""
    response_value = record['response'] if record['response'] is not None else ""
    
    # Wrapping the instruction with [INST] tokens
    instruction_formatted = f"<s> [INST] {instruction_value} [/INST]"
    
    return {
        'combined_text': instruction_formatted +
                         '### context: ' + context_value[:1280] +
                         '### response: ' + response_value +' </s>' 
    }

class TrainingPipeline:
    def __init__(self):
        self.datasets = {}
        self.model = None
        self.tokenizer = None
        self.final_dataset = None
        self.target_modules = []
        self.qlora_config = None

    def load_datasets(self):
        logging.info("Loading datasets...")

        self.datasets = {
            'dataset_1': load_dataset("lingjoor/databricks-dolly-15k-context-32k-rag"),
            'dataset_4': load_dataset("lingjoor/lima_with_scores"),
            'dataset_5': load_dataset("alexMTL/guanaco_q_a_dataset_1k")
        }

        logging.info("Datasets loaded successfully!")

    def preprocess_datasets(self):
        logging.info("Preprocessing datasets...")

        dataset_4_modified = self.datasets['dataset_4']['train'].map(modify_dataset_4_record)
        dataset_5_modified = self.datasets['dataset_5']['train'].map(lambda record: {'combined_text': record['text']})
        concatenated_dataset = concatenate_datasets(
            [self.datasets['dataset_1']['train'],
            dataset_4_modified, 
            dataset_5_modified])
        self.final_dataset = concatenated_dataset.map(create_combined_text)

        logging.info("Datasets preprocessed successfully!")

    def initialize_model_and_tokenizer(self):
        logging.info("Initializing model and tokenizer...")
        model_name = config.model_name
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.use_4bit,
            bnb_4bit_use_double_quant=config.use_nested_quant,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtyp=config.bnb_4bit_compute_dtype,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            trust_remote_code=True)

        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Find linear layers and configure LoRA
        self.target_modules = find_linear_layers(self.model)
        if not self.target_modules:
            logging.error("No target modules found for LoRA. Check the model architecture.")
            return
        self.qlora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

    def train(self):
        logging.info("Starting training...")
        # Initialize Wandb
        wandb.init(project="1-epoch-dolly-15k-context-32k-rag-lima-guanaco-neft-qlora")

        # Training Configuration and Execution
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            optim=config.optim,
            logging_steps=config.logging_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            fp16=config.fp16,
            bf16=config.bf16,
            max_grad_norm=config.max_grad_norm,
            # max_steps=config.max_steps,
            warmup_ratio=config.warmup_ratio,
            group_by_length=config.group_by_length,
            lr_scheduler_type=config.lr_scheduler_type,
            save_total_limit=config.save_total_limit,
            # evaluation_strategy="no",
            save_strategy=config.save_strategy,
            report_to=config.report_to,
            # load_best_model_at_end=config.load_best_model_at_end,
            seed=config.seed,
            )
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.final_dataset,
            args=training_args,
            tokenizer=self.tokenizer,
            peft_config=self.qlora_config,  
            dataset_text_field='combined_text',
            max_seq_length=config.max_seq_length,
            neftune_noise_alpha=config.neftune_noise_alpha,
        )
        trainer.train()

        logging.info("Training completed!")

        # Save the Model
        logging.info("Saving the model...")
        model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        model_to_save.save_pretrained(config.save_dir)
        logging.info("Saving completed!")

    def run_pipeline(self):
        self.load_datasets()
        self.preprocess_datasets()
        self.initialize_model_and_tokenizer()
        self.train()

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()