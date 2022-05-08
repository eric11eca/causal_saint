import os
import math
import torch
import argparse
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

from plm.utils.python.py_io import read_jsonl
from plm.utils.zlog import ZLogger
from plm.utils.datastructures import BiDict

from torch.utils.data import Dataset, ConcatDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class DefeasibleCausalDataset(Dataset):
    def __init__(self, model_name, data_pth, intervention, max_len_inp):
        self.examples = read_jsonl(data_pth)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len_input = max_len_inp
        self.intervention = intervention
        self.inputs = []
        self.inputs_counter = []
        self.update_types = []
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        x = self.inputs[index]
        x['input_ids'] = x['input_ids'].squeeze(0)
        x['attention_mask'] = x['attention_mask'].squeeze(0)
        x_hat = self.inputs_counter[index]
        x_hat['input_ids'] = x_hat['input_ids'].squeeze(0)
        x_hat['attention_mask'] = x_hat['attention_mask'].squeeze(0)
        update_type = self.update_types[index]

        return {
            "x": x,
            "x_hat": x_hat,
            "update_type": update_type,
        }

    def _build(self):
        for data in self.examples:
            premise = data["Premise"]
            premise_hat = f"{premise}. {data['Update']}"
            hypothesis = data['Hypothesis']
            update_type = data["UpdateType"]

            if update_type != self.intervention:
                continue

            tokenized_inputs = self.tokenizer(
                premise, hypothesis,
                truncation=True,
                padding="max_length",
                max_length=self.max_len_input,
                return_tensors="pt")

            tokenized_inputs_counter = self.tokenizer(
                premise_hat, hypothesis,
                truncation=True,
                padding="max_length",
                max_length=self.max_len_input,
                return_tensors="pt")

            self.update_types.append(update_type)
            self.inputs.append(tokenized_inputs)
            self.inputs_counter.append(tokenized_inputs_counter)


class DefeasibleCausalDataModule(pl.LightningDataModule):
    def __init__(self, model_name, data_pth, intervention, save_to_cache=False, max_len_inp=128):

        if save_to_cache:
            self.train_dataset = DefeasibleCausalDataset(
                model_name,
                data_pth=f"{data_pth}/train.jsonl",
                intervention=intervention,
                max_len_inp=max_len_inp)

            self.dev_dataset = DefeasibleCausalDataset(
                model_name,
                data_pth=f"{data_pth}/dev.jsonl",
                intervention=intervention,
                max_len_inp=max_len_inp)

            self.test_dataset = DefeasibleCausalDataset(
                model_name,
                data_pth=f"{data_pth}/test.jsonl",
                intervention=intervention,
                max_len_inp=max_len_inp)

            os.makedirs(f"{data_pth}/cache/{intervention}", exist_ok=True)

            torch.save(self.train_dataset,
                       f"{data_pth}/cache/{intervention}/train.pt")
            torch.save(self.dev_dataset,
                       f"{data_pth}/cache/{intervention}/dev.pt")
            torch.save(self.test_dataset,
                       f"{data_pth}/cache/{intervention}/test.pt")
        else:
            self.train_dataset = torch.load(
                f"{data_pth}/cache/{intervention}/train.pt")
            self.dev_dataset = torch.load(
                f"{data_pth}/cache/{intervention}/dev.pt")
            self.test_dataset = torch.load(
                f"{data_pth}/cache/{intervention}/test.pt")

        self.train_batch_size = 8

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size)

    def test_dataloader(self):
        dataset = ConcatDataset([self.dev_dataset, self.test_dataset])
        return DataLoader(dataset, batch_size=128)


class StrengthenerDataModule(DefeasibleCausalDataModule):
    def __init__(self, model_name, data_pth, save_to_cache=False, max_len_inp=72):
        super().__init__(
            model_name,
            data_pth,
            intervention="strengthener",
            save_to_cache=save_to_cache,
            max_len_inp=max_len_inp
        )


class WeakenerDataModule(DefeasibleCausalDataModule):
    def __init__(self, model_name, data_pth, save_to_cache=False, max_len_inp=72):
        super().__init__(
            model_name,
            data_pth,
            intervention="weakener",
            save_to_cache=save_to_cache,
            max_len_inp=max_len_inp
        )


class DefeasibleCausalProbe(pl.LightningModule):
    def __init__(self, model_name, probe_name):
        super().__init__()
        self.total_effects = []
        self.model_name = model_name
        self.probe_name = probe_name
        self.zlogger = ZLogger("zlogger", overwrite=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name)

    def test_step(self, batch, batch_idx):
        x = batch['x']
        x_hat = batch['x_hat']
        with torch.no_grad():
            y = self.model(**x).logits
            y_hat = self.model(**x_hat).logits
            prob = torch.softmax(y, dim=1)
            prob_hat = torch.softmax(y_hat, dim=1)

            output = {
                "prob": prob,
                "prob_hat": prob_hat
            }

            return output

    def test_step_end(self, batch_parts):
        prob = batch_parts["prob"]
        prob_hat = batch_parts["prob_hat"]
        bias = self.inference_bias(
            prob[:, 0], prob[:, 1] + prob[:, 2])
        bias_hat = self.inference_bias(
            prob_hat[:, 0], prob_hat[:, 1] + prob_hat[:, 2])
        total_effect_batch = self.total_effect(
            bias_hat, bias).cpu().detach().numpy()
        return total_effect_batch

    def test_epoch_end(self, outputs):
        total_effects = np.concatenate(outputs)
        for i, effect in enumerate(total_effects):
            self.zlogger.write_entry(
                f"{self.probe_name}-eval_log",
                {
                    "guid": i,
                    "total_effect": effect.tolist()
                })
        total_effect_avg = self.total_effect_avg(
            total_effects, len(total_effects))
        self.zlogger.write_entry(
            f"{self.probe_name}-avg_eval_log",
            {
                "total_effect_avg": total_effect_avg.tolist()
            })
        self.log("total_effect_avg", total_effect_avg)

    def inference_bias(self, prob_entail, prob_non):
        return prob_entail / prob_non

    def total_effect(self, inference_bias, inference_bias_null):
        return torch.atan((inference_bias / inference_bias_null) - 1) / (math.pi / 2)

    def total_effect_avg(self, total_effects, num_examples):
        return np.sum(total_effects) / num_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="snli",
                        help="probing dataset name")
    parser.add_argument("--model", type=str, default="anli_roberta",
                        help="probing model name")
    parser.add_argument("--intervention", type=str, default="strengthener",
                        help="intervention type")
    parser.add_argument("--save_to_cache", action='store_true',
                        help="save tokenized data to cache")
    args = parser.parse_args()

    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        )
    )

    # wandb_logger = WandbLogger(
    # name='delta_snli_pos-0.001', project='causal_semantic_alignment')

    model_dict = BiDict({
        "mnli_bert": "sentence-transformers/bert-large-nli-mean-tokens",
        "mnli_roberta": "roberta-large-mnli",
        "mnli_deberta": "microsoft/deberta-base-mnli",
        "mnli_bart": "textattack/facebook-bart-large-MNLI",
        "anli_roberta": "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
        "anli_xlnet": "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli",
        "anli_deberta": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    })

    do_operations = {
        "strengthener": StrengthenerDataModule,
        "weakener": WeakenerDataModule
    }

    data_pth = f"data/defeasible-nli/{args.task}"
    model_name = model_dict[args.model]
    probe_name = f"{args.task}_{args.model}_{args.intervention}"

    intervention_data_module = do_operations[args.intervention](
        model_name, data_pth, save_to_cache=args.save_to_cache)

    causal_prober = DefeasibleCausalProbe(model_name, probe_name)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        # logger=wandb_logger,
        callbacks=[progress_bar]
    )

    trainer.test(
        causal_prober, intervention_data_module.test_dataloader())
