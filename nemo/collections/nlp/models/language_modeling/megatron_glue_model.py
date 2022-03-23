# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import torch
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.common.metrics.classification_accuracy import ExactStringPerCategoryMatchMetric
from nemo.collections.nlp.data.glue_benchmark.glue_benchmark_dataset import TextToTextGLUEDataset
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
)
from nemo.utils import AppState, logging

try:
    from apex.transformer import parallel_state
    from apex.transformer.pipeline_parallel.utils import _reconfigure_microbatch_calculator

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


__all__ = ['MegatronT5GLUEModel']


class MegatronT5GLUEModel(MegatronT5Model):
    """GLUE Model that Inherits from MegatronT5Model instead.""" 
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        if parallel_state.is_pipeline_first_stage():
            self.acc_metric = ExactStringPerCategoryMatchMetric()

    def setup(self, stage=None):
        # This is just to keep the parent class happy since we override its setup() method.
        self.init_consumed_samples = 0

        if stage == 'predict':
            return

        # NOTE: PTL uses the same stage string "test" for both testing and validation.
        self.build_train_valid_test_datasets(validation_only=stage == 'validate')
        self.setup_validation_data()
        if stage == 'validate':
            return
        self.setup_training_data()

    def on_validation_model_eval(self):
        app_state = AppState()
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=self.cfg.data.validation_ds.global_batch_size,
            micro_batch_size=self.cfg.data.validation_ds.micro_batch_size,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        return super().on_validation_model_eval()

    def on_validation_model_train(self):
        app_state = AppState()
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=self.cfg.data.train_ds.global_batch_size,
            micro_batch_size=self.cfg.data.train_ds.micro_batch_size,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        return super().on_validation_model_train()

    def inference_step(self, batch, batch_idx):
        # Call parent validation step to get the loss.
        loss = super().validation_step(batch, batch_idx, reconfigure_microbatch_size=True)

        # Remainder of the code is to run the decoding loop, and compute accuracies.
        tokens_enc, _, _, labels, enc_mask, _ = self.process_global_batch(batch)

        predicted_token_ids, _ = self.decode(tokens_enc=tokens_enc, enc_mask=enc_mask, num_tokens_to_generate=10)

        preds = predicted_token_ids.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()
        for _, (pred, label) in enumerate(zip(preds, labels)):
            if self.tokenizer.eos_id in pred:
                idx = pred.index(self.tokenizer.eos_id)
                pred = pred[:idx]

            # Legacy sentencepiece detokenization still preserves special tokens which messes up exact string match.
            if hasattr(self.tokenizer, 'special_token_to_id'):
                pred = [id for id in pred if id not in self.tokenizer.special_token_to_id.values()]
                label = [id for id in label if id not in self.tokenizer.special_token_to_id.values()]
            pred = self.tokenizer.ids_to_text(pred)
            label = self.tokenizer.ids_to_text(label)
            _ = self.acc_metric(pred, label)

        return loss

    def inference_epoch_end(self, outputs):
        # This will handle logging of the loss.
        super().validation_epoch_end(outputs)
        # TODO: Add logging of the accuracy.

    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        val_loss, val_acc = self.inference_epoch_end(outputs)
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)
        logging.info(f'Validation loss: {val_loss}')
        logging.info(f'Validation accuracy: {val_acc}')

    def test_step(self, batch, batch_idx):
        raise NotImplementedError(
            "Testing is not supported for GLUE because the test data does not have labels. To evaluate on the validation dataset, call trainer.validate(model)"
        )

    def test_epoch_end(self, outputs):
        raise NotImplementedError(
            "Testing is not supported for GLUE because the test data does not have labels. To evaluate on the validation dataset, call trainer.validate(model)"
        )

    def build_pretraining_data_loader(self, dataset, batch_size, shuffle, num_workers, pin_memory, drop_last):
        """Buld dataloader given an input dataset."""

        if dataset is None:
            return None

        rank = parallel_state.get_data_parallel_rank()
        world_size = parallel_state.get_data_parallel_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )

        # Data loader. Note that batch size is the per GPU batch size.
        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    def setup_training_data(self):
        self._train_dl = self.build_pretraining_data_loader(
            self._train_ds,
            self.cfg.data.train_ds.micro_batch_size,
            shuffle=self.cfg.data.train_ds.shuffle,
            num_workers=self.cfg.data.train_ds.num_workers,
            pin_memory=self.cfg.data.train_ds.pin_memory,
            drop_last=self.cfg.data.train_ds.drop_last,
        )

    def setup_validation_data(self):
        self._validation_dl = self.build_pretraining_data_loader(
            self._validation_ds,
            self.cfg.data.validation_ds.micro_batch_size,
            shuffle=self.cfg.data.validation_ds.shuffle,
            num_workers=self.cfg.data.validation_ds.num_workers,
            pin_memory=self.cfg.data.validation_ds.pin_memory,
            drop_last=self.cfg.data.validation_ds.drop_last,
        )

    def build_train_valid_test_datasets(self, validation_only=False):
        logging.info('Building GLUE datasets.')
        self._validation_ds = TextToTextGLUEDataset(
            self.cfg.data.validation_ds.file_path,
            task_name=self.cfg.data.validation_ds.task_name,
            tokenizer=self.tokenizer,
            max_seq_length=self.cfg.data.validation_ds.max_seq_length,
        )
        if validation_only:
            return None, self._validation_ds
        self._train_ds = TextToTextGLUEDataset(
            self.cfg.data.train_ds.file_path,
            task_name=self.cfg.data.train_ds.task_name,
            tokenizer=self.tokenizer,
            max_seq_length=self.cfg.data.train_ds.max_seq_length,
        )
        logging.info(f'Length of train dataset: {len(self._train_ds)}')
        logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        logging.info(f'Finished building GLUE datasets.')
        return self._train_ds, self._validation_ds
