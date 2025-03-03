import torch, os, math, json,wandb, logging
import torch.nn as nn
from tqdm import tqdm
from .magic_utils import GetLoss
from transformers import AdamW, get_linear_schedule_with_warmup
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

"""
This is a general class for training neueral models, it supports 
accelerate and deepspeed to speedup training procedure. To use the
API to train your model, you need initialize your model and tokenizer
and pass them as parameters to the initialization function. In addition,
the inference function, the function of computing evaluation score,
and the function of prrocessing outputs during 
inference are also needed.

The function of inference: inference(model,tokenizer,batch,do_sample)
The function of computing score: compute_score(results)
The function of processing outputs: process_outs(tokenizer,accelerator, batch_outputs, batch)
"""

class MagicModel(nn.Module):
    def __init__(self, model, tokenizer, 
                 cache_dir=None, loss_function=None, inference=None, 
                 compute_score=None, process_outs=lambda tokenizer,outs,batch:outs,
                 init_eval_score= 10000, optimize_direction='min'):
        nn.Module.__init__(self)
        self._model = model
        self._tokenizer = tokenizer

        self._optimizer = None
        self._global_step = 0
        self._lr_scheduler = None

        self._dataset = {}
        self._eval_steps = None
        self._log_dir = None
        self._log_file = None
        self._best_eval_score = init_eval_score
        self._optimize_direction = optimize_direction

        self.get_loss = loss_function if loss_function is not None else GetLoss
        self.inference = inference
        self.process_outs = process_outs
        self.compute_score=compute_score

    def get_optimizer(self, lr, training_steps, warmup_steps,
                      weight_decay, adam_epsilon):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self._model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay},
            {"params": [p for n, p in self._model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0
             }
        ]
        self._optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
        self._lr_scheduler = get_linear_schedule_with_warmup(self._optimizer, num_warmup_steps=warmup_steps,
                                                             num_training_steps=training_steps)

    def save_model(self, model_path):
        torch.save(self._model.state_dict(),model_path)



    def load_data(self, split, data_loader):
        self._dataset[split] = data_loader

    def train_epoch(self, epoch=0, no_tqdm=False, inference_with_sampling=False,accumulated_size=None):
        assert "train" in self._dataset
        logger.info(f'==>>>there are [{len(self._dataset["train"])}] batches...')
        for batch in (self._dataset["train"] if no_tqdm else tqdm(self._dataset["train"])):
            total_batch_num = len(self._dataset["train"])
            self._model.train()
            #self._optimizer.zero_grad()
            #batch_loss = self.get_loss(self._model,batch)
            #batch_loss.backward()
            #self._optimizer.step()
            #self._lr_scheduler.step()
            #self._global_step += 1
            #wandb.log({'loss': batch_loss.item()})

            batch_loss = self.get_loss(self._model, batch)
            wandb.log({'loss': batch_loss.item()})
            batch_loss = batch_loss / accumulated_size
            batch_loss.backward()
            if self._global_step % accumulated_size == 0:
                self._optimizer.step()
                self._optimizer.zero_grad()

            self._lr_scheduler.step()
            self._global_step += 1





    def test(self, no_tqdm=False, inference_with_sampling=False):
        assert "test" in self._dataset, 'no test set'
        assert not self._dataset["test"].dataset.is_train, "dataloader is not in evaluation mode"
        
        logger.info('==>>> starting prediction on test set...')
        logger.info(f'==>>> there are [{len(self._dataset["test"])}] batches...')
        
        self._model.eval()

        index2insts = {inst['index']:inst for inst in self._dataset['test'].dataset.data}
        results = []
        count = 0
        for batch in (self._dataset["test"] if no_tqdm else tqdm(self._dataset["test"])):
            total_batch_num = len(self._dataset["test"])
            with torch.no_grad():
                batch_outputs = self.inference(
                    self._model,
                    self._tokenizer,
                    batch,
                    do_sample=inference_with_sampling)
                batch_outputs = self.process_outs(self._tokenizer, batch_outputs, batch)
                for output in batch_outputs:
                    index = output['index']
                    sample_dict = index2insts[index]
                    output.update(sample_dict)
                results += batch_outputs
                
                if count % 10==0:
                    logger.info(f'==>>> inference the outputs for batch [{count}]/[{total_batch_num}]')
                count += 1
                #if count>100:
                #    break
        return results




    def add_special_tokens(self, special_token_dict):
      assert self.tokenizer is not None
      self.tokenizer.add_special_tokens(special_token_dict)
      self._model.resize_token_embeddings(len(self.tokenizer))
      self.Print(f'==>>>add special tokens {special_token_dict} to tokenizer.')

