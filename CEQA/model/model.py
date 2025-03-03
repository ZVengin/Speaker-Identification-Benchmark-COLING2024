import sys

import torch
import os

import logging
import torch.nn as nn
from transformers import AutoModel,AutoTokenizer, BartModel
from transformers.modeling_outputs import BaseModelOutput,Seq2SeqModelOutput
from collections import Counter,defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR,'./../..'))
from utils import CharMaskId,MaxCharNum

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



class CharEncoder(nn.Module):
    def __init__(self,layer_num,hidden_num,head_num):
        super(CharEncoder,self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_num,
                nhead=head_num,
                batch_first=True),
                num_layers = layer_num)
        self.pool = mean_pooling
        self.device = None
        
    def forward(self,embeddings,mask,**kwargs):
        """
        parameters:
            embeddings: the embedding of tokens which has the shape 
                (Batch x CharNum) x MaxMentNum x HiddenNum
            mask: (Batch x CharNum) x MaxMentNum
            output:(Batch x CharNum) x HiddenNum
        """
        #print(f'embeddings:{embeddings.get_device()}, encoder:{next(self.encoder.parameters()).device}')
        enc_outs = self.encoder(embeddings)
        pool_outs = self.pool(enc_outs,mask)
        return pool_outs


class MyBartEncoder(BartModel):
    def __init__(self, config):
        super(MyBartEncoder,self).__init__(config)
        #delattr(self,'decoder')

    def forward(
            self,
            input_ids = None,
            attention_mask = None,
            decoder_input_ids = None,
            decoder_attention_mask = None,
            head_mask = None,
            decoder_head_mask = None,
            cross_attn_head_mask = None,
            encoder_outputs = None,
            past_key_values  = None,
            inputs_embeds = None,
            decoder_inputs_embeds = None,
            use_cache  = None,
            output_attentions  = None,
            output_hidden_states = None,
            return_dict = None,
    ) :

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        if not return_dict:
            return encoder_outputs

        return Seq2SeqModelOutput(
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )




class SpeakerModel(nn.Module):
    def __init__(self,layer_num,model_name,head_num,hidden_num=768,tokenizer=None):
        super(SpeakerModel,self).__init__()
        #self.bart_encoder = AutoModel.from_pretrained(model_name)
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer

        if 'bart' in model_name:
            self.bart_encoder = MyBartEncoder.from_pretrained(model_name)
        else:
            self.bart_encoder = AutoModel.from_pretrained(model_name)
        self.bart_encoder.resize_token_embeddings(len(self.tokenizer))
        self.model_name=model_name
        self.char_encoder = CharEncoder(
            layer_num = layer_num,
            hidden_num = hidden_num,
            head_num = head_num)
        self.device_map = None
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=CharMaskId)
        
    def to_device(self,device_map):
        self.device_map = device_map
        self.bart_encoder = self.bart_encoder.to(device_map['bart_encoder'])
        self.char_encoder = self.char_encoder.to(device_map['char_encoder'])

    def save_pretrained(self,model_dir,is_main_process,save_function,state_dict):
        torch.save(self.state_dict(),os.path.join(model_dir,'best_model.pt'))

        
        
    def forward(self,**kwargs):
        #print(f'index:{kwargs["index"]}')
        #print(f'input_ids:{kwargs["input_ids"].size()}')
        batch_size = kwargs['input_ids'].size(0)
        bart_outs = self.bart_encoder(**{'input_ids':kwargs['input_ids'].to(
            self.device_map['bart_encoder']),
                                         'attention_mask':kwargs['attention_mask'].to(
                                             self.device_map['bart_encoder']
                                         )}
                                      )
        if any([t in self.model_name for t in ['bart','t5']]):
            bart_outs=bart_outs.encoder_last_hidden_state #Batch x SeqLen x HiddenSize
        else:
            bart_outs=bart_outs.last_hidden_state

        bart_outs = bart_outs.to(self.device_map['char_encoder'])
        token_types = kwargs['token_type']
        #get the index of utterances in each instance
        utters_poses = [(kwargs['input_ids'][i]==self.tokenizer.mask_token_id).nonzero(as_tuple=True)
                        for i in range(kwargs['input_ids'].size(0))]

        #select the hidden state of utterances in each instance
        utters_hidds = [bart_outs[i][utters_poses[i]] 
                        for i in range(len(utters_poses))]

        #utterance number in each instance
        utters_nums = [(kwargs['input_ids'][i]==self.tokenizer.mask_token_id).sum()
                     for i in range(kwargs['input_ids'].size(0))]

        #maximum utterance number in each instance
        max_utter_num = max(utters_nums)
        utters_hidds = [torch.cat([utters_hidds[i],
                                   torch.zeros(
                                       max_utter_num-utters_nums[i],
                                       utters_hidds[i].size(-1)
                                       ).to(self.device_map['char_encoder'])])
                                  for i in range(len(utters_hidds))]
        #utters_hidds: Batch x MaxUtterNum x HiddSize
        utters_hidds = torch.stack(utters_hidds)

        #utters_masks: Batch x MaxUtterNum
        utters_masks = torch.Tensor([[1]*utters_nums[i]+[0]*(max_utter_num-utters_nums[i])
                                    for i in range(len(utters_nums))]
                                    ).to(self.device_map['char_encoder'])
        
        
        #compute the frequency of each character being mentioned for each instance
        men_counts = [Counter([men for men in mens if men != CharMaskId])
        for mens in token_types.tolist()]
        #get the maximum frequency being mentioned
        max_mens = [counter.most_common(1)[0][1]  for counter in men_counts if len(counter)>0]
        if len(max_mens)==0:
            max_men=1
        else:
            max_men = max(max_mens)


        #gather the hidden states of mentions for each character and extend the 
        #character number in each instance to MaxCharNum
        #the gathered hidden states: (Batch x MaxCharNum) x MaxMenNum x HiddSize
        mens_hidds,hidds_masks, chars_masks = [],[],[]
        for idx,mc in enumerate(men_counts):
            sorted_mc = sorted(mc.most_common(),key=lambda x:x[0])
            instance_hidds,instance_masks = [],[]
            for men,freq in sorted_mc:
                men_idxs = (token_types[idx]==men).nonzero(as_tuple=True)
                #hiddn states for the mentions of a character
                men_hidds = bart_outs[idx][men_idxs]
                #print(f'men_hidds:{men_hidds.size()}')
                #print(f"zeros:{torch.zeros(max_men-freq,men_hidds.size(-1)).to(self.device_map['char_encoder']).size()}")
                #expand men_hidds(Frequency x HiddSize) -> (MaxMenNum x HiddSize)
                men_hidds = torch.cat([men_hidds,
                                       torch.zeros(max_men-freq,
                                                   men_hidds.size(-1)).to(
                                                       self.device_map['char_encoder'])],dim=0)
                hidds_mask = torch.Tensor([1]*freq+[0]*(max_men-freq)).to(
                    self.device_map['char_encoder'])
                instance_hidds.append(men_hidds)
                instance_masks.append(hidds_mask)
            chars_mask = torch.Tensor([1]*len(instance_masks)
                +[0]*(MaxCharNum-len(instance_masks))).to(
                    self.device_map['char_encoder']
                )
            chars_masks.append(chars_mask)
            #instance_hidds: MaxCharNum x MaxMenNum x HiddSize
            instance_hidds += [torch.zeros(max_men,bart_outs.size(-1)).to(
                self.device_map['char_encoder']) for _ in range(MaxCharNum-len(instance_hidds))]
            instance_masks += [torch.zeros(max_men).to(self.device_map['char_encoder'])
                for _ in range(MaxCharNum-len(instance_masks))]
            
            mens_hidds.extend(instance_hidds)
            hidds_masks.extend(instance_masks)

        #mens_hidds: (Batch x MaxCharNum) x MaxMenNum x HiddSize
        mens_hidds = torch.stack(mens_hidds,dim=0)
        hidds_masks = torch.stack(hidds_masks,dim=0) #(Batch x MaxCharNum)x MaxMenNum
        #chars_masks: Batch x MaxCharNum
        chars_masks = torch.stack(chars_masks)

        #char_outs: (Batch x MaxCharNum) x HiddSize

        char_outs = self.char_encoder(mens_hidds,hidds_masks)
        char_outs = char_outs.view(batch_size,MaxCharNum,-1)

        #probs: Batch x MaxUtterNum x MaxCharNum
        #print(f'utter_hidds:{utters_hidds.size()}')
        probs = torch.matmul(utters_hidds,char_outs.transpose(1,2))
        #print(f'probs:{probs},size:{probs.size()}')
        probs = probs.masked_fill_(mask=(chars_masks==0).unsqueeze(dim=1),value=-10000)
        #print(f'mask_probs:{probs.size()}')
        if 'labels' in kwargs:
            #probs: (Batch x MaxUtterNum) x MaxCharNum
            collapse_probs = probs.view(-1,probs.size(-1))
            #print(f'collapse_probs:{collapse_probs.size()}')
            
            #probs: (Batch x MaxUtterNum)
            labels = kwargs['labels'].view(-1).to(self.device_map['char_encoder'])
            #print(f'labels:{labels.size()}')
            
            loss = self.criterion(collapse_probs,labels)
            logging.info(f'loss:{loss}')
        else:
            loss = None
        return {'logits':probs,'logits_mask':utters_masks,'loss':loss}

