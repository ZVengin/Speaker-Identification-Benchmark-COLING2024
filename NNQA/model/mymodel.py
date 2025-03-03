# CSN module definition

import torch.nn as nn
import torch.nn.functional as functional
import torch
from transformers import AutoModel


def get_nonlinear(nonlinear):
    """
    Activation function.
    """
    nonlinear_dict = {'relu':nn.ReLU(), 'tanh':nn.Tanh(), 'sigmoid':nn.Sigmoid(), 'softmax':nn.Softmax(dim=-1)}
    try:
        return nonlinear_dict[nonlinear]
    except:
        raise ValueError('not a valid nonlinear type!')


class SeqPooling(nn.Module):
    """
    Sequence pooling module.

    Can do max-pooling, mean-pooling and attentive-pooling on a list of sequences of different lengths.
    """
    def __init__(self, pooling_type, hidden_dim):
        super(SeqPooling, self).__init__()
        self.pooling_type = pooling_type
        self.hidden_dim = hidden_dim
        if pooling_type == 'attentive_pooling':
            self.query_vec = nn.parameter.Parameter(torch.randn(hidden_dim))

    def max_pool(self, seq):
        return seq.max(dim=1).values

    def mean_pool(self, seq):
        return seq.mean(dim=1)

    def attn_pool(self, seq):
        attn_score = torch.mm(seq, self.query_vec.view(-1, 1)).view(-1)
        attn_w = nn.Softmax(dim=0)(attn_score)
        weighted_sum = torch.mm(attn_w.view(1, -1), seq).view(-1)     
        return weighted_sum

    def forward(self, batch_seq):
        pooling_fn = {'max_pooling': self.max_pool,
                      'mean_pooling': self.mean_pool,
                      'attentive_pooling': self.attn_pool}
        pooled_seq = pooling_fn[self.pooling_type](batch_seq)
        return pooled_seq


class MLP_Scorer(nn.Module):
    """
    MLP scorer module.

    A perceptron with two layers.
    """
    def __init__(self, args, classifier_input_size):
        super(MLP_Scorer, self).__init__()
        self.scorer = nn.ModuleList()

        self.scorer.append(nn.Linear(classifier_input_size, args.classifier_intermediate_dim))
        self.scorer.append(nn.Linear(args.classifier_intermediate_dim, 1))
        self.nonlinear = get_nonlinear(args.nonlinear_type)
        #self.nonlinear2 = get_nonlinear(args.nonlinear_type)

    def forward(self, x):
        for model in self.scorer:
            x = self.nonlinear(model(x))
        return x





class CSN(nn.Module):
    """
    Candidate Scoring Network.

    It's built on BERT with an MLP and other simple components.
    """
    def __init__(self, args):
        super(CSN, self).__init__()
        self.args = args
        self.bert_model = AutoModel.from_pretrained(args.bert_pretrained_dir)
        self.pooling = SeqPooling(args.pooling_type, self.bert_model.config.hidden_size)
        self.mlp_scorer = MLP_Scorer(args, self.bert_model.config.hidden_size * 3)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, features_batch, sent_char_lens_batch, mention_poses_batch, quote_idxes_batch, true_index_batch, device):
        """
        params
            features: the candidate-specific segments (CSS) converted into the form of BERT input.  
            sent_char_lens: character-level lengths of sentences in CSSs.
                [[character-level length of sentence 1,...] in the CSS of candidate 1,...]
            mention_poses: the positions of the nearest candidate mentions.
                [(sentence-level index of nearest mention in CSS, 
                 character-level index of the leftmost character of nearest mention in CSS, 
                 character-level index of the rightmost character + 1) of candidate 1,...]
            quote_idxes: the sentence-level index of the quotes in CSSs.
                [index of quote in the CSS of candidate 1,...]
            true_index: the index of the true speaker.
            device: gpu/tpu/cpu device.
        """
        # encoding
        
        input_ids_batch = [torch.LongTensor([feature.input_ids  for feature in features]).to(device)
                           for features in features_batch]
        input_mask_batch = [torch.LongTensor([feature.input_mask for feature in features]).to(device)
                            for features in features_batch]

        cdd_ids_batch = torch.LongTensor(sum([[i]*input_ids_batch[i].size(0)
                                              for i in range(len(input_ids_batch))],[])).to(device)
        bert_out = self.bert_model(
            torch.cat(input_ids_batch,dim=0).contiguous(),
            token_type_ids=None,
            attention_mask = torch.cat(input_mask_batch,dim=0).contiguous())

        cdds_accum_char_len_batch = []
        for sent_char_lens in sent_char_lens_batch:
            cdds_accum_char_len = []
            for cdd_sent_char_lens in sent_char_lens:
                accum_char_len = [1]
                for sent_idx in range(len(cdd_sent_char_lens)):
                    accum_char_len.append(accum_char_len[-1] + cdd_sent_char_lens[sent_idx])
                cdds_accum_char_len.append(accum_char_len)
            cdds_accum_char_len_batch.append(cdds_accum_char_len)
        
        CSS_hids = bert_out['last_hidden_state']
        qs_hid_mask_batch = []
        for input_ids,quote_idxes,cdds_accum_char_len in zip(input_ids_batch,quote_idxes_batch,cdds_accum_char_len_batch):
            qs_hid_mask = torch.Tensor([[1 if (
                    cdds_accum_char_len[cdd_id][quote_idxes[cdd_id]] <= token_id <= cdds_accum_char_len[cdd_id][quote_idxes[cdd_id] + 1])
                                         else 0 for token_id in range(input_ids.size(-1))]
                                        for cdd_id in range(input_ids.size(0))]).unsqueeze(-1).contiguous().to(device)
            qs_hid_mask_batch.append(qs_hid_mask)
        qs_hid_mask_batch = torch.cat(qs_hid_mask_batch,dim=0)
        qs_hids = CSS_hids*qs_hid_mask_batch #Batch x MaxSeqLen x HiddDim

        ctx_hid_mask_batch = []

        for input_ids,mention_poses,sent_char_lens,cdds_accum_char_len in zip(
                input_ids_batch,
                mention_poses_batch,
                sent_char_lens_batch,
                cdds_accum_char_len_batch):
            ctx_hid_mask = []
            # "i" is css candidate index, mask the quote in the context
            for i in range(input_ids.size(0)):
                if len(sent_char_lens[i]) == 1:
                    # if candidate only has one sentence, mask all tokens,
                    # due to that this sentence is quote
                    mask = [0]*input_ids.size(-1)
                elif mention_poses[i][0]==0:
                    # if the nearest mention of candidate is in the first sentence, mask the last sentences
                    # due to that the last sentence is quote
                    mask = [1 if token_id < cdds_accum_char_len[i][-2] else 0
                       for token_id in range(input_ids.size(-1))]
                else:
                    # if the nearest mention not in the first sentence, mask the first sentence,
                    # due to that the first sentence is quote in this case
                    mask = [1 if token_id >= cdds_accum_char_len[i][1] else 0
                            for token_id in range(input_ids.size(-1))]
                ctx_hid_mask.append(mask)
            ctx_hid_mask_batch.append(torch.Tensor(ctx_hid_mask).unsqueeze(-1).contiguous().to(device))
        ctx_hid_mask_batch = torch.cat(ctx_hid_mask_batch,dim=0)
        
        ctx_hids = CSS_hids*ctx_hid_mask_batch

        cdd_hid_mask_batch = []
        for mention_poses,input_ids in zip(mention_poses_batch,input_ids_batch):
            cdd_hid_mask = torch.Tensor([[1 if (token_id>=mention_poses[cdd_id][1]+1
                                                      and token_id<mention_poses[cdd_id][2]+1)
                                                else 0 for token_id in range(input_ids.size(-1))]
                                               for cdd_id in range(input_ids.size(0))]).unsqueeze(-1).contiguous().to(device)
            cdd_hid_mask_batch.append(cdd_hid_mask)
        cdd_hid_mask_batch = torch.cat(cdd_hid_mask_batch,dim=0)
        cdd_hids = CSS_hids * cdd_hid_mask_batch
        

        # pooling
        qs_rep = self.pooling(qs_hids) #Batch x HidDim
        ctx_rep = self.pooling(ctx_hids)
        #print(f'ctx_hid:{ctx_hid},cdd_hid:{cdd_hid}')
        cdd_rep = self.pooling(cdd_hids)

        # concatenate
        feature_vector = torch.cat([qs_rep, ctx_rep, cdd_rep], dim=-1)

        # dropout
        feature_vector = self.dropout(feature_vector)
        
        # scoring
        scores_batch = self.mlp_scorer(feature_vector)
        cdd_scores = []
        cdd_scores_false = []
        cdd_scores_true = []
        for k in range(len(input_ids_batch)):
            cdd_ids = (cdd_ids_batch == k ).nonzero(as_tuple=True)
            scores = scores_batch[cdd_ids]
            scores_false = [scores[i] for i in range(scores.size(0)) if i != true_index_batch[k]]
            scores_true = [scores[true_index_batch[k]] for i in range(scores.size(0) - 1)]

            cdd_scores.append(scores)
            cdd_scores_false.append(scores_false)
            cdd_scores_true.append(scores_true)
            #print(f'k:{k}')
            #print(f'cdd_ids_batch:{cdd_ids_batch}')
            #print(f'true_index_batch:{true_index_batch[k]}')
            #print(f'scores:{scores}')
            #print(f'scores_false:{[s.item() for s in scores_false]}')
            #print(f'scores_true:{[s.item() for s in scores_true]}')

        #print(f'scores_false:{scores_false}')
        #print(f'scores_true:{scores_true}')

        return cdd_scores, cdd_scores_false, cdd_scores_true

        

