import json,os,sys,wandb
import random

from tqdm import tqdm
import collections
import tokenizations.official_tokenization as tokenization
import os
import numpy as np
from .prepro_utils import *
import gc

SCRIP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIP_DIR,'./../..'))
print(list(os.listdir(os.path.join(SCRIP_DIR,'./../..'))))
from utils import ConstructSingleQuoteInstance
SPIECE_UNDERLINE = '▁'



def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def json2features(input_file, output_files, tokenizer, is_training=False, repeat_limit=3, max_query_length=64,
                  max_seq_length=500, doc_stride=128):
    with open(input_file, 'r', encoding='utf8') as f:
        train_data = json.load(f)



    def _is_chinese_char(cp):
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def is_fuhao(c):
        if c == '。' or c == '，' or c == '！' or c == '？' or c == '；' or c == '、' or c == '：' or c == '（' or c == '）' \
                or c == '－' or c == '~' or c == '「' or c == '《' or c == '》' or c == ',' or c == '」' or c == '"' or c == '“' or c == '”' \
                or c == '$' or c == '『' or c == '』' or c == '—' or c == ';' or c == '。' or c == '(' or c == ')' or c == '-' or c == '～' or c == '。' \
                or c == '‘' or c == '’' or c=="." or c=="'" or c=="!" or c== "?" or c==":" or c=="`":
            return True
        return False



    def _tokenize_chinese_chars(text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if _is_chinese_char(cp) or is_fuhao(char):
                if len(output) > 0 and output[-1] != SPIECE_UNDERLINE:
                    output.append(SPIECE_UNDERLINE)
                output.append(char)
                output.append(SPIECE_UNDERLINE)
            else:
                output.append(char)
        return "".join(output)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c == SPIECE_UNDERLINE:
            return True
        return False

    # Construct instances with single quote
    # each instance {'preceding_paragraphs':[],'succeeeding_paragraphs':[],'dialogue':[],'character':{},'id':origin_id-utter_id}
    examples = []
    omit_quote = 0
    single_quote_insts = ConstructSingleQuoteInstance(data=train_data,tokenizer=tokenizer)
    for inst in tqdm(single_quote_insts,desc='construct inputs'):
        context = '\n'.join(map(lambda x:x['paragraph'],inst['preceding_paragraphs']+inst['dialogue']+inst['succeeding_paragraphs']))
        context_chs = _tokenize_chinese_chars(context)
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in context_chs:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            if c != SPIECE_UNDERLINE:
                char_to_word_offset.append(len(doc_tokens) - 1)

        for quote_index in range(len(inst['dialogue'][0]['utterance'])):
            qid = inst['dialogue'][0]['utterance'][quote_index]['quote_id']
            ques_text = inst['dialogue'][0]['utterance'][quote_index]['quote']
            ans_text = inst['dialogue'][0]['utterance'][quote_index]['speaker']

            ques_start = context.find(ques_text)
            if ques_start == -1:
                omit_quote += 1
                continue
            ques_end = ques_start + len(ques_text)
            char_dict = inst['character']
            aliases = char_dict['id2names'][str(char_dict['name2id'][ans_text])]
            irrelevant_aliases = [name for name in char_dict['name2id'].keys() if name not in aliases]
            near_mention = FindNearestMention(context,ques_start,ques_end,aliases,irrelevant_aliases)

            if near_mention is None:
                omit_quote += 1
                continue
            ans_text = near_mention['speaker']
            ans_start = near_mention['speaker_start']

            start_position_final = None
            end_position_final = None
            if True:#is_training:
                count_i = 0
                start_position = ans_start

                end_position = start_position + len(ans_text) - 1
                while context[start_position:end_position + 1].lower() != ans_text.lower() and count_i < repeat_limit:
                    start_position -= 1
                    end_position -= 1
                    count_i += 1

                while context[start_position] == " " or context[start_position] == "\t" or \
                        context[start_position] == "\r" or context[start_position] == "\n":
                    start_position += 1

                start_position_final = char_to_word_offset[start_position]
                end_position_final = char_to_word_offset[end_position]

                if doc_tokens[start_position_final] in {"。", "，", "：", ":", ".", ","}:
                    start_position_final += 1

                actual_text = "".join(doc_tokens[start_position_final:(end_position_final+1)])
                cleaned_answer_text = "".join(tokenization.whitespace_tokenize(ans_text))

                if actual_text != cleaned_answer_text:
                    tqdm.write(f'answer_text:{inst["dialogue"][0]["utterance"][quote_index]["speaker"]}, '
                               f'actual text:{actual_text} , '
                               f'cleaned_answer_text:{cleaned_answer_text}'
                               f'context:{context[ans_start:ans_start+len(ans_text)]}')
                    # ipdb.set_trace()

            examples.append({'doc_tokens': doc_tokens,
                             'context':context,
                             'orig_answer_text': ans_text,
                             'qid': qid,
                             'question': ques_text,
                             'answer': ans_text,
                             'start_position': start_position_final,
                             'end_position': end_position_final,
                             'character':inst['character']})

    tqdm.write(f'examples num: {len(examples)}')
    tqdm.write(f'omit quote: {omit_quote}')
    os.makedirs('/'.join(output_files[0].split('/')[0:-1]), exist_ok=True)
    json.dump(examples, open(output_files[0], 'w'))

    # to features
    features = []
    unique_id = 1000000000
    count_outspan = 0
    for (example_index, example) in enumerate(tqdm(examples)):
        query_tokens = tokenizer.tokenize(example['question'])
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example['doc_tokens']):
            orig_to_tok_index.append(len(all_doc_tokens))
            #the ith doc token to the start index j of its sub tokens in the doc sub token array
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                # the jth sub token to the index i of its original token in the token array
                all_doc_tokens.append(sub_token)

        tok_start_position = None #the start index of answer text tokens in the sub tokens array
        tok_end_position = None  #the end index of answer text tokens in the sub tokens array
        if True:#is_training:
            if example['start_position'] < len(example['doc_tokens']) - 1:
                tok_start_position = orig_to_tok_index[example['start_position']]  # 原来token到新token的映射，这是新token的起点
            else:
                tok_start_position = len(all_doc_tokens) - 1
            if example['end_position'] < len(example['doc_tokens']) - 1:
                tok_end_position = orig_to_tok_index[example['end_position'] + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example['orig_answer_text'])

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # if the original doc is too long, then cut it into many snippets with its length being less than 512-len(query)
        # the next snippet is obtained by shiftting the start index of snippet by 128 tokens

        doc_spans = []
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            # recompute the

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if True:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                if tok_start_position == -1 and tok_end_position == -1:
                    start_position = 0  # 问题本来没答案，0是[CLS]的位子
                    end_position = 0
                else:  # 如果原本是有答案的，那么去除没有答案的feature
                    out_of_span = False
                    doc_start = doc_span.start  # 映射回原文的起点和终点
                    doc_end = doc_span.start + doc_span.length - 1

                    if not (tok_start_position >= doc_start and tok_end_position <= doc_end):  # 该划窗没答案作为无答案增强
                        out_of_span = True
                        count_outspan +=1
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset

            features.append({'unique_id': unique_id,
                             'example_index': example['qid'],
                             'question':example['question'],
                             'doc_span_index': doc_span_index,
                             'tokens': tokens,
                             'token_to_orig_map': token_to_orig_map,
                             'token_is_max_context': token_is_max_context,
                             'input_ids': input_ids,
                             'input_mask': input_mask,
                             'segment_ids': segment_ids,
                             'start_position': start_position,
                             'end_position': end_position,
                             'character':example['character']})
            unique_id += 1

    tqdm.write(f'features num: {len(features)}')
    tqdm.write(f'out of span instances: {count_outspan}')
    json.dump(features, open(output_files[1], 'w'))

    if is_training:
        data_table = wandb.Table(columns=['unique_id','example_index','truncated_span','quote','truncated_answer'])
        for feat in random.sample(features,k=20):
            data_table.add_data(
                feat['unique_id'],
                feat['example_index'],
                ' '.join(feat['tokens']),
                feat['question'],
                ' '.join(feat['tokens'][feat['start_position']:feat['end_position'] + 1]))
        wandb.log({'feature':data_table})



def _convert_index(index, pos, M=None, is_start=True):
    if pos >= len(index):
        pos = len(index) - 1
    if index[pos] is not None:
        return index[pos]
    N = len(index)
    rear = pos
    while rear < N - 1 and index[rear] is None:
        rear += 1
    front = pos
    while front > 0 and index[front] is None:
        front -= 1
    assert index[front] is not None or index[rear] is not None
    if index[front] is None:
        if index[rear] >= 1:
            if is_start:
                return 0
            else:
                return index[rear] - 1
        return index[rear]
    if index[rear] is None:
        if M is not None and index[front] < M - 1:
            if is_start:
                return index[front] + 1
            else:
                return M - 1
        return index[front]
    if is_start:
        if index[rear] > index[front] + 1:
            return index[front] + 1
        else:
            return index[rear]
    else:
        if index[rear] > index[front] + 1:
            return index[rear] - 1
        else:
            return index[front]


