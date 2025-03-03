# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import unicodedata
import six

SPIECE_UNDERLINE = '▁'


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def print_(*args):
    new_args = []
    for arg in args:
        if isinstance(arg, list):
            s = [printable_text(i) for i in arg]
            s = ' '.join(s)
            new_args.append(s)
        else:
            new_args.append(printable_text(arg))
    print(*new_args)


def preprocess_text(inputs, lower=False, remove_space=True, keep_accents=False):
    if remove_space:
        outputs = ' '.join(inputs.strip().split())
    else:
        outputs = inputs
    outputs = outputs.replace("``", '"').replace("''", '"')

    if six.PY2 and isinstance(outputs, str):
        outputs = outputs.decode('utf-8')

    if not keep_accents:
        outputs = unicodedata.normalize('NFKD', outputs)
        outputs = ''.join([c for c in outputs if not unicodedata.combining(c)])
    if lower:
        outputs = outputs.lower()

    return outputs


def encode_pieces(sp_model, text, return_unicode=True, sample=False):
    # return_unicode is used only for py2

    # note(zhiliny): in some systems, sentencepiece only accepts str for py2
    if six.PY2 and isinstance(text, unicode):
        text = text.encode('utf-8')

    if not sample:
        pieces = sp_model.EncodeAsPieces(text)
    else:
        pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
    new_pieces = []
    for piece in pieces:
        if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
            cur_pieces = sp_model.EncodeAsPieces(
                piece[:-1].replace(SPIECE_UNDERLINE, ''))
            if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                if len(cur_pieces[0]) == 1:
                    cur_pieces = cur_pieces[1:]
                else:
                    cur_pieces[0] = cur_pieces[0][1:]
            cur_pieces.append(piece[-1])
            new_pieces.extend(cur_pieces)
        else:
            new_pieces.append(piece)

    # note(zhiliny): convert back to unicode for py2
    if six.PY2 and return_unicode:
        ret_pieces = []
        for piece in new_pieces:
            if isinstance(piece, str):
                piece = piece.decode('utf-8')
            ret_pieces.append(piece)
        new_pieces = ret_pieces

    return new_pieces


def encode_ids(sp_model, text, sample=False):
    pieces = encode_pieces(sp_model, text, return_unicode=False, sample=sample)
    ids = [sp_model.PieceToId(piece) for piece in pieces]
    return ids



def FindNearestMention(text, quote_start,quote_end, aliases, other_aliases):
    mens_poses = []
    other_mens_poses = []
    rule = r'((?<=[^a-zA-Z])|^){}((?=[^a-zA-Z])|$)'
    for alias in other_aliases:
        cur_pos = 0
        try:
            match = re.search(rule.format(alias.lower()),text.lower())
        except Exception as e:
            print('invalid pattern error!')
            continue

        while match != None:
            start_pos,end_pos = match.span()
            start_pos, end_pos = start_pos+cur_pos,end_pos+cur_pos
            other_mens_poses.append(
                {
                    "speaker": alias,
                    "speaker_start": start_pos,
                    "speaker_end": end_pos,
                }
            )
            cur_pos = end_pos
            match = re.search(rule.format(alias.lower()),text[cur_pos:].lower())

    aliases = sorted(aliases, key=lambda x: len(x), reverse=True)
    for alias in aliases:
        cur_pos = 0
        try:
            match = re.search(rule.format(alias.lower()),text.lower())
        except Exception as e:
            print('invalid pattern error!')
            continue
        while match != None:
            # remove the shorter mention if a longer mention exists in current posistion
            start_pos,end_pos = match.span()
            start_pos,end_pos = start_pos+cur_pos,end_pos+cur_pos
            contained = False
            for men_pos in mens_poses + other_mens_poses:
                if (
                    start_pos >= men_pos["speaker_start"]
                    and end_pos <= men_pos["speaker_end"]
                ):
                    contained = True
                    break
            if not contained and (end_pos <= quote_start or start_pos >= quote_end):
                mens_poses.append(
                    {
                        "speaker": alias,
                        "speaker_start": start_pos,
                        "speaker_end": end_pos,
                        "distance": min(abs(start_pos - quote_start),abs(start_pos-quote_end)),
                    }
                )
            cur_pos = end_pos
            match = re.search(rule.format(alias.lower()),text[cur_pos:].lower())
    if len(mens_poses) > 0:
        near_men = min(mens_poses, key=lambda x: x["distance"])
    else:
        near_men = None
    return near_men