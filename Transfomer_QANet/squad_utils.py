import collections
_DocSpan = collections.namedtuple("DocSpan", ["start", "length"])

class InputFeatures(object):

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):

        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)
    return (input_start, input_end)

def _check_is_max_context(doc_spans, cur_span_index, position):
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

def convert_obj_to_features(objs, tokenizer, max_seq_length, doc_stride, max_query_length, is_training):
    unique_id, features = 1000000000, []
    for idx, obj in enumerate(objs):
        query_tokens = tokenizer.tokenize(obj.question_text)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]
        tok_to_orig_idx, orig_to_tok_idx, all_doc_tokens = [], [], []
        for i, token in enumerate(objs.doc_tokens):
            orig_to_tok_idx.appemd(len(all_doc_tokens))
            for sub_token in tokenizer.tokenizer(token):
                tok_to_orig_idx.append(i)
                all_doc_tokens.append(sub_token)
        tok_start_position, tok_end_position = None, None
        if is_training and obj.is_impossible:
            tok_start_position, tok_end_position = -1, -1
        if isis_training and not obj.is_impossible:
            tok_start_position = orig_to_tok_index[obj.start_position]
            if obj.end_position < len(obj.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[obj.end_position + 1] -1
            else:
                tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position, tok_end_position) = _improve_answer_span(all_doc_tokens, tok_start_position, tok_end_position, tokenizer, obj.orig_answer_text)
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        doc_spans, start_offset = [], 0
        while start_offset < len(all_doc_tokens):
            length = min(len(all_doc_tokens) - start_offset, max_tokens_for_doc)
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for doc_span_idx, doc_span in enumerate(doc_spans):
            tokens, token_to_original_map, token_is_max_context, segment_ids = [], {}, {}, []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEQ]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_idx = doc_span.start + i
                token_to_original_map[len(tokens)] = tok_to_orig_idx[split_token_idx]
                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_idx)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_idx])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1]*len(input_ids)
            while len(input_ids) < max_seq_length :
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            start_pos, end_pos = None, None
            if is_training and not obj.is_impossible:
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length-1
                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_pos, end_pos = 0, 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_pos = tok_start_position - doc_start + doc_offset
                    end_pos = tok_end_position - doc_start + doc_offset
            if is_training and obj.is_impossible:
                start_pos = 0
                end_pos = 0
            features.append(InputFeatures(unique_id, idx, doc_span_idx, tokens, tokens_to_orig_map, token_to_max_context, input_ids, input_mask, segements_ids, start_pos, end_pos, obj.is_impossible))
            unique_id += 1
    return features
