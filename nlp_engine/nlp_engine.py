import spacy
import neuralcoref
from transformers import *
nlp = spacy.load("en")
neuralcoref.add_to_pipe(nlp)
        
class nlp_engine:
    def __init__(self,):
        self.tokenizer   = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model       = BertModel.from_pretrained("bert-base-uncased")
        self.use_coref = False

    def __call__(self, context):
        context_doc = nlp(context)
        sentences_doc = [(x.text, x.start_char) for x in context_doc.sents]
        self.use_coref = context_doc._.has_coref
        print(self.use_coref)
        sentence_lengths = [len(sentences_doc[0][0])] 
        sentence_lengths += [sentence_lengths[i-1] + len(sentences_doc[i][0]) for i in range(1, len(sentences))]
        ents = context_doc.ents
        sent_idx = list()
        #use ner
        print(ents)
        for ent in ents:
            for i in range(sentence_lengths[i]):
                print(sentence)
                print(ent.start_char)
                if(ent.start_char < sentence_lengths[i]):
                    sent_idx[ent.text] = sentence[i]             
                    break



        
    def find_word_in_ctx(self, x, sentences, sentence_lengths):
        sent = ''
        for i in range(len(sentence_lengths)):
            if(x.start_char < sentence_lengths[i]):
                return sentences[i]
            else:
                sent = sentences[0]
        return sent

    def _get_answer_spans(self, ents, sentences, sentence_lengths):          
        entities, entity_dict = [], {}       
        for x in ents:      
            if x.text in entity_dict:
                continue
            entity_dict[x.text] = 1
            sent = self.find_word_in_ctx(x, setences, sentence_lengths)
            entities.append((x.text, sent))
        return entities

    def _neural_get_answer_spans(self, para_text, _processed_spans=[]):    
        if _processed_spans == []:
            _processed_spans = _get_answer_spans(para_text)
        context_doc = self.nlp(para_text)
        sentences = [str(sents) for sents in context_doc.sents]
        sent_lengths = [len(sentences[0])]
        for i in range(1, len(sentences)):
            sent_lengths.append(len(sentences[i]) + sent_lengths[i-1])
        spans = list()
        for span, start, end in _processed_spans:       
                for i in range(len(sent_lengths)):
                    if(start < sent_lengths[i]):
                        sent = sentences[i]
                        break
                    else:
                        sent = sentences[0]               
                spans.append(span.lower())
                input_ids = self.tokenizer.encode("[CLS]") + self.tokenizer.encode(sent) + self.tokenizer.encode("[SEP]")
                token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
                start_scores, end_scores = self.model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
                all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
                bert_span = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]).lower()
                if(bert_span not in spans and bert_span != ''):
                    spans.append((bert_span, sent))     
        return spans


if(__name__=="__main__"):
    eng = nlp_engine()
    ctx = "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries."
    eng(ctx)
