### CONFIG
# !pip install transformers
# # mazajak
# # !wget http://mazajak.inf.ed.ac.uk:8000/get_sg_250
# # !wget http://mazajak.inf.ed.ac.uk:8000/get_cbow_250
# # aravec
# !wget https://bakrianoo.ewr1.vultrobjects.com/aravec/full_grams_cbow_100_twitter.zip
# !wget https://bakrianoo.ewr1.vultrobjects.com/aravec/full_grams_sg_100_twitter.zip
# !wget https://bakrianoo.ewr1.vultrobjects.com/aravec/full_grams_cbow_100_wiki.zip
# !wget https://bakrianoo.ewr1.vultrobjects.com/aravec/full_grams_sg_100_wiki.zip
# !unzip "full_grams_cbow_100_twitter.zip"
# !unzip "full_grams_cbow_100_wiki.zip"
# !unzip "full_grams_sg_100_twitter.zip"
# !unzip "full_grams_sg_100_wiki.zip"


## still did not use it
# def pos(token):
#   url = 'https://farasa.qcri.org/webapi/pos/'
#   payload = {'text': token, 'api_key': "KMxvdPGsKHXQAbRXGL"}
#   data = requests.post(url, data=payload)
#   result = json.loads(data.text)
#   return result

## importing
import json
import requests
from transformers import GPT2LMHeadModel, pipeline, GPT2TokenizerFast
import gensim
from transformers import pipeline
import re
import time

def process(text):
  # remove any punctuations in the text
  punc = "،.:!?"
  text = text.strip()
  if text[-1] in punc:
      text = text[0:-1]
  text = text.strip()
  # keep only arabic text
  text = " ".join(re.findall(r'[\u0600-\u06FF]+', text))
  return text

def load_GPT(model_name):
  model = GPT2LMHeadModel.from_pretrained(model_name)
  tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
  generation_pipeline = pipeline("text-generation",model=model,tokenizer=tokenizer)
  return model , tokenizer , generation_pipeline

def GPT(model,tokenizer , generation_pipeline ,sentence):
  org_text = sentence
  sentence = process(sentence)
  l = []
  if len(sentence.split()) < 11:
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    #method 1
    for n in range(1,4):
      for i in range(5):
        pred = generation_pipeline(sentence,
          return_full_text = True,
          pad_token_id=tokenizer.eos_token_id,
          num_beams=10 ,
          max_length=len(input_ids[0]) + n,
          top_p=0.9,
          repetition_penalty = 3.0,
          no_repeat_ngram_size = 3)[0]['generated_text'].replace("."," ").replace("،"," ").replace(":"," ").strip()
        pred = " ".join(pred.split())
        if not pred in l:
          l.append(org_text.replace(sentence,pred))
    ## for now lets just use the first method
    # method 2
    # sentence = " ".join(sentence.split()[0:-1])
    # for n in range(1,4):
    #   for i in range(5):
    #     pred = generation_pipeline(sentence,
    #       return_full_text = True,
    #       pad_token_id=tokenizer.eos_token_id,
    #       num_beams=10 ,
    #       max_length=len(input_ids[0]) + n,
    #       top_p=0.9,
    #       repetition_penalty = 3.0,
    #       no_repeat_ngram_size = 3)[0]['generated_text'].replace("."," ").replace("،"," ").replace(":"," ").strip()
    #     pred = " ".join(pred.split())
    #     if not pred in l:
    #       l.append(org_text.replace(sentence,pred))
  return l

# text here can be list of sentences or on string sentence
def aug_GPT(model_name,text):
  print("loading GPT...")
  tic = time.perf_counter()
  model , tokenizer , generation_pipeline = load_GPT(model_name)
  toc = time.perf_counter()
  print("loading GPT done: " + str(toc-tic) + " seconds")
  print("augmenting with GPT...")
  tic = time.perf_counter()
  if isinstance(text, str):
    ret = GPT(model,tokenizer , generation_pipeline ,text)
    toc = time.perf_counter()
    print("augmenting with GPT done: " + str(toc-tic) + " seconds")
    return ret
  else:
    all_sentences = []
    for sentence in text:
      sentence = sentence.strip()
      all_sentences.append([sentence,GPT(model,tokenizer , generation_pipeline ,sentence)])
    toc = time.perf_counter()
    print("augmenting with GPT done: " + str(toc-tic) + " seconds")
    return all_sentences
  

def load_w2v(model_path):
  try:
      model = gensim.models.KeyedVectors.load_word2vec_format(model_path,binary=True,unicode_errors='ignore')
  except:
      model = gensim.models.Word2Vec.load(model_path)
  return model

## TO DO: do not replace words such as و 
def w2v(model,sentence):
  org_text = sentence
  sentence = process(sentence)
  l = []
  if len(sentence.split()) < 11:
    for token in sentence.split():
      try:
        word_vectors = model.wv
        if token in word_vectors.key_to_index:
           exist = True
        else:
           exist = False
      except:
        if token in model:
          exist = True
        else:
          exist = False
      if is_replacable(token):
        if exist:
          try:
            most_similar = model.wv.most_similar( token, topn=5 )
          except:
            most_similar = model.most_similar( token, topn=5 )
          for term, score in most_similar:
                if term != token:
                    term = term.replace("_"," ")
                    aug = sentence.replace(token,term)
                    l.append(org_text.replace(sentence,aug))
  return l

# text here is a list of sentences
def aug_w2v(model_path,text):
  print("loading w2v...")
  tic = time.perf_counter()
  model = load_w2v(model_path)
  toc = time.perf_counter()
  print("loading w2v done: " + str(toc-tic) + " seconds")
  print("augmenting with w2v...")
  tic = time.perf_counter()
  if isinstance(text, str):
    ret = w2v(model,text)
    toc = time.perf_counter()
    print("augmenting with w2v done: " + str(toc-tic) + " seconds")
    return ret
  else:
    all_sentences = []
    for sentence in text:
       sentence = sentence.strip()
       all_sentences.append([sentence,w2v(model,sentence)])
    toc = time.perf_counter()
    print("augmenting with w2v done: " + str(toc-tic) + " seconds")
    return all_sentences

def load_bert(model):
  model = pipeline('fill-mask', model= model)
  return model

# Contextual word embeddings
def bert(model, sentence):
  org_text = sentence
  sentence = process(sentence)
  l = []
  if len(sentence.split()) < 11:
    for token in sentence.split():
        if is_replacable(token):
          masked_text = sentence.replace(token,"[MASK]")
          pred = model(masked_text , top_k = 20)
          for i in pred:
            if isinstance(i, dict):
              output = i['token_str']
              if not len(output) < 2 and not "+" in output and not "[" in output:
                aug = sentence.replace(token, i['token_str'])
                l.append(org_text.replace(sentence,aug))
  return l

# text here is a list of sentences
def aug_bert(model,text):
  print("loading bert...")
  tic = time.perf_counter()
  model = load_bert(model)
  toc = time.perf_counter()
  print("loading bert done: " + str(toc-tic) + " seconds")
  print("augmenting with bert...")
  tic = time.perf_counter()
  if isinstance(text, str):
    ret = bert(model, text)
    toc = time.perf_counter()
    print("augmenting with bert done: " + str(toc-tic) + " seconds")
    return ret
  else:
    all_sentences = []
    for sentence in text:
      sentence = sentence.strip()
      all_sentences.append([sentence,bert(model, sentence)])
    toc = time.perf_counter()
    print("augmenting with bert done: " + str(toc-tic) + " seconds")
    return all_sentences

def is_replacable(token):
   if token in ["يا","و"]:
     return False
   return True


if __name__ == '__main__':
    # try the mega
    print(aug_bert('aubmindlab/bert-large-arabertv2',"RT @USER: رحمك الله يا صدام يا بطل ومقدام. URL	NOT_OFF	NOT_HS"))
    print("------------------------------------------")
    print(aug_GPT('aubmindlab/aragpt2-medium',"RT @USER: رحمك الله يا صدام يا بطل ومقدام. URL	NOT_OFF	NOT_HS"))
    print("------------------------------------------")
    print(aug_w2v('full_grams_cbow_100_twitter.mdl',"RT @USER: رحمك الله يا صدام يا بطل ومقدام. URL	NOT_OFF	NOT_HS"))
    # model = load_bert('aubmindlab/bert-large-arabertv2')
    # print(bert(model,"RT @USER: رحمك الله يا صدام يا بطل ومقدام. URL	NOT_OFF	NOT_HS"))
    # print("loading mazajak...")
    # tic = time.perf_counter()
    # model = load_w2v('cbow_100.bin')
    # toc = time.perf_counter()
    # print("loading model done: " + str(toc-tic) + " seconds")
    # print("running ...")
    # tec = time.perf_counter()
    # w2v(model,"RT @USER: رحمك الله يا صدام يا بطل ومقدام. URL	NOT_OFF	NOT_HS")
    # toc = time.perf_counter()
    # print("running done: " + str(toc-tec) + " seconds")
    # print("total : " + str(toc-tic) + " seconds")
   
    #globals()[sys.argv[1]](sys.argv[2])
