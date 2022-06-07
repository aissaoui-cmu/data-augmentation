# -*- coding: utf-8 -*-
"""paraphrasing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KuhuluSqeACuFNEFOGLJ3FAvK0vko7ep
"""

### CONFIG
!pip install transformers
# mazajak
# !wget http://mazajak.inf.ed.ac.uk:8000/get_sg_250
# !wget http://mazajak.inf.ed.ac.uk:8000/get_cbow_250
# aravec
!wget https://bakrianoo.ewr1.vultrobjects.com/aravec/full_grams_cbow_100_twitter.zip
!wget https://bakrianoo.ewr1.vultrobjects.com/aravec/full_grams_sg_100_twitter.zip
!wget https://bakrianoo.ewr1.vultrobjects.com/aravec/full_grams_cbow_100_wiki.zip
!wget https://bakrianoo.ewr1.vultrobjects.com/aravec/full_grams_sg_100_wiki.zip
!unzip "full_grams_cbow_100_twitter.zip"
!unzip "full_grams_cbow_100_wiki.zip"
!unzip "full_grams_sg_100_twitter.zip"
!unzip "full_grams_sg_100_wiki.zip"

## importing
import json
import requests
from transformers import GPT2LMHeadModel, pipeline, GPT2TokenizerFast
import gensim
from transformers import pipeline
import re
import time

## still did not use it
def pos(token):
  url = 'https://farasa.qcri.org/webapi/pos/'
  payload = {'text': token, 'api_key': "KMxvdPGsKHXQAbRXGL"}
  data = requests.post(url, data=payload)
  result = json.loads(data.text)
  return result

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

def text_generation(model,tokenizer , generation_pipeline ,sentence):
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
    # method 2
    sentence = " ".join(sentence.split()[0:-1])
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
  return l

# text here is a list of sentences
def aug_gpt(model_name,text):
  print("loading GPT...")
  tic = time.perf_counter()
  model , tokenizer , generation_pipeline = load_GPT(model_name)
  toc = time.perf_counter()
  print("loading GPT done: " + str(toc-tic) + " seconds")
  all_sentences = []
  print("augmenting with GPT...")
  tic = time.perf_counter()
  for sentence in text:
    sentence = sentence.strip()
    all_sentences.append([sentence,text_generation(model,tokenizer , generation_pipeline ,sentence)])
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
    word_vectors = model.wv
    for token in sentence.split():
      if is_replacable(token):
        if token in word_vectors.vocab:
          # print("org: " + token)
          most_similar = model.most_similar( token, topn=5 )
          # print(most_similar)
          for term, score in most_similar:
                if term != token:
                    term = term.replace("_"," ")
                    aug = sentence.replace(token,term)
                    # print("aug: " + term)
                    # print(aug + "\n")
                    l.append(org_text.replace(sentence,aug))
  return l

# text here is a list of sentences
def aug_w2v(model_path,text):
  print("loading w2v...")
  tic = time.perf_counter()
  model = load_w2v(model_path)
  toc = time.perf_counter()
  print("loading w2v done: " + str(toc-tic) + " seconds")
  all_sentences = []
  print("augmenting with w2v...")
  tic = time.perf_counter()
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
    for token in sentence.split()[1:]:
        if is_replacable(token):
          masked_text = sentence.replace(token,"[MASK]")
          pred = model(masked_text , top_k = 20)
          for i in pred:
            if type(i) == "<class 'dict'>":
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
  all_sentences = []
  print("augmenting with bert...")
  tic = time.perf_counter()
  for sentence in text:
    sentence = sentence.strip()
    all_sentences.append([sentence,bert(model, sentence)])
  toc = time.perf_counter()
  print("augmenting with bert done: " + str(toc-tic) + " seconds")
  return all_sentences

!wget https://edinburghnlp.inf.ed.ac.uk/workshops/OSACT4/datasets/OSACT2020-sharedTask-dev.txt

def is_replacable(token):
   if token in ["يا","و"]:
     return False
   return True

model = load_w2v("full_grams_cbow_100_twitter.mdl")
w2v(model,"كحل عينك يا منحط يا وضيع يا متخلف @USER URL	OFF	NOT_HS")

## execution cell
def augment(filename):

  with open (filename) as f:
    text = f.read().strip().split("\n")[0:5]
    
  BERT = aug_bert('aubmindlab/bert-large-arabertv2',text)
  W2V = aug_w2v('full_grams_cbow_100_twitter.mdl',text)
  GPT = aug_gpt('aubmindlab/aragpt2-medium',text) # use the mega

  output = ""
  for b,w,g in zip(BERT,W2V,GPT):
    output += "org: " + w[0] + "\n"
    for aug in b[1]:
      output += "bert_aug: " + aug + "\n"
    for aug in w[1]:
      output += "w2v_aug: " + aug + "\n"
    for aug in g[1]:
      output += "gpt_aug: " + aug + "\n"
  
  with open('output', 'w') as f:
       f.write(output)

augment("tweet.txt")
  # # aravec models (4)
  # aravec_sg_300_tw = w2v('full_grams_sg_100_twitter.mdl',text)
  # aravec_cbow_300_tw = w2v('full_grams_cbow_100_twitter.mdl',text)
  # aravec_sg_300_wk = w2v('full_grams_sg_100_wiki.mdl',text)
  # aravec_cbow_300_wk = w2v('full_grams_cbow_100_wiki.mdl',text)

  # # still did not mazajak, because it craches on colab
  # # mazajak models (2)
  # # mazajek_cbow = w2v('get_cbow_250',text)
  # # mazajek_sg = w2v('get_sg_250',text)

  # # BERT-based fill mask models (4)
  # arabert = fill_mask('aubmindlab/bert-base-arabert',text)
  # arabertv2 = fill_mask('aubmindlab/bert-large-arabertv2',text)
  # arabertv02 = fill_mask('aubmindlab/bert-large-arabertv02',text)
  # arabertv01 = fill_mask('aubmindlab/bert-base-arabertv01',text)

  # # GPT2-based text generation models
  # aragpt2 = text_generation('aubmindlab/aragpt2-medium',text) ## use the mega model

  # output = "org: " + text + "\n"

  # gpt = list(set(aragpt2))
  # for aug in gpt:
  #   output += "GPT2-based: " + aug + "\n"

  # bert = list(set(arabert) | set(arabertv2) | set(arabertv02) | set(arabertv01) )
  # for aug in bert:
  #   output += "BERT-based: " + aug + "\n"

  # aravec = list(set(aravec_sg_300_tw) | set(aravec_cbow_300_tw) | set(aravec_sg_300_wk) | set(aravec_cbow_300_wk) )
  # for aug in aravec:
  #   output += "aravec: " + aug + "\n"

  # # mazajak = list(set(mazajek_cbow) | set(mazajek_sg) )
  # # for aug in mazajak:
  # #   output += "mazajak: " + aug + "\n"

  # with open('output', 'w') as f:
  #   f.write(output)

"""# explanation"""

## this is an example
text = "انه موضوع شيق"
aravec_sg_300_tw = w2v('full_grams_sg_100_twitter.mdl',text)
arabertv2 = fill_mask('aubmindlab/bert-large-arabertv2',text)
aragpt2 = text_generation('aubmindlab/aragpt2-medium',text)
print("AraVec results: ")
print(aravec_sg_300_tw)
print("AraBert results: ")
print(arabertv2)
print("AraGPT2 results: ")
print(aragpt2)

"""What can be improved:

* Use more techneiques such us araElctra and wordnet
* make it quicker by reducing the number of models (different versions of the model)

*   Improve the quality of the augmented sentecnes as some of them do not make sense (espicially using bert and gpt) or have different meaning than the original sentence

# Synonym Replacement (EN)
__Explaination of Steps__
- Split a sentence
- Find the synonyms of each word in sentence (1 by one)
- Make a new sentence after replacing only one word
- Do a similarity to the original sentence and remove sentences with similarity less than threshold
"""

!python -m spacy download en_core_web_md 
# After you download spacy model, remember to restart the runtime and then run the rest of the commands.

import nltk
nltk.download("stopwords")
nltk.download("wordnet")

from nltk.corpus import wordnet, stopwords
from random import choice, shuffle

# Functions 
def get_synonyms(word):
  """
  Get synonyms of a word
  """
  synonyms = set()

  for syn in wordnet.synsets(word):
    for l in syn.lemmas():
      synonym = l.name().replace("_", " ").replace("-"," ").lower()
      synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
      synonyms.add(synonym)
  
  if word in synonyms:
    synonyms.remove(word)
  
  return list(synonyms)

def synonym_replacement(words, n):
  """
  Replace synonyms word by word
  """
  words = words.split()

  new_words = words.copy()
  random_word_list = list(set([word for word in words if word not in stopwords.words('english')]))
  shuffle(random_word_list)
  sentences = []
  g_sentences = []

  for random_word in random_word_list:
    synonyms = get_synonyms(random_word)

    if len(synonyms) >= 1:
      for syn_word in list(synonyms):
        new_words_generated = [syn_word if word == random_word else word for word in new_words]
        sentences.append(new_words_generated)
  
  for i in range(len(sentences)):
    final_sentence = ' '.join(sentences[i])
    g_sentences.append(final_sentence)
  
  return g_sentences

# Similarity Checker 
import spacy 
nlp = spacy.load('en_core_web_md') # spacy model

def similarity_checker(text, sentences, spacy_model):
  text_doc = spacy_model(text)
  f_sentences = []

  for i in range(len(sentences)):
    doc = spacy_model(sentences[i])
    if doc.similarity(text_doc) > 0.967: # threshold for similarity
      f_sentences.append(sentences[i])

  return f_sentences

text = "I have a nice home"
words_in_sentence = len(text.split())
syn_sentences = synonym_replacement(text, words_in_sentence) # do synonym replacement
f_sentences = similarity_checker(text, syn_sentences, nlp) # similarity checker for each sentences
print(f_sentences) # print the final sentences