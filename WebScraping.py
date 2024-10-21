from bs4 import BeautifulSoup
import requests
import nltk
# nltk.download('all')
from nltk.corpus import stopwords,wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer,WordNetLemmatizer
import random
from collections import Counter
from math import sqrt


# ------------------------Scraping - removing \n - removing digits---------------------------
html_text = requests.get("https://en.wikipedia.org/wiki/Text_mining").text
soupText = BeautifulSoup(html_text, 'lxml').get_text().strip().replace("\n" , "")
soup_without_number = ''.join([w for w in soupText if not w.isdigit()])
# with open('C:/Users/Desktop/text files/1-textMining_text(without_numbers).txt' , 'w' , encoding='utf-8') as f1:
#     f1.write(soup)
# f1.close()



# -----------------------------tokenizing - remove punctuations----------------------------
tokenize = nltk.RegexpTokenizer(r"\w+")
tokenize_words = tokenize.tokenize(soup_without_number)
# with open('C:/Users/Desktop/text files/2-tokenized_text.txt' , 'w' , encoding='utf-8') as f2:
#     f2.write(str(tokenize_words))
# f2.close()



# --------------------------converting to lowercase-----------------------------------
tokenize_words_lowercase = []
for word in tokenize_words:
    tokenize_words_lowercase.append(word.lower())



# -----------------------removing stop words-------------------------------------------
stop_words = set(stopwords.words('english'))
tokenized_withOut_stopWords = []
for word in tokenize_words_lowercase:
    if word not in stop_words and len(word)>1:
        tokenized_withOut_stopWords.append(word)
# with open('C:/Users/Desktop/text files/3-lowercase&removed_stopwords.txt' , 'w' , encoding='utf-8') as f3:
#         f3.write(str(tokenized_withOut_stopWords))
# f3.close()



# -----------------------------token's stemming--------------------------------------
ps = PorterStemmer()
stemmed_words = []
for word in tokenized_withOut_stopWords:
    stemmed_words.append(ps.stem(word))
# with open('C:/Users/Desktop/text files/4-stemmed_words.txt' , 'w' , encoding='utf-8') as f4:
#         f4.write(str(stemmed_words))
# f4.close()
 


# ------------------------------token's pos tagging-----------------------------------
tagged = nltk.pos_tag(tokenized_withOut_stopWords)
# with open('C:/Users/Desktop/text files/5-pos_tagged_words.txt' , 'w' , encoding='utf-8') as f5:
#         f5.write(str(tagged))
# f5.close()



# -----------------------------finding verbs from tokens--------------------------------
pos_verbs = [(word, tag) for word, tag in tagged if ("VB" in tag)]
# with open('C:/Users/Desktop/text files/6-verbs.txt' , 'w' , encoding='utf-8') as f6:
#         f6.write(str(pos_verbs))
# f6.close()



# ---------------------------------verb's lemmatizing----------------------------------
lemmatizer = WordNetLemmatizer()
lemmatized_verbs = []
for verb in pos_verbs:
    lemmatized_verbs.append(lemmatizer.lemmatize(verb[0],wordnet.VERB)) 
# with open('C:/Users/Desktop/text files/7-lemmatized_verbs.txt' , 'w' , encoding='utf-8') as f7:
#         f7.write(str(lemmatized_verbs))
# f7.close()




# ----------------------------------verb synonyms------------------------------
verb_synonyms = []
for i in range(0,100):
    for syn in wordnet.synsets(lemmatized_verbs[i]):
        verb_synonyms.append(f'({lemmatized_verbs[i]})  {syn.name()}: lemma-names{syn.lemma_names()} <<definition:{syn.definition()}>>')
    verb_synonyms.append("\n")
# with open('C:/Users/Desktop/text files/8-verbs_synonyms.txt' , 'w' , encoding='utf-8') as f8:
#     for i in verb_synonyms:
#         if ( i == "\n"):
#             f8.write("\n")
#         else: 
#             f8.write(i)
#             f8.write("\n")   
# f8.close()




# --------------------------verb hyponyms & hypernyms----------------------------
verb_hyponyms = []
hypo = []
verb_hypernyms = []
hyper = []
for i in range(0,100):
    syn_set = wordnet.synsets(lemmatized_verbs[i])
    if (syn_set):
        for syn in syn_set:
            syn_name = syn.name()
            synVerb_synset = wordnet.synset(syn_name)
            hyponyms = list(set([w for s in synVerb_synset.closure(lambda s:s.hyponyms()) for w in s.lemma_names()]))
            hypo.extend(hyponyms)
            hypernyms = list(set([w for s in synVerb_synset.closure(lambda s:s.hypernyms()) for w in s.lemma_names()]))
            hyper.extend(hypernyms)
            
        verb_hyponyms.append(f'{i}> word:({lemmatized_verbs[i]}) --> hyponyms:{hypo}')
        verb_hypernyms.append(f'{i}> Word:({lemmatized_verbs[i]}) --> hypernyms:{hyper}')
# with open('C:/Users/Desktop/text files/9-verbs_hyponyms.txt' , 'w' , encoding='utf-8') as f9:
#     for i in verb_hyponyms:
#         f9.write(i)
#         f9.write("\n\n")
# f9.close()
# with open('C:/Users/Desktop/text files/10-verbs_hypernyms.txt' , 'w' , encoding='utf-8') as f10:
#     for i in verb_hypernyms:
#         f10.write(i)
#         f10.write("\n\n")
# f10.close()




# ----------------------------verb similarity by distance--------------------------------
two_verb_similarity_byDistance = []
for i in range(0,100):
    w1 = random.choice(lemmatized_verbs)
    while((not wordnet.synsets(w1))):
        w1 = random.choice(lemmatized_verbs)
           
    w2 = random.choice(lemmatized_verbs)
    while((not wordnet.synsets(w2))):
        w2 = random.choice(lemmatized_verbs) 

    max_wup_similarity=0
    max_path_similarity=0
    for syn_w1 in wordnet.synsets(w1):
        for syn_w2 in wordnet.synsets(w2):
            wup_sim = syn_w1.wup_similarity(syn_w2)
            path_sim = syn_w1.path_similarity(syn_w2)
            if(wup_sim > max_wup_similarity):
                max_wup_similarity = wup_sim
            if(path_sim > max_path_similarity):
                max_path_similarity = path_sim

    two_verb_similarity_byDistance.append(f'{i}> w1:({w1}),w2:({w2}) --> wup-similarity:{max_wup_similarity}, path-similarity:{max_path_similarity}')
# with open('C:/Users/Desktop/text files/11-verbs_distance_similarity.txt' , 'w' , encoding='utf-8') as f11:
#     for i in two_verb_similarity_byDistance:
#         f11.write(i)
#         f11.write("\n\n")
# f11.close()





# ---------------------------words cosine similarity----------------------------------
words_cosine_sim = []
for i in range(0,100):
    w1 = random.choice(stemmed_words)
    w2 = random.choice(stemmed_words)

    numOf_words_chars1 = Counter(w1)
    set_words_chars1 = set(numOf_words_chars1)
    len_word_vector1 = sqrt(sum(char*char for char in numOf_words_chars1.values()))
    first_word_vector = (numOf_words_chars1,set_words_chars1,len_word_vector1)

    numOf_words_chars2 = Counter(w2)
    set_words_chars2 = set(numOf_words_chars2)
    len_word_vector2 = sqrt(sum(char*char for char in numOf_words_chars2.values()))
    second_word_vector = (numOf_words_chars2,set_words_chars2,len_word_vector2)

    words_common_chars = first_word_vector[1].intersection(second_word_vector[1])
    cosine_similarity = sum(first_word_vector[0][char]*second_word_vector[0][char] for char in words_common_chars)/(first_word_vector[2]*second_word_vector[2])
    words_cosine_sim.append(f'{i}> w1:({w1}),w2:({w2}) --> cosine-similarity:{cosine_similarity}')
# with open('C:/Users/Desktop/text files/12-cosine_similarity_betweenWords.txt' , 'w' , encoding='utf-8') as f12:
#     for i in words_cosine_sim:
#         f12.write(i)
#         f12.write("\n\n")
# f12.close()


