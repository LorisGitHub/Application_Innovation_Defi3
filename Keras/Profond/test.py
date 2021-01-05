import gensim, re, sys, pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import numpy as np

word2vec = gensim.models.FastText.load("Models/fasttext_dev.model")
embed_dim = 300 

processed_docs_train = ["De nos histoires d’amour, de nos passions et de nos ruptures, subsistent des souvenirs. Ils forment un récit qu’on s’approprie et qu’on se raconte loin de l’être aimé. Enveloppé dans une comédie romantique teintée de fantastique, Mon Inconnue interroge notre (in)capacité à écrire nos propres histoires. Finalement le dernier film de Hugo Gélin se regarde comme on se remémore nos amours passés.  Roberto Garçon","Oui agréable surprise que ce film pour une fois Benjamin Lavernhe a un vrai rôle il est irrésistible très drôle et charmeur Joséphine Japy est toujours aussi ravissante quant à Françoise Civil j'ai un peu de mal à comprendre l'engouement pour cet acteur ce n'est pas qu'il soit mauvais mais bon il est trop sur surmédiatisé on le voit trop au cinéma . Le film est original , et les second rôles bons également ( mention spéciale à Edith Scob toujours très élégante ) bref un bon moment et comme à la 1ère séance du matin nous n'étions que 3 c'était encore mieux bref film très recommandable","Dans le genre de la comédie romantique française « à l’américaine », Mon Inconnue est sans aucun doute l’une des plus jolies réussites. L’hommage appuyé à Un Jour sans fin donne le ton, celui d’une modestie pleine d’autodérision, qui ne tombe pas pour autant dans les embardées comiques douteuses façon L’Arnacoeur, mais assume le premier degré nécessaire à toute romance qui se respecte. Oui, tout ça est relativement prévisible et formaté, mais l’emballage est soigné et les comédiens sont très bons, alors on se laisse embarquer sans résister."]
tokenizer = Tokenizer(num_words=250, lower=True, char_level=False)
tokenizer.fit_on_texts(processed_docs_train)
#tokenizer.fit_on_texts("De nos histoires d’amour, de nos passions et de nos ruptures, subsistent des souvenirs. Ils forment un récit qu’on s’approprie et qu’on se raconte loin de l’être aimé. Enveloppé dans une comédie romantique teintée de fantastique, Mon Inconnue interroge notre (in)capacité à écrire nos propres histoires. Finalement le dernier film de Hugo Gélin se regarde comme on se remémore nos amours passés.  Roberto Garçon")
word = tokenizer.texts_to_sequences(processed_docs_train)
word_sec_train = sequence.pad_sequences(word, maxlen=150)
word_index = tokenizer.word_index
print(word_index)


# embedding_matrix = np.zeros((150, embed_dim))
# print(embedding_matrix)

# for word, i in word_index.items():
#     print(word, word2vec.wv[word])
#print(word_index)