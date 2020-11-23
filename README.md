# Hate-Speech-Identification-in-Tweet-Antisemitism-perspective
This is NLP project that utilizes the word embedding techniques( i.e. bag-of-words, TFIDF and word2vec) to extract antisemitic features from tweet dataset.
The extracted features were used to train classifiers(Random Forest and LSTM)
See pdf for more informtion about the project.

Dataset Labeling
----------------

Label       Description  
0     --->  Indicates tweet tweet is neutral, non offensive and present no hate speech.
1     --->  Indicates tweet is offensive but do not present any hate or a segregative/racist speech
2			---> Indicates tweet is hateful, in other words, it is offensive, and present hate, racist and segregative words or expressions but does not target jews.
3   ---> Indicates tweet is antisemiti, that is, it contains offensive,hate, racist or segregative words/expressions specifically against Jew and Jew community.
