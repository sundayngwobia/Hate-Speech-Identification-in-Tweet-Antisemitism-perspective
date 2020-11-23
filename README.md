# Hate-Speech-Identification-in-Tweet-Antisemitism-perspective
This is NLP project that utilizes the word embedding techniques( i.e. bag-of-words, TFIDF and word2vec) to extract antisemitic features from tweet dataset.
The extracted features were used to train classifiers(Random Forest and LSTM)
See pdf for more informtion about the project.

Dataset Labeling
----------------  

Label    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Description  
0 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;---> Indicates tweet tweet is neutral, non offensive and present no hate speech.  
1 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;---> Indicates tweet is offensive but do not present any hate or a segregative/racist speech  
2	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;---> Indicates tweet is offensive, and present hate, racist and segregative words but does not target jews.  
3 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;---> Indicates tweet is antisemiti, that is, it contains offensive,hate, racist or segregative words/expressions specifically
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  against Jew and Jew community.  
