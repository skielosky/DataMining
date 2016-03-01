# DataMining
Examples and exercise solutions

# Introduction to Information Retrieval. Christopher D. Manning, Raghavan, Schutze

## Chapter 6 - Scoring, term weighting, and the vector space model
Exercise 6.8 Why is the idf of a term always finite?
If we assume that DF(T) ≠ 0 (i.e. that the respective query term T appears at least once, we can give an
upper bound for the IDF (for DF(T) = 1) as well as a lower bound (for DF(T) = N, see below). Because
we are dealing with a fixed, finite document set, there also is a finite set of possible values for the IDF
which are all between these two extreme values.

Exercise 6.9 What is the idf of a term that occurs in every document? Compare this with the use of stop word lists.
It is 0. For a word that occurs in every document, putting it on the stop list has the same effect as idf weighting: the word is ignored.

Exercise 6.10 Consider the table of term frequencies for 3 documents denoted Doc1, Doc2, Doc3 in Figure 6.9. Compute the tf–idf weights for the terms car , auto , insurance , and best , for each document, using the idf values from Figure 6.8 (806,791)

Exercise 6.10 Consider the table of term frequencies for 3 documents denoted Doc1, Doc2, Doc3 in Figure 6.9. Compute the tf–idf weights for the terms car , auto , insurance , and best , for each document, using the idf values from Figure 6.8 (806,791)





