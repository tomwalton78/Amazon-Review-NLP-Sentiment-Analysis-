# Amazon-Review-NLP-Sentiment-Analysis-

These python files are designed to perform sentiment analysis on amazon reviews, classifying them as positive
or negative using a range of classifiers (E.g. Naive Bayes, Logistic Regression) and then using a voting system
on the results of these classifiers to choose the final classification decision.
An accuracy of ~77 % was achieved using the Tools_and_Home_Improvements dataset and the classifiers selected
in the classifiers_to_use list in nlp_model_v3.py.

Instructions to use:
1. Download datasets from: http://jmcauley.ucsd.edu/data/amazon/
2. Extract and place json file in Datasets folder
3. Run json_to_tsv.py file, entering the file name of the json file in the file_name variable
4. Run nlp_model_v3.py file, entering correct file_name
5. Ensuring light_mode is set to False (in nlp_model_v3.py file) when you need to train the classifiers,
but can set to True when classifiers are pre-trained for the dataset
6. Can then use review_tester.py file to classify manually inputted texts
