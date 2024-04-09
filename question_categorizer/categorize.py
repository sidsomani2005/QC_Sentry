import os
import warnings
from transformers import pipeline
import pandas as pd




def categorizer(question_txt, question_types):
    """
    Uses zero-shot-classification as the LLM model from HuggingFace
        => "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli" is the pre-trained zero-shot-classification model that we are using to categorize questions
    
    Returns the category with the largest score based on the results of the model

    Args:
    - question_txt (str): the question in string format
    - question_types (list(str)): list of strings of the types/categories of questions

    Returns:
    - max_score_category: string with the category of the maximum score based on the results of the model and the given inputs
    """

    # Create instance of zero-shot-classification classifier with provided pre-trained model
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    
    #Ignore warnings
    warnings.simplefilter("ignore")

    # Create instance of the output of the classifier's evaluation of the provided text and candidate_labels
    output = classifier(question_txt, candidate_labels=question_types, truncation=True, multi_label=False)
    
    # Reset warnings
    warnings.resetwarnings()

    # Find the index of the candidate_label with the maximum score based on the provided text
    max_score_index = output["scores"].index(max(output["scores"]))

    # Retrieve that index's value from the provided list of candidate_labels
    max_score_category = output["labels"][max_score_index]

    # Print the identified max score category/label
    #print("Category with the maximum score:", max_score_category)
    
    # Retun the label or category with the maximum score
    return max_score_category, str(max(output["scores"]))




def categorize_question(filepath, question_types):
    """
    Returns the category with the largest score based on the results of the model

    Args:
    - filepath (str): path of the csv file
    - question_types (list(str)): list of strings of the types/categories of questions
    """

    # Read csv file into dataframe
    df = pd.read_csv(filepath)

    for i, row in df.iterrows():
        print(categorizer(row[1], question_types))
    



topic_names = ['Anatomy of the Digestive system', 'Mechanism of Digestion of food', 'Respiratory Organs Anatomy', 'Mechanism of Breathing and Exchange of gases', 'Transportation of gases', 'Regulation of Respiration', 'Respiratory disorders', 'Types of chemical reactions', 'Order Of Reaction', 'Influence of Temperature on Reaction Rates']

categorize_question("/Users/sidsomani/Desktop/question_categorizer/Que_Test_Data_040124.csv", topic_names)









