import gradio as gr
import json
import pandas as pd
from fastai.learner import load_learner

# Load the question-to-answer mapping from a CSV or JSON file
df = pd.read_csv('project_answer_dataset (3).csv')  # Ensure this file contains columns: 'label', 'answer'

# Load the labels
with open('question_labels.json', 'r') as f:
    question_dictionary = json.load(f)
que_classes = list(question_dictionary.keys())

# Load the model
blurr_model = load_learner('Project_v2.pkl')

def get_answer_for_label(label):
    """
    Given a label, returns the corresponding answer from the dataframe.
    """
    try:
        answer = df[df['label'] == label]['answer'].values[0]
    except IndexError:
        answer = "Answer not found for label: {}".format(label)
    return answer

def detect_question_with_answer(text):
    """
    Given an input text, predicts the label and retrieves the corresponding answer.
    """
    # Perform prediction
    prediction = blurr_model.blurr_predict(text)
    pred_info = prediction[0]  # Assuming prediction is a list with a dictionary as its first element
    
    # Extract label information
    label = pred_info['label']
    score = pred_info['score']
    class_index = pred_info['class_index']
    class_labels = pred_info['class_labels']
    
    # Get the predicted class label
    predicted_class_label = class_labels[class_index]
    
    # Get the corresponding answer for the label
    answer = get_answer_for_label(predicted_class_label)
    
    # Combine results for display
    return label, f"{score:.2f}", class_index, predicted_class_label, answer

# Create a Gradio interface
iface = gr.Interface(
    fn=detect_question_with_answer, 
    inputs="text", 
    outputs=[
        gr.Textbox(label="Predicted Label"),
        gr.Textbox(label="Prediction Score"),
        gr.Textbox(label="Predicted Class Index"),
        gr.Textbox(label="Predicted Class Label"),
        gr.Textbox(label="Answer")
    ]
)

iface.launch(inline=False)
				