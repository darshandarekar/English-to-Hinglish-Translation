import torch
from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu

# Initialize the model and tokenizer for English to Indian languages
model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


# Define a function to preserve easy English words
def preserve_easy_words(input_text, output_text):
    # Tokenize input and output text
    input_tokens = input_text.lower().split()
    # print(input_tokens)
    output_tokens = output_text.split()
    # print(output_text)

    # Replace easy English words in the output with the input
    for i, token in enumerate(output_tokens):
        if token in dect.keys():
            output_tokens[i] = dect[token]

    # Recreate the output text
    updated_output_text = " ".join(output_tokens)
    return updated_output_text

# Perform inference while preserving easy English words
def perform_inference(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=128, num_beams=4, length_penalty=2.0, early_stopping=True)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    final_text = preserve_easy_words(input_text, output_text)
    # print(final_text)

    return final_text

# Calculate BLEU score
def calculate_bleu(reference: str, translation: str) -> float:
    reference_tokens = reference.split()
    translation_tokens = translation.split()
    return sentence_bleu([reference_tokens], translation_tokens)
# Evaluate the translations using BLEU score
def evaluate_translations(dataset, num_samples=10):
    references = dataset['Hinglish'][:num_samples]
    translations = []

    for source_text in dataset['English'][:num_samples]:
        translation = perform_inference(source_text)
        translations.append(translation)

    for i in range(num_samples):
        bleu = calculate_bleu(references[i], translations[i])
        print(f"Sample {i + 1} BLEU Score: {bleu:.2f}")
# Load your custom dataset (Assuming it has 'source_text' and 'target_text' columns)
custom_dataset = pd.read_csv('your_file.csv')

# Define the source and target languages
source_lang = "en"
target_lang = "hi"

# Define a list of easy English words
easy_english_words = ["like", "minute", "subscribe", "tutorial", "notification","event", "guide", "video",
                      "content","feedback", "camera", "lighting", "collaboration", "social media", "comment",
                      "section","products","bag","mention","video","share"]

dect = {}
for i in easy_english_words:
  input_text = i
  output_text = perform_inference(input_text)
  dect[output_text]=input_text
dect['खण्ड']='section'
dect['बैग']='bag'
dect['प्रतिक्रिया']='feedback'


# Test the translation
# input_text = "Don't forget to turn on the notification bell to stay updated."
input_text = input("Enter your Text : ")

output_text = perform_inference(input_text)
print("Translated Output:", output_text)
#print(calculate_bleu(input_text,input_text))

# Evaluate the translations on the custom dataset
evaluate_translations(custom_dataset)