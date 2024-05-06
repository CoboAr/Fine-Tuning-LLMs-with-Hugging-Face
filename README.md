# Fine-Tuning-LLMs-with-Hugging-Face

This repository contains code for fine-tuning large language models (LLMs) using the Hugging Face library. 
LLMs are pre-trained models that have been trained on vast amounts of text data and can be fine-tuned on specific tasks to achieve state-of-the-art performance.

## Overview 
This project provides a step-by-step guide to fine-tuning an LLM using Hugging Face's Transformers library. 
The process involves installing and importing the necessary libraries, loading the pre-trained model and tokenizer, setting training arguments, training the model, and interacting with the trained model for text generation tasks.   

## Usage
<ol>
  <li>Installing and Importing Libraries: Install the required libraries and import them into your Python environment.</li>
  <li>Loading the Model: Load the pre-trained model using AutoModelForCausalLM.from_pretrained().</li>
  <li>Loading the Tokenizer: Load the tokenizer corresponding to the pre-trained model using AutoTokenizer.from_pretrained().</li>
  <li>Setting the Training Arguments: Define the training arguments such as output directory, batch size, and maximum number of steps.</li>
  <li>Creating the Supervised Fine-Tuning Trainer: Create the trainer object using SFTTrainer from TRL library, providing the model, training arguments, training dataset, tokenizer, and any additional configuration.</li>
  <li>Training the Model: Start the training process by calling the train() method on the trainer object.</li>
  <li>Chatting with the Model: Interact with the trained model by providing user prompts and generating text responses.</li>
</ol>

## Dataset
The training dataset used in this project is the "aboonaji/wiki_medical_terms_llam2_format". Please ensure that you have access to this dataset or replace it with your desired dataset.

## Demo
<img width="1099" alt="Screenshot 2024-05-06 at 1 22 37 AM" src="https://github.com/CoboAr/Fine-Tuning-LLMs-with-Hugging-Face/assets/144629565/283be264-8644-435e-a416-b8a98d46d49f">




## Model and dataset links
[model used](https://huggingface.co/aboonaji/llama2finetune-v2)       
[original dataset](https://huggingface.co/datasets/gamino/wiki_medical_terms)      
[formatted dataset](https://huggingface.co/datasets/aboonaji/wiki_medical_terms_llam2_format?row=0)       

Enjoy! And please do let me know if you have any comments, constructive criticism, and/or bug reports.
## Author
## Arnold Cobo
