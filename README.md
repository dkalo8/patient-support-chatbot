# Clinical Patient-Support Chatbot

An AI-powered clinical chatbot designed to help patients identify potential medical conditions based on their symptoms, recommend relevant drugs using real-world reviews, and provide general medical QA. This was developed as a final project for **CS 6120 - Natural Language Processing**.

## Features

- **Symptom Extraction & Disease Prediction**: Uses `spaCy` to dynamically extract clinical entities (symptoms, age, gender) from human-readable text and predicts potential diseases using a trained Machine Learning model (Random Forest / Decision Tree).
- **Drug Recommendation**: Suggests applicable drugs based on querying condition characteristics. Utilizes TF-IDF and Nearest Neighbors models on the cleaned *Drug Review Dataset*.
- **Generative QA**: Incorporates a fine-tuned GPT-2 instance trained on `COVID-QA` and `MedQuAD` datasets to handle conversational and context-aware medical queries.

---

## Project Structure

This repository is organized logically for easy navigation and contribution:

- **`/data/`**: Stores raw and processed datasets, including:
  - `Cleaned_Drug_Review_Dataset_Train.csv` and `generated_responses_random_forest.csv`
  - `COVID-QA/` and `medquad_data/`
- **`/models/`**: Houses all pre-trained machine learning weights, including:
  - `disease_model.pkl` & `drug_recommendation_model.pkl`
  - `tfidf_vectorizer.pkl`, `label_encoder.pkl`, & `BERT_tokenizer_model.pkl`
  - `gpt2-finetuned/` transformer module
- **`/notebooks/`**: Contains raw exploratory notebooks used for dataset processing and model training (e.g., `Clinical_Chatbot_Project.ipynb` & `Preprocessing_Draft.ipynb`). It also houses the main entry point pipeline (`chatbot.ipynb`).
- **`Final_Report.pdf`**: The detailed NLP final project report.

---

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/dkalo8/patient-support-chatbot.git
   cd patient-support-chatbot
   ```

2. **Install Required Libraries**
   Ensure you have Python 3 installed. Run the following to install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn spacy transformers torch
   ```
   *Note: You also need to download the English core dataset for `spaCy`:*
   ```bash
   python -m spacy download en_core_web_sm
   python -m spacy download en_core_web_lg
   ```

---

## Usage

The quickest way to interact with the chatbot is through the `chatbot.ipynb` notebook located in the `notebooks` repository.

1. Open `notebooks/chatbot.ipynb` into a Jupyter environment.
2. Run the cells to initialize the models.
3. Call the `chat()` function, passing in your text Query. For example:
   ```python
   user_input = "what drugs for hepatitis?"
   result = chat(user_input, context)
   print(result["response"])
   ```

*The chatbot manages a dynamic `context` dictionary per session to track stated symptoms and demographics.*

---

## Documentation

For an in-depth dive into the model architectures, the extraction strategies, and rigorous benchmark results, please refer to the project research paper contained in **`Final_Report.pdf`**.
