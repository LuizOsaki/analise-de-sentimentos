Sentiment Analysis with TensorFlow, Keras, and Streamlit

A Natural Language Processing (NLP) project developed in Python using TensorFlow and Keras to classify text as positive or negative.

The project also includes a web interface built with Streamlit, allowing users to test predictions in real time.

Objective

Train a neural network capable of identifying the sentiment of a sentence based on previously labeled examples.

Examples:

Text	Sentiment
I really liked the product	Positive
Excellent customer service	Positive
Bad product	Negative
I don't recommend it	Negative
Technologies Used
Python
Pandas
NumPy
Scikit-Learn
TensorFlow
Keras
Streamlit
Jupyter Notebook
Project Structure
ANALISE-DE-SENTIMENTOS
│
├── data
│   └── sentimentos.csv
│
├── models
│   └── modelo_sentimentos.keras
│
├── notebooks
│   └── analise.ipynb
│
├── src
│   ├── train.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md

Getting Started
1. Clone the repository
git clone <REPOSITORY_URL>
cd ANALISE-DE-SENTIMENTOS

2. Create a virtual environment
python -m venv venv

Windows
venv\Scripts\activate

Linux/macOS
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

Training the Model
python src/train.py


The training process will:

Load the dataset
Split the data into training and test sets
Vectorize the text
Train the neural network
Evaluate its accuracy
Save the trained model to:
models/modelo_sentimentos.keras

Making Predictions from the Terminal
python src/predict.py


Example output:

Probability: 87.31%
Positive

Web Interface with Streamlit

To start the web application:

streamlit run app.py


The interface allows users to enter text and instantly receive the model's prediction, including:

Calculated probability
Positive or negative classification
Real-time results
Neural Network Architecture

The model consists of the following layers:

TextVectorization
Embedding
GlobalAveragePooling1D
Dense (ReLU)
Dense (Sigmoid)

Pipeline:

Text
 ↓
TextVectorization
 ↓
Embedding
 ↓
GlobalAveragePooling1D
 ↓
Dense (ReLU)
 ↓
Dense (Sigmoid)
 ↓
Positive sentiment probability

Concepts Applied
Natural Language Processing (NLP)
Text Vectorization
Word Embeddings
Neural Networks
Binary Classification
Binary Crossentropy
Model Training and Inference
Model Persistence (.keras)
Application Deployment with Streamlit
Possible Future Improvements
Use a larger and more diverse dataset
Support multi-class sentiment classification
Experiment with more advanced models such as LSTM, GRU, or Transformers
Integrate with external APIs
