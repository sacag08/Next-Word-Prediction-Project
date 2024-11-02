# Next Word Prediction using LSTM and GRU

## Project Overview
This project focuses on developing a next-word prediction model using Recurrent Neural Networks (RNNs), specifically with Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) architectures. The model predicts the most likely next word based on a given text sequence, simulating predictive text applications commonly used in keyboards and chatbots. The model is deployed using Streamlit for real-time predictions.

## Objective
The objective of this project is to build a language model that learns word dependencies and context, providing accurate next-word suggestions. This is achieved by training on a large dataset of text, where the model learns patterns, syntax, and semantics, ultimately enhancing user experience in applications that rely on text generation or completion.

## Key Features
- **Data Preprocessing**: Cleaned and tokenized the text data, transforming it into sequences of words for training. Used padding to handle sequences of varying lengths, creating a balanced dataset.
  
- **Model Development**: Built and trained two neural network models:
  - **LSTM Model**: Implemented an LSTM-based architecture to capture long-term dependencies in text sequences, allowing the model to retain context over longer passages.
  - **GRU Model**: Implemented a GRU-based architecture as an alternative, balancing efficiency and performance by reducing computational overhead while still capturing meaningful word dependencies.
  
- **Training & Optimization**: Experimented with different hyperparameters such as the number of layers, sequence length, batch size, and learning rate to optimize model performance. The models were trained using cross-entropy loss to maximize the accuracy of the next-word predictions.

- **Evaluation**: Assessed model performance using accuracy and loss metrics on both training and validation sets to ensure generalizability and to minimize overfitting.

- **Deployment**: Deployed the final model on Streamlit, enabling users to input sentences and receive real-time next-word predictions in a web-based interface.

## Tools and Technologies
- **Python**: Core programming language for model development.
- **TensorFlow/Keras**: Used for building, training, and evaluating the LSTM and GRU models.
- **NumPy & Pandas**: For data manipulation and preprocessing.
- **NLTK/Spacy**: Utilized for text preprocessing and tokenization.
- **Streamlit**: Deployed the model for real-time, interactive next-word prediction.

## Usage
Clone the repository and install the necessary libraries. Run the Streamlit app to access the next-word prediction model and test predictions with custom text inputs.

## Future Enhancements
- **Transformer-based Model**: Experimenting with transformer models like BERT or GPT for improved context retention.
- **Fine-tuning with Domain-Specific Data**: Adapting the model to specific domains (e.g., medical or legal text) to enhance predictive accuracy in specialized applications.
- **Enhanced User Interface**: Improving the Streamlit app with additional options for text input and visualization of prediction probabilities.

This project demonstrates key skills in natural language processing, RNN architectures, and deep learning, highlighting proficiency in text-based machine learning applications and real-time deployment.
