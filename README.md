# project title:
   🧠 Text Classification using RNN, LSTM, and GRU


# 📖 Overview:
  Text classification is a core task in NLP where text is categorized into predefined labels. This project explores three different recurrent neural architectures:

  RNN (Recurrent Neural Network)

  LSTM (Long Short-Term Memory)

  GRU (Gated Recurrent Unit)

  Each model is trained and evaluated for performance to compare accuracy and generalization on unseen data.

## 🛠 Tech Stack
    Python 3.8+
    TensorFlow / Keras
    NumPy
    Matplotlib / Seaborn (for visualization)
    Pandas
    Scikit-learn
    NLTK or spaCy (for preprocessing)

# 🧹 Text Preprocessing:
     Lowercasing
     Removing punctuation & stopwords
     Tokenization
     Text to sequences using Tokenizer
     Padding sequences


# Padd_seq_Example:
   tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
   tokenizer.fit_on_texts(X_train)
  sequences = tokenizer.texts_to_sequences(X_train)
  padded = pad_sequences(sequences, maxlen=100, truncating='post')


# 🧠 Model Architectures:

  # 1️⃣ RNN Model:


     maxlen=maxlen
     
     model=Sequential([
    Input(shape=(maxlen,)),
    
    Embedding(input_dim=10000,output_dim=64,input_length=maxlen),
    
    SimpleRNN(150,return_sequences=True),
    Dropout(0.3),
   
    SimpleRNN(150,return_sequences=False),
    Dropout(0.3),
    
    
    
    
    Dense(256,activation="relu",kernel_regularizer=tensorflow.keras.regularizers.l2(0.03)),
    Dropout(0.3),
    BatchNormalization(),
    
    
    
    
    Dense(1,activation='sigmoid')
    
    
    
])


# ✅ Results:
  
   Model	 Accuracy
   RNN	    95.3%
   LSTM	  97.7%
   GRU    	86.4%
 

# 🚀 Future Improvements:

    Experiment with pre-trained embeddings (GloVe, FastText)

    Add attention mechanisms

    Deploy via Flask or Streamlit


    


