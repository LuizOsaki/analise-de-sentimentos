import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization

# Carrega o modelo treinado
model = load_model("models/modelo_sentimentos_utf8.keras")

# Carrega o vocabulário salvo durante o treinamento
with open("models/vocab.txt", "r", encoding="utf-8") as f:
    vocab = [linha.strip() for linha in f]

# Recria a camada de vetorização utilizada no treinamento
vectorizer = TextVectorization(
    max_tokens=1000,
    output_mode="int",
    output_sequence_length=100
)

# Define o vocabulário da camada
vectorizer.set_vocabulary(vocab)

# Texto de exemplo para teste do modelo
texto = pd.Series(
    ["Esse produto é maravilhoso"],
    dtype="string"
)

# Converte o texto para representação numérica
texto_vec = vectorizer(texto)

# Faz a predição
resultado = model.predict(texto_vec)

# Obtém a probabilidade prevista
score = resultado[0][0]

print(f"Probabilidade: {score:.2%}")

# Classifica o sentimento
if score > 0.5:
    print("Positivo")
else:
    print("Negativo")