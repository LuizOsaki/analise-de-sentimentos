import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    TextVectorization,
    Embedding,
    GlobalAveragePooling1D,
    Dense
)

# Carregar dataset
df = pd.read_csv("data/sentimentos.csv", encoding="utf-8")


# Separar dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    df["texto"],
    df["sentimento"],
    test_size=0.2,
    random_state=42
)

X_train = X_train.astype("string")
X_test = X_test.astype("string")


# Configuração do TextVectorization
max_tokens = 1000
sequence_length = 100

vectorizer = TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=sequence_length
)

# Adaptar vocabulário ao conjunto de treino
vectorizer.adapt(X_train)

# Salvar vocabulário em UTF-8
vocab = vectorizer.get_vocabulary()

with open("models/vocab.txt", "w", encoding="utf-8") as f:
    for palavra in vocab:
        f.write(palavra + "\n")

# Converter textos para sequências númericas
X_train = vectorizer(X_train)
X_test = vectorizer(X_test)

#Criar modelo sem o TexrtVectorization, pois ele já foi aplicado aos dados
model = Sequential([
    Embedding(
        input_dim=max_tokens,
        output_dim=8
    ),
    GlobalAveragePooling1D(),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compilar modelo
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Treinar modelo
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    validation_split=0.2
)

# Avaliar modelo
loss, accuracy = model.evaluate(X_test, y_test)

print(f"Acurácia: {accuracy:.2f}")

# Teste rápido
texto = pd.Series(
    ["Produto maravilhoso"],
    dtype="string"
)

texto_vec = vectorizer(texto)

resultado = model.predict(texto_vec)

score = resultado[0][0]

print(f"Probabilidade: {score:.2%}")

if score > 0.5:
    print("Positivo")
else:
    print("Negativo")

# Salvar modelo
model.save("models/modelo_sentimentos_utf8.keras")

print("Modelo salvo com sucesso!")
print("Vocabulário salvo em models/vocab.txt")