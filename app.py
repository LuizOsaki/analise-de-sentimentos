import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization

# Título exibido na interface web
st.title("Análise de Sentimentos")

# Configurações utilizadas durante o treinamento.
# Precisam ser as mesmas para que a vetorização funcione corretamente.
MAX_TOKENS = 1000
SEQUENCE_LENGTH = 100

# Carrega o modelo treinado salvo em disco
model = load_model("models/modelo_sentimentos_utf8.keras")

# Carrega o vocabulário gerado durante o treinamento.
# Ele foi separado do modelo para evitar problemas de encoding
# ao fazer deploy no Streamlit Cloud.
with open("models/vocab.txt", "r", encoding="utf-8") as f:
    vocab = [linha.strip() for linha in f]

# Recria a camada de vetorização utilizada no treinamento.
# Ela transforma texto em sequências numéricas.
vectorizer = TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH
)

# Aplica o vocabulário treinado à camada de vetorização
vectorizer.set_vocabulary(vocab)

# Campo onde o usuário digita um comentário
texto = st.text_area("Digite um comentário")

# Executa a análise quando o botão é pressionado
if st.button("Analisar"):

    # Converte o texto digitado para uma sequência numérica
    entrada = vectorizer([texto])

    # Realiza a predição utilizando o modelo treinado
    resultado = model.predict(entrada)

    # Extrai a probabilidade prevista
    score = resultado[0][0]

    # Exibe a probabilidade na tela
    st.write(f"Probabilidade: {score:.2%}")

    # Classificação final baseada no limiar de 50%
    if score > 0.5:
        st.success("Positivo")
    else:
        st.error("Negativo")