import streamlit as st
import docx
import PyPDF2
# from fb2reader import FB2Reader
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import re
import requests
import json
import time
from dotenv import load_dotenv
import os
from lxml import etree

# Загружаем переменные окружения
load_dotenv()

# Получаем API ключ из переменных окружения
DEFAULT_API_KEY = os.getenv('PPLX_API_KEY', '')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return '\n'.join(text)

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = []
    for page in pdf_reader.pages:
        text.append(page.extract_text())
    return '\n'.join(text)

def extract_text_from_fb2(file):
    content = file.read()
    tree = etree.fromstring(content)
    texts = tree.xpath('//body//section//p')
    return '\n'.join([etree.tostring(t, method='text', encoding='unicode') for t in texts])

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize and join back
    tokens = nltk.word_tokenize(text)
    return ' '.join(tokens)

def calculate_similarity(text1, text2):
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Calculate TF-IDF matrices
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except:
        return 0.0

def rewrite_text(text, api_key):
    """
    Перефразирует текст с помощью Perplexity API
    """
    if not api_key:
        raise ValueError("Требуется API ключ Perplexity")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Разбиваем текст на части по 2000 символов
    chunks = [text[i:i + 2000] for i in range(0, len(text), 2000)]
    rewritten_chunks = []

    for chunk in chunks:
        try:
            data = {
                "model": "sonar-reasoning-pro",  # новая поддерживаемая модель
                "messages": [
                    {
                        "role": "system",
                        "content": "Ты - эксперт по перефразированию текста. Твоя задача - переписать текст, сохраняя его смысл, но используя другие слова и структуры предложений. Текст должен быть грамотным и естественным."
                    },
                    {
                        "role": "user",
                        "content": f"Перепиши следующий текст другими словами, сохраняя смысл и стиль, но избегая плагиата:\n\n{chunk}"
                    }
                ]
            }

            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                rewritten_chunks.append(result['choices'][0]['message']['content'])
            else:
                try:
                    error_detail = response.json()
                except Exception:
                    error_detail = response.text
                st.error(f"Ошибка API: {response.status_code} - {error_detail}")
                return None
                
            time.sleep(1)  # Небольшая задержка между запросами
            
        except Exception as e:
            st.error(f"Ошибка при перефразировании: {str(e)}")
            return None

    return "\n".join(rewritten_chunks)

def main():
    st.set_page_config(page_title="Проверка на плагиат", layout="wide")
    
    # Инициализация состояния сессии
    if 'api_key' not in st.session_state:
        st.session_state['api_key'] = DEFAULT_API_KEY
    
    st.title("Проверка на плагиат")
    
    # Боковая панель для настроек
    with st.sidebar:
        st.header("Настройки")
        api_key = st.text_input("Perplexity API Key", value=st.session_state['api_key'], type="password")
        if api_key:
            st.session_state['api_key'] = api_key
        if DEFAULT_API_KEY:
            st.success("API ключ загружен из .env файла")

    st.write("Загрузите два файла для сравнения (поддерживаемые форматы: DOCX, PDF, FB2, TXT)")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Первый файл")
        file1 = st.file_uploader("Выберите первый файл", type=['docx', 'pdf', 'fb2', 'txt'])

    with col2:
        st.subheader("Второй файл")
        file2 = st.file_uploader("Выберите второй файл", type=['docx', 'pdf', 'fb2', 'txt'])

    if file1 is not None and file2 is not None:
        try:
            # Extract text from files based on their types
            text1 = ""
            text2 = ""

            for file, text_var in [(file1, 'text1'), (file2, 'text2')]:
                if file.name.endswith('.docx'):
                    text = extract_text_from_docx(file)
                elif file.name.endswith('.pdf'):
                    text = extract_text_from_pdf(file)
                elif file.name.endswith('.fb2'):
                    text = extract_text_from_fb2(file)
                else:  # txt files
                    text = file.read().decode('utf-8')
                
                if text_var == 'text1':
                    text1 = text
                else:
                    text2 = text

            # Preprocess texts
            processed_text1 = preprocess_text(text1)
            processed_text2 = preprocess_text(text2)

            # Calculate similarity
            similarity = calculate_similarity(processed_text1, processed_text2)
            
            # Display results
            st.subheader("Результаты анализа")
            similarity_percentage = similarity * 100
            
            # Create a progress bar for visualization
            st.progress(similarity)
            
            if similarity_percentage < 30:
                result_color = "green"
                verdict = "Низкий уровень сходства"
            elif similarity_percentage < 60:
                result_color = "orange"
                verdict = "Средний уровень сходства"
            else:
                result_color = "red"
                verdict = "Высокий уровень сходства"

            st.markdown(f"<h3 style='color: {result_color};'>Сходство: {similarity_percentage:.2f}% - {verdict}</h3>", 
                       unsafe_allow_html=True)

            # Show text comparison
            with st.expander("Показать тексты"):
                col1, col2 = st.columns(2)
                with col1:
                    st.text_area("Текст 1", text1, height=300)
                with col2:
                    st.text_area("Текст 2", text2, height=300)

            # Добавляем кнопки для перефразирования текста
            st.subheader("Перефразирование текста")
            if not st.session_state['api_key']:
                st.warning("Для использования функции перефразирования, пожалуйста, введите API ключ Perplexity в боковой панели.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Перефразировать текст 1"):
                        with st.spinner("Перефразирование текста 1..."):
                            rewritten_text1 = rewrite_text(text1, st.session_state['api_key'])
                            if rewritten_text1:
                                st.success("Текст успешно перефразирован!")
                                st.text_area("Перефразированный текст 1", rewritten_text1, height=300)
                                
                                # Добавляем кнопку для скачивания
                                st.download_button(
                                    label="Скачать перефразированный текст 1",
                                    data=rewritten_text1,
                                    file_name="rewritten_text1.txt",
                                    mime="text/plain"
                                )
                
                with col2:
                    if st.button("Перефразировать текст 2"):
                        with st.spinner("Перефразирование текста 2..."):
                            rewritten_text2 = rewrite_text(text2, st.session_state['api_key'])
                            if rewritten_text2:
                                st.success("Текст успешно перефразирован!")
                                st.text_area("Перефразированный текст 2", rewritten_text2, height=300)
                                
                                # Добавляем кнопку для скачивания
                                st.download_button(
                                    label="Скачать перефразированный текст 2",
                                    data=rewritten_text2,
                                    file_name="rewritten_text2.txt",
                                    mime="text/plain"
                                )

        except Exception as e:
            st.error(f"Произошла ошибка при обработке файлов: {str(e)}")

if __name__ == "__main__":
    main() 