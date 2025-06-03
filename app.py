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
import concurrent.futures
import streamlit_lottie as st_lottie
import random

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
    Перефразирует текст с помощью Perplexity API параллельно по чанкам
    """
    if not api_key:
        raise ValueError("Требуется API ключ Perplexity")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Разбиваем текст на части по 2000 символов
    chunks = [text[i:i + 2000] for i in range(0, len(text), 2000)]
    rewritten_chunks = [None] * len(chunks)

    def paraphrase_chunk(idx, chunk):
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
                return idx, result['choices'][0]['message']['content']
            else:
                try:
                    error_detail = response.json()
                except Exception:
                    error_detail = response.text
                return idx, f"[Ошибка API: {response.status_code} - {error_detail}]"
        except Exception as e:
            return idx, f"[Ошибка при перефразировании: {str(e)}]"

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(paraphrase_chunk, idx, chunk) for idx, chunk in enumerate(chunks)]
        for future in concurrent.futures.as_completed(futures):
            idx, rewritten = future.result()
            rewritten_chunks[idx] = rewritten

    return "\n".join(rewritten_chunks)

def main():
    st.set_page_config(page_title="Проверка на плагиат — Plagiarism Checker", layout="wide")

    # --- Кастомный CSS для модного дизайна ---
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700&display=swap');
        html, body, [class*="css"]  {
            font-family: 'Montserrat', sans-serif !important;
            background: linear-gradient(135deg, #1f1c2c 0%, #928dab 100%) fixed;
            color: #fff !important;
        }
        .stApp {
            background: transparent;
        }
        .stButton>button {
            background: linear-gradient(90deg, #ff512f 0%, #dd2476 100%);
            color: #fff;
            border: none;
            border-radius: 8px;
            padding: 0.6em 1.5em;
            font-weight: bold;
            font-size: 1.1em;
            box-shadow: 0 4px 20px #dd247655;
            transition: 0.2s;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #24c6dc 0%, #5433ff 100%);
            color: #fff;
            transform: scale(1.05);
        }
        .stTextInput>div>div>input[type="password"] {
            background: #232526;
            color: #fff;
            border-radius: 6px;
        }
        .stTextArea textarea {
            background: #232526;
            color: #fff;
            border-radius: 6px;
        }
        .stDownloadButton>button {
            background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
            color: #fff;
            border-radius: 8px;
            font-weight: bold;
        }
        .stProgress>div>div>div>div {
            background: linear-gradient(90deg, #ff512f 0%, #dd2476 100%) !important;
        }
        .stExpanderHeader {
            font-size: 1.1em;
            color: #ffb347 !important;
        }
        .stSidebar {
            background: #232526 !important;
        }
        .stMarkdown h3 {
            font-size: 1.5em;
            font-weight: bold;
        }
        footer {
            visibility: hidden;
        }
        .custom-footer {
            position: fixed;
            left: 0; right: 0; bottom: 0;
            width: 100%;
            background: rgba(31,28,44,0.95);
            color: #fff;
            text-align: center;
            padding: 0.5em 0 0.5em 0;
            font-size: 1em;
            z-index: 100;
        }
        /* Анимация для кнопок */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        .stButton>button:hover {
            animation: pulse 0.5s infinite;
        }
        /* Кастомные уведомления (toast) */
        .toast {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(35,37,38,0.9);
            color: #fff;
            padding: 1em;
            border-radius: 8px;
            box-shadow: 0 4px 20px #0002;
            z-index: 1000;
            animation: fadeIn 0.3s, fadeOut 0.3s 2.7s;
            animation-fill-mode: forwards;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes fadeOut {
            from { opacity: 1; }
            to { opacity: 0; }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # --- Модный заголовок с эмодзи ---
    st.markdown("""
    <h1 style='text-align: center; font-size: 2.7em; margin-bottom: 0.2em;'>🦄 Plagiarism Checker <span style='font-size:0.7em;'>by SergeyLocal</span></h1>
    <h3 style='text-align: center; color: #ffb347; margin-top: 0;'>Проверь свой текст на оригинальность и перефразируй по-новому! 🚀</h3>
    """, unsafe_allow_html=True)

    # Инициализация состояния сессии
    if 'api_key' not in st.session_state:
        st.session_state['api_key'] = DEFAULT_API_KEY
    if 'theme' not in st.session_state:
        st.session_state['theme'] = 'dark'

    # Боковая панель для настроек
    with st.sidebar:
        st.markdown("""
        <h2 style='color:#ffb347;'>⚙️ Настройки</h2>
        """, unsafe_allow_html=True)
        api_key = st.text_input("Perplexity API Key", value=st.session_state['api_key'], type="password")
        if api_key:
            st.session_state['api_key'] = api_key
        if DEFAULT_API_KEY:
            st.success("API ключ загружен из .env файла")
        # Переключатель темы
        theme = st.radio("Тема", ["Темная", "Светлая"], index=0 if st.session_state['theme'] == 'dark' else 1)
        if theme == "Светлая" and st.session_state['theme'] != 'light':
            st.session_state['theme'] = 'light'
            st.markdown("""
            <script>
            document.body.style.background = 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)';
            document.body.style.color = '#333';
            </script>
            """, unsafe_allow_html=True)
        elif theme == "Темная" and st.session_state['theme'] != 'dark':
            st.session_state['theme'] = 'dark'
            st.markdown("""
            <script>
            document.body.style.background = 'linear-gradient(135deg, #1f1c2c 0%, #928dab 100%)';
            document.body.style.color = '#fff';
            </script>
            """, unsafe_allow_html=True)
        st.markdown("""
        <hr style='border:1px solid #444; margin:1em 0;'>
        <div style='color:#aaa; font-size:0.95em;'>
        <b>Поддержка:</b> <a href='https://github.com/SergeyLocal/plagiarism_checker' style='color:#38ef7d;' target='_blank'>GitHub</a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-bottom:1.5em;'></div>
    <div style='background:rgba(35,37,38,0.7); border-radius:16px; padding:1.5em; box-shadow:0 2px 16px #0002;'>
    <b>Загрузите два файла для сравнения (DOCX, PDF, FB2, TXT)</b>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h4>📄 Первый файл</h4>", unsafe_allow_html=True)
        file1 = st.file_uploader("Выберите первый файл", type=['docx', 'pdf', 'fb2', 'txt'])

    with col2:
        st.markdown("<h4>📄 Второй файл</h4>", unsafe_allow_html=True)
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
            similarity_percentage = similarity * 100
            
            # --- Модный прогресс-бар с эмодзи ---
            st.markdown(f"""
            <div style='margin:1.5em 0;'>
                <div style='font-size:1.2em;'>
                    <b>Результаты анализа</b> <span style='font-size:1.5em;'>{'🟢' if similarity_percentage < 30 else ('🟠' if similarity_percentage < 60 else '🔴')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(similarity)
            
            if similarity_percentage < 30:
                result_color = "#38ef7d"
                verdict = "Низкий уровень сходства — оригинально!"
            elif similarity_percentage < 60:
                result_color = "#ffb347"
                verdict = "Средний уровень сходства — будь осторожен!"
            else:
                result_color = "#ff512f"
                verdict = "Высокий уровень сходства — ⚠️ Плагиат!"

            st.markdown(f"<h3 style='color: {result_color};'>Сходство: {similarity_percentage:.2f}% — {verdict}</h3>", unsafe_allow_html=True)

            # Show text comparison
            with st.expander("👀 Показать тексты"):
                col1, col2 = st.columns(2)
                with col1:
                    st.text_area("Текст 1", text1, height=300)
                with col2:
                    st.text_area("Текст 2", text2, height=300)

            # Добавляем кнопки для перефразирования текста
            st.markdown("<h4>🤖 Перефразирование текста</h4>", unsafe_allow_html=True)
            if not st.session_state['api_key']:
                st.warning("Для использования функции перефразирования, пожалуйста, введите API ключ Perplexity в боковой панели.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✨ Перефразировать текст 1"):
                        with st.spinner("Перефразирование текста 1..."):
                            rewritten_text1 = rewrite_text(text1, st.session_state['api_key'])
                            if rewritten_text1:
                                st.success("Текст успешно перефразирован! 🥳")
                                st.text_area("Перефразированный текст 1", rewritten_text1, height=300)
                                st.download_button(
                                    label="⬇️ Скачать перефразированный текст 1",
                                    data=rewritten_text1,
                                    file_name="rewritten_text1.txt",
                                    mime="text/plain"
                                )
                
                with col2:
                    if st.button("✨ Перефразировать текст 2"):
                        with st.spinner("Перефразирование текста 2..."):
                            rewritten_text2 = rewrite_text(text2, st.session_state['api_key'])
                            if rewritten_text2:
                                st.success("Текст успешно перефразирован! 🥳")
                                st.text_area("Перефразированный текст 2", rewritten_text2, height=300)
                                st.download_button(
                                    label="⬇️ Скачать перефразированный текст 2",
                                    data=rewritten_text2,
                                    file_name="rewritten_text2.txt",
                                    mime="text/plain"
                                )

        except Exception as e:
            st.error(f"Произошла ошибка при обработке файлов: {str(e)}")

    # --- Модный футер ---
    st.markdown("""
    <div class='custom-footer'>
        <span>Made with ❤️ for <b>молодёжь</b> | <a href='https://github.com/SergeyLocal/plagiarism_checker' style='color:#38ef7d;' target='_blank'>GitHub</a> | <a href='https://t.me/SKIZZI4I' style='color:#ffb347;' target='_blank'>Telegram</a></span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 