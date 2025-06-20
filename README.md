# Plagiarism Checker

## Описание

Веб-приложение для проверки текстовых документов на плагиат с поддержкой форматов DOCX, PDF, FB2 и TXT. Также реализована функция автоматического перефразирования текста с помощью Perplexity AI.

## Возможности
- Загрузка и сравнение двух файлов
- Поддержка форматов: DOCX, PDF, FB2, TXT
- Визуальное отображение процента сходства
- Просмотр исходных текстов
- Предварительная обработка текста для более точного сравнения
- Перефразирование текста через Perplexity AI
- Скачивание перефразированного текста

## Установка
1. Убедитесь, что у вас установлен Python 3.8 или выше
2. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/SergeyLocal/plagiarism_checker.git
   cd plagiarism_checker
   ```
3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
4. Создайте файл `.env` и добавьте ваш API-ключ Perplexity:
   ```
   PPLX_API_KEY=ваш_ключ
   ```

## Запуск приложения
```bash
streamlit run app.py
```

## Использование
1. Введите ваш API-ключ Perplexity в боковой панели (или используйте .env)
2. Загрузите два файла для сравнения
3. Ознакомьтесь с результатом анализа
4. При необходимости воспользуйтесь функцией перефразирования
5. Скачайте перефразированный текст

## Интерпретация результатов
- 0-30% — Низкий уровень сходства (зелёный)
- 31-60% — Средний уровень сходства (оранжевый)
- 61-100% — Высокий уровень сходства (красный)

## Публикация на GitHub
1. Инициализируйте git (если ещё не инициализирован):
   ```bash
   git init
   ```
2. Добавьте все файлы:
   ```bash
   git add .
   ```
3. Сделайте коммит:
   ```bash
   git commit -m "Initial commit"
   ```
4. Добавьте удалённый репозиторий:
   ```bash
   git remote add origin https://github.com/SergeyLocal/plagiarism_checker.git
   ```
5. Отправьте изменения:
   ```bash
   git push -u origin main
   ```

## Важно
- Файл `.env` не попадёт в репозиторий благодаря `.gitignore`.
- Для работы функции перефразирования необходим действующий API-ключ Perplexity.
- Для FB2 используется парсинг через lxml.

## Лицензия
MIT

