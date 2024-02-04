# Генерация заголовков статей для Telegram
Сервис по генерации заголовков статей для Telegram на основании текстов статей.

## Целевая аудитоиря
* Content makers
* Студенты
* Малый бизнес

## Предобученная модель ML-модель
Hugging face - basil-77/rut5-base-absum-hh

## Использование сервиса
### через Streamlit
1) вводим код:
    streamlit run streamlit_app.py
### через API (fastapi + uvicorn)
1) вводим код:
    uvicorn fastapi_app:app
2) открываем полученный http в браузере
3) более подробная документация по адресу:
   полученный Вами http + \docs
   пример: http://120.0.0.1:9000/docs
   
