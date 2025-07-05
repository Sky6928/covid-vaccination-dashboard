# COVID-19 Vaccination Dashboard

Интерактивный дашборд в Streamlit для анализа и прогнозирования данных по вакцинации против COVID-19 во всём мире.

## 📊 Возможности:
- Выбор одной или нескольких стран
- Отображение темпов вакцинации и даты старта
- Графики вакцинации
- Прогноз на 30 дней (Prophet)
- Генерация PDF-отчёта с поддержкой русского языка

## 📁 Файлы:
- `app.py` — основной код дашборда
- `country_vaccinations.csv` — данные
- `DejaVuSans.ttf` — шрифт для PDF
- `requirements.txt` — зависимости

## ▶ Запуск
```bash
pip install -r requirements.txt
streamlit run app.py

