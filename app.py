import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import os
from prophet import Prophet

# Настройки страницы
st.set_page_config(page_title="COVID-19 Вакцинация", layout="wide")
st.title("📊 Дашборд: Прогресс вакцинации по COVID-19")

# --- Загрузка и кэширование данных ---
@st.cache_data
def load_data():
    df = pd.read_csv("country_vaccinations.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

# --- Выбор стран ---
countries = sorted(df['country'].unique())
selected_countries = st.multiselect("Выберите одну или несколько стран", countries, default=["Russia", "United States", "India"])

# --- Метрики ---
st.header("📌 Основные метрики по странам")
col1, col2, col3 = st.columns(3)

for idx, country in enumerate(selected_countries[:3]):
    country_df = df[df['country'] == country].dropna(subset=["total_vaccinations", "daily_vaccinations"])
    start_date = country_df['date'].min().strftime('%Y-%m-%d')
    total = int(country_df['total_vaccinations'].max())
    avg_daily = int(country_df['daily_vaccinations'].mean())

    col = [col1, col2, col3][idx]
    with col:
        st.metric(f"{country}", f"{total:,} доз", f"Старт: {start_date}")
        st.caption(f"Средний темп: {avg_daily:,}/день")

# --- График вакцинаций ---
st.header("📈 График вакцинаций")
fig, ax = plt.subplots(figsize=(10, 5))
for country in selected_countries:
    country_df = df[df['country'] == country].dropna(subset=['total_vaccinations'])
    ax.plot(country_df['date'], country_df['total_vaccinations'], label=country)

ax.set_title("Общее количество вакцинаций по странам")
ax.set_xlabel("Дата")
ax.set_ylabel("Прививок")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# --- Топ-10 стран ---
st.header("🏆 Топ-10 стран по вакцинации")
top_10 = df.groupby('country')['total_vaccinations'].max().sort_values(ascending=False).head(10)
fig2, ax2 = plt.subplots(figsize=(8, 4))
top_10.plot(kind='bar', ax=ax2)
ax2.set_title("Топ-10 по общему числу вакцинаций")
ax2.set_ylabel("Прививок")
st.pyplot(fig2)

# --- Прогноз Prophet ---
st.header("🔮 Прогноз вакцинации на 30 дней")
if len(selected_countries) == 1:
    country = selected_countries[0]
    df_country = df[df['country'] == country].dropna(subset=['daily_vaccinations'])
    df_prophet = df_country[['date', 'daily_vaccinations']].rename(columns={"date": "ds", "daily_vaccinations": "y"})

    if len(df_prophet) > 10:
        model = Prophet()
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        fig3 = model.plot(forecast)
        st.pyplot(fig3)
        st.success("Прогноз построен на основе ежедневных данных.")
    else:
        st.warning("Недостаточно данных для прогноза.")
else:
    st.info("Прогноз можно построить только при выборе одной страны.")

# --- Генерация PDF с поддержкой Unicode ---
def generate_pdf_report_unicode(countries_selected, total_data, font_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", size=12)
    pdf.cell(200, 10, txt="COVID-19: Вакцинация по странам", ln=1, align='C')
    pdf.ln(10)
    for country in countries_selected:
        total = total_data.loc[country]
        pdf.cell(200, 10, txt=f"{country}: {int(total):,} прививок", ln=1)
    return pdf

# --- Кнопка скачать PDF ---
st.header("📥 Скачать отчёт в PDF")
if st.button("📄 Сформировать PDF"):
    font_path = "DejaVuSans.ttf"  # Файл должен лежать в папке с app.py
    report_data = df.groupby('country')['total_vaccinations'].max()
    pdf = generate_pdf_report_unicode(selected_countries, report_data, font_path)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as file:
        st.download_button(
            label="📎 Скачать PDF",
            data=file,
            file_name="vaccination_report.pdf",
            mime="application/pdf"
        )

    os.unlink(tmp_path)

# --- Источник ---
st.markdown("---")
st.caption("Данные: [Kaggle – COVID-19 World Vaccination Progress](https://www.kaggle.com/datasets/gpreda/covid-world-vaccination-progress)")
