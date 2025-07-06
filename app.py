import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from fpdf import FPDF
import tempfile
import os

# Настройки
st.set_page_config(page_title="Вакцинация по COVID-19", layout="wide")
st.title("🌍 Мировая статистика вакцинации от COVID-19 — LIVE")

@st.cache_data
def load_data():
    df = pd.read_csv("country_vaccinations.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

# Выбор страны
countries = sorted(df["country"].unique())
selected_countries = st.multiselect("Выберите страну", countries, default=["Russia", "United States", "India"])

# Метрики
st.header("📌 Основные метрики")
for country in selected_countries:
    country_df = df[df["country"] == country].dropna(subset=["total_vaccinations"])
    start = country_df["date"].min().strftime('%Y-%m-%d')
    total = int(country_df["total_vaccinations"].max())
    avg = int(country_df["daily_vaccinations"].mean())
    st.metric(label=f"{country}", value=f"{total:,}", delta=f"Старт: {start} | Ср. {avg:,}/день")

# График
st.header("📈 График вакцинаций")
fig, ax = plt.subplots(figsize=(10, 5))
for country in selected_countries:
    df_c = df[df["country"] == country].dropna(subset=["total_vaccinations"])
    ax.plot(df_c["date"], df_c["total_vaccinations"], label=country)
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Прогноз (Prophet)
st.header("🔮 Прогноз на 30 дней")
if len(selected_countries) == 1:
    country = selected_countries[0]
    df_p = df[df["country"] == country].dropna(subset=["daily_vaccinations"])
    df_p = df_p[["date", "daily_vaccinations"]].rename(columns={"date": "ds", "daily_vaccinations": "y"})
    if len(df_p) > 10:
        model = Prophet()
        model.fit(df_p)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        fig2 = model.plot(forecast)
        st.pyplot(fig2)
    else:
        st.warning("Недостаточно данных для прогноза.")
else:
    st.info("Прогноз доступен только при выборе одной страны.")

# PDF отчёт
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

st.header("📥 Скачать отчёт в PDF")
if st.button("📄 Сформировать PDF"):
    font_path = "DejaVuSans.ttf"  # Убедись, что он в папке
    report_data = df.groupby("country")["total_vaccinations"].max()
    pdf = generate_pdf_report_unicode(selected_countries, report_data, font_path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        tmp_path = tmp.name
    with open(tmp_path, "rb") as file:
        st.download_button("📎 Скачать PDF", data=file, file_name="vaccination_report.pdf", mime="application/pdf")
    os.unlink(tmp_path)

st.markdown("---")
st.caption("Источник данных: [Kaggle – COVID-19 Vaccination Progress](https://www.kaggle.com/datasets/gpreda/covid-world-vaccination-progress)")

