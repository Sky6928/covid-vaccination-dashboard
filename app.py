
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from fpdf import FPDF
import tempfile
import os

st.set_page_config(page_title="COVID-19 Вакцинация", layout="centered")
st.title("💉 Вакцинация от COVID-19: Глобальный мониторинг")

# Загрузка данных
@st.cache_data
def load_data():
    df = pd.read_csv("country_vaccinations.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

# Выбор стран
countries = df['country'].dropna().unique().tolist()
selected_countries = st.multiselect("Выберите страны:", countries, default=["Russia", "United States", "India"])

# Фильтрация по странам
filtered_df = df[df['country'].isin(selected_countries)]

# Визуализация
st.subheader("📈 График вакцинации по странам")
for country in selected_countries:
    country_data = filtered_df[filtered_df['country'] == country]
    plt.plot(country_data['date'], country_data['total_vaccinations'], label=country)

plt.xlabel("Дата")
plt.ylabel("Всего прививок")
plt.legend()
st.pyplot(plt.gcf())
plt.clf()

# Прогнозирование для одной страны
st.subheader("🔮 Прогноз вакцинации")
forecast_country = st.selectbox("Выберите страну для прогноза:", selected_countries)

df_country = df[df['country'] == forecast_country]
df_country = df_country[['date', 'total_vaccinations']].dropna()
df_country.columns = ['ds', 'y']

m = Prophet()
m.fit(df_country)
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)

fig = m.plot(forecast)
st.pyplot(fig)

# Генерация PDF отчёта
st.subheader("📄 Генерация отчёта PDF")

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

if st.button("📎 Сформировать PDF"):
    font_path = "DejaVuSans.ttf"
    report_data = df.groupby('country')['total_vaccinations'].max()
    pdf = generate_pdf_report_unicode(selected_countries, report_data, font_path)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as file:
        st.download_button(
            label="📥 Скачать PDF",
            data=file,
            file_name="vaccination_report.pdf",
            mime="application/pdf"
        )

    os.unlink(tmp_path)

st.caption("🧠 Данные: Our World In Data — обновление через Kaggle")

