"""
Öznitelik Mühendisliği

Amaç:
    1. Mevcut sütunlardan yeni öznitelik üretme mantığını basit bir örnek ile uygulama
    2. Korelasyon üzerinden modele daha faydalı olabilecek öznitelikleri seçme mantığını gösterme

Adımlar:
    1. Gerekli kütüphanelerin içeriye aktarılması
    2. Veri seti yükleme
    3. Mevcut sütunlardan yeni öznitelikler üretmek (Feature extraction)
    4. Hedef değişken ile öznitelikler arasındaki korelasyonları inceleme
    5. Mutlak korelasyon değerine göre yüksek olan özniteliklerin seçilmesi (Feature selection)

Kurulumlar:
pip install pandas
"""

# 1. Gerekli kütüphanelerin içeriye aktarılması
import pandas as pd

# 2. veri setinin yüklenmesi
df = pd.read_csv("oznitelik_muhendisligi_pratik.csv")

print(df)

# 3. Mevcut sütunlardan yeni öznitelikler üretmek (Feature extraction)
df["deneyim_orani"] = df["deneyim_yili"] / df["yas"]
df["yillik_harcama_tahmini"] = df["aylik_harcama"] * 12

print(df.head())

# 4. Hedef değişken ile öznitelikler arasındaki korelasyonları inceleme
sayisal_df = df.drop("sehir", axis = 1)
korelasyonlar = sayisal_df.corr(numeric_only=True)["performans_puani"].sort_values(ascending=False)
print(korelasyonlar)

"""
performans_puani          1.000000
deneyim_orani             0.821244 (yüksek pozitif korelasyon)
deneyim_yili              0.597232 (orta yüksek pozitif korelasyon)
yillik_harcama_tahmini    0.317301 (orta pozitif korelasyon)
aylik_harcama             0.317301 (orta pozitif korelasyon)
yas                      -0.224902 (orta negatif korelasyon)
uyelik_suresi_ay         -0.238212 (orta negatif korelasyon)
"""

# 5. Mutlak korelasyon değerine göre yüksek olan özniteliklerin seçilmesi (Feature selection)
secilen_ozniteliler = korelasyonlar[abs(korelasyonlar) > 0.75].index.tolist()
secilen_ozniteliler.remove("performans_puani")

print(secilen_ozniteliler)