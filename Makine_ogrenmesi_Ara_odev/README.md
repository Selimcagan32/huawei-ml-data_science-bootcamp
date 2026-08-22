# Musteri Ayrilma (Churn) Tahmini - Ara Odev

## Amac
Bu proje, temel bir siniflandirma problemi uzerinde makine ogrenmesi akisini
uygulamak icin hazirlanmistir. Musteri verileri kullanilarak bir musterinin
hizmeti birakip birakmayacagi (churn) tahmin edilmeye calisilmistir.

Veri seti hazir bulunmadigi icin Python ile 150 satirlik ornek bir musteri
verisi olusturulmustur (`musteri_churn_verisi.csv`).

## Icerik
- `churn_tahmini.py` - veri inceleme, on isleme, oznitelik uretme,
  train/validation/test bolme, model egitimi ve degerlendirme adimlarini
  iceren ana dosya
- `musteri_churn_verisi.csv` - kullanilan veri seti
- `requirements.txt` - gerekli kutuphaneler

## Nasil Calistirilir
```bash
pip install -r requirements.txt
python churn_tahmini.py
```

`musteri_churn_verisi.csv` dosyasinin `churn_tahmini.py` ile ayni klasorde
olmasi gerekmektedir.

## Sonuc Yorumu
Logistic Regression, KNN ve bonus olarak Decision Tree modelleri egitilmis;
validasyon verisinde en yuksek accuracy'yi Logistic Regression vermistir ve
bu yuzden test degerlendirmesi icin secilmistir. Test setinde elde edilen
sonuclar:

- Accuracy: 0.867
- Precision: 0.750
- Recall: 0.750
- F1-score: 0.750

Veri setinde churn=1 sinifinin churn=0 sinifina gore daha az sayida olmasi
(dengesiz siniflar) nedeniyle precision ve recall degerleri accuracy'nin
biraz altinda kalmistir. Bu yuzden model degerlendirmesinde sadece accuracy'ye
degil, confusion matrix ile birlikte precision, recall ve f1-score
degerlerine de bakilmistir.
