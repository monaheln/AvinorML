# AFIS Concurrency Prediction - Dokumentasjon

## a. Metodevalg og tilnærming

**Problemstilling**: Predikere når AFIS-operatører vil oppleve samtidig kommunikasjon med flere fly i samme time.

**Valgt metode**: Random Forest maskinlæring
- Samtidighet oppstår når kommunikasjonsvinduene til fly overlapper
- For ankomster: 16 minutter før til 5 minutter etter landing
- For avganger: 15 minutter før til 8 minutter etter avgang

**Hvorfor Random Forest**:
- Kan håndtere mange forskjellige typer variabler (tall, kategorier, tid)
- Gir tydelig forklaring på hvilke faktorer som påvirker prediksjoner
- Robust mot feil i data
- Egnet for denne type binær klassifisering (samtidighet/ikke samtidighet)

## b. Systemstruktur og arkitektur

**Systemkomponenter**:
```
Historiske flydata → Databehandling → Modelltrening → Prediksjoner
     (2018-2025)         ↓               ↓            ↓
                    Feature engineering  Validering   Oktober 2025
```

**Arkitektur**:
1. **Datainnlesing**: CSV-filer med flydata og flyplassgrupper
2. **Databehandling**: Rydding av data, håndtering av kansellerte fly
3. **Feature engineering**: Lage nye variabler fra rådata
4. **Modelltrening**: Random Forest med tidsbasert validering
5. **Prediksjon**: Generer sannsynligheter for oktober 2025

**Tekniske komponenter**:
- Python 3.9+ med pandas, scikit-learn
- Ingen eksterne API-er nødvendig
- Standalone applikasjon

## c. Modeller og algoritmer

**Hovedmodell**: Random Forest Classifier
- 100 trær i ensemblet
- Maksimal dybde: 10 nivåer
- Minimum samples per split: 10
- Balanserte klassegewichter for å håndtere sjeldne samtidighetshendelser

**Evalueringsmetode**:
- AUC-ROC som hovedmetrikk (oppnådd: 0.9563)
- Tidsbasert validering: data før 2024 for trening, 2024+ for validering
- Unngår datalekkasje ved å ikke bruke fremtidig informasjon

**Viktigste variabler** (feature importance):
1. Planlagt samtidighet (34%): Om rutetabellen antyder overlapp
2. Antall fly per time (20%): Flere fly = høyere risiko
3. Samtidighetsrisiko-flagg (20%): Binær indikator for kritiske timer
4. Fly-time interaksjon (14%): Kombinasjonseffekt av volum og tidspunkt

## d. Kildekode

**Filstruktur**:
- `afis_simple_model.py`: Hovedmodell som genererer prediksjoner
- `afis_concurrency_model.py`: Utvidet implementering med detaljert feature engineering
- `october_2025_predictions.csv`: Ferdig prediksjonsfil
- `feature_importance.csv`: Forklaring av modellens beslutningsgrunnlag

**Installasjon og kjøring**:
```bash
# Krav: Python 3.9+, pandas, scikit-learn, numpy
pip install pandas scikit-learn numpy

# Kjør modellen
python afis_simple_model.py
```

**Hovedkomponenter i koden**:

1. **SimplifiedAFISModel klasse**:
   - `load_and_prepare_data()`: Laster inn data
   - `engineer_core_features()`: Lager prediksjonsvariabler
   - `train_model()`: Trener Random Forest
   - `predict_october()`: Generer oktober-prediksjoner

2. **Feature engineering**:
   - Tid-baserte variabler: time på døgnet, ukedag, sesong
   - Flytrafikk-variabler: antall fly, planlagt samtidighet
   - Interaksjonsvariabler: kombinerte effekter

3. **Modelltrening**:
   - Tidsbasert datasplit for realistisk validering
   - Hyperparameter-optimalisering for beste ytelse
   - Feature importance-analyse for forklarbarhet

**Skalering og videreutvikling**:
- Kan håndtere større datasett ved å øke sample-størrelse
- Enkel å legge til nye variabler (vær, forsinkelser, etc.)
- Modellen kan retrenes med nye data uten kodeendringer

**Resultater**:
- **Validerings-AUC**: 0.9563 (svært god prediksjonsevne)
- **Oktober-prediksjoner**: 5,047 time-prediksjoner generert
- **Høyrisiko-timer**: 1,256 timer identifisert (25% av total)

Modellen viser at antall planlagte fly og tidsintervaller mellom fly er de sterkeste indikatorene for samtidighetsrisiko, noe som stemmer godt med operasjonell erfaring.