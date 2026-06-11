<div align="center">
  <img src="img/logo.png" alt="HealthData Analytics Logo" width="300"/>

  # HealthData Analytics
  ### Machine Learning per la Predizione delle Malattie Cardiache

  ![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
  ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
  ![Jupyter](https://img.shields.io/badge/Notebook-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)
  ![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle&logoColor=white)
  ![License](https://img.shields.io/badge/License-MIT-green)

  *Progetto di Machine Learning — A.A. 2025/2026*  
  **Carlomagno Elena · Cervino Rosaria**
</div>

---

## Indice

- [Panoramica del Progetto](#panoramica-del-progetto)
- [Il Dataset](#il-dataset)
- [Struttura del Repository](#struttura-del-repository)
- [Pipeline di Machine Learning](#pipeline-di-machine-learning)
- [Modelli e Risultati](#modelli-e-risultati)
- [Requisiti e Installazione](#requisiti-e-installazione)
- [Come Eseguire il Progetto](#come-eseguire-il-progetto)
- [Sviluppi Futuri](#sviluppi-futuri)

---

## Panoramica del Progetto

Le malattie cardiovascolari rappresentano una delle principali cause di mortalità a livello globale. La diagnosi precoce è determinante per ridurre il rischio di complicazioni gravi, ma la valutazione clinica tradizionale è complessa: richiede di incrociare decine di parametri eterogenei come età, livelli di colesterolo, pressione sanguigna e segnali elettrocardiografici.

**HealthData Analytics** è un sistema di supporto alla diagnosi (CAD — *Computer-Aided Diagnosis*) basato su Machine Learning, capace di classificare automaticamente i pazienti in base alla probabilità di essere affetti da una patologia cardiaca, partendo da esami clinici di routine.

### Task

Il problema è formulato come un task di **Classificazione Binaria Supervisionata**:

| Classe | Etichetta | Significato |
|--------|-----------|-------------|
| `0` | Normal | Paziente sano, nessuna patologia cardiaca |
| `1` | Heart Disease | Paziente con presenza confermata di patologia |

> **Priorità clinica:** il progetto pone particolare attenzione alla minimizzazione dei **Falsi Negativi** — un paziente malato classificato come sano rappresenta il rischio clinico più grave.

---

## Il Dataset

**Fonte:** [Heart Failure Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

Il dataset nasce dalla fusione di cinque database cardiologici indipendenti (Cleveland, Hungarian, Switzerland, Long Beach e Stalog), rendendolo uno dei più completi disponibili per la ricerca sulle malattie cardiovascolari.

| Proprietà | Valore |
|-----------|--------|
| Istanze (pazienti) | 918 |
| Feature | 11 |
| Variabile target | 1 (`HeartDisease`) |
| Distribuzione classi | ~45% sani · ~55% malati |

### Descrizione delle Feature

**Dati anagrafici e clinici generali**

| Feature | Tipo | Descrizione |
|---------|------|-------------|
| `Age` | Numerico | Età del paziente (anni) |
| `Sex` | Categorico | Sesso (`M` / `F`) |
| `RestingBP` | Numerico | Pressione sanguigna a riposo (mm Hg) |
| `Cholesterol` | Numerico | Colesterolo sierico (mm/dl) |
| `FastingBS` | Binario | Glicemia a digiuno > 120 mg/dl (`1` = sì, `0` = no) |

**Sintomatologia ed ECG**

| Feature | Tipo | Descrizione |
|---------|------|-------------|
| `ChestPainType` | Categorico | Tipo di dolore toracico (`TA`, `ATA`, `NAP`, `ASY`) |
| `RestingECG` | Categorico | Risultato ECG a riposo (`Normal`, `ST`, `LVH`) |
| `MaxHR` | Numerico | Frequenza cardiaca massima raggiunta (60–202 bpm) |
| `ExerciseAngina` | Binario | Angina da sforzo fisico (`Y` / `N`) |
| `Oldpeak` | Numerico | Depressione segmento ST indotta dall'esercizio |
| `ST_Slope` | Categorico | Pendenza segmento ST (`Up`, `Flat`, `Down`) |

---

## Struttura del Repository

```
HealthDataAnalytics/
│
├── img/
│   └── logo.png                        # Logo del progetto
│
├── HealthDataAnalytics.ipynb           # Notebook principale (Google Colab)
├── heart.csv                           # Dataset originale (da Kaggle)
├── heart_disease_processed.csv         # Dataset pre-processato (output)
│
├── HealthDataAnalytics.pdf             # Relazione tecnica del progetto
└── README.md
```

---

## Pipeline di Machine Learning

```
Raw Data (heart.csv)
        │
        ▼
┌─────────────────────────────┐
│     1. DATA EXPLORATION     │  Analisi distribuzione classi, info(),
│                             │  identificazione anomalie
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│    2. DATA PREPROCESSING    │
│                             │
│  ├─ Imputazione Colesterolo │  172 valori = 0 → sostituiti con mediana
│  ├─ One-Hot Encoding        │  Sex, ChestPainType, RestingECG,
│  │                          │  ExerciseAngina, ST_Slope (drop_first=True)
│  └─ Feature Scaling         │  StandardScaler → μ=0, σ=1
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│    3. DATASET SPLITTING     │  Hold-out: 80% train / 20% test
│                             │  (random_state=42 per riproducibilità)
│                             │  → 734 train | 184 test
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│    4. MODEL TRAINING        │
│                             │
│  ├─ Logistic Regression     │
│  ├─ Random Forest           │  n_estimators=100, random_state=42
│  └─ SVM (kernel RBF)        │  probability=True
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│    5. EVALUATION            │  Accuracy, Precision, Recall, F1-Score
│                             │  Confusion Matrix, ROC Curve, AUC,
│                             │  Feature Importance (RF)
└─────────────────────────────┘
```

### Scelte di Preprocessing — Dettaglio

**Gestione del Colesterolo a zero**  
172 istanze presentavano `Cholesterol = 0`, valore biologicamente impossibile. Tali record sono stati trattati come missing values e imputati con la **mediana calcolata escludendo gli zeri** (237.00 mm/dl), preferita alla media per la sua robustezza agli outlier.

**One-Hot Encoding con `drop_first=True`**  
Le variabili categoriche sono state codificate in colonne binarie. Il parametro `drop_first=True` elimina la prima categoria da ogni gruppo, prevenendo la **multicollinearità perfetta** — condizione critica per la corretta convergenza della Regressione Logistica.

**StandardScaler**  
I range numerici sono molto eterogenei (l'età va da 28 a 77, il colesterolo supera i 600). Lo scaling garantisce che nessuna variabile domini le altre per magnitudine, fondamentale per SVM e algoritmi basati su discesa del gradiente.

---

## Modelli e Risultati

### Metriche di valutazione sul Test Set

| Metrica | Logistic Regression | **Random Forest** | SVM |
|---------|:-------------------:|:-----------------:|:---:|
| Accuracy | 0.8641 | **0.8696** | 0.8478 |
| Precision | **0.9100** | 0.8879 | 0.8762 |
| Recall | 0.8505 | **0.8879** | 0.8598 |
| F1-Score | 0.8792 | **0.8879** | 0.8679 |
| AUC (ROC) | 0.9260 | 0.9260 | **0.9303** |

### Analisi per modello

**Logistic Regression** — Precision più alta (0.91), ma 16 Falsi Negativi rendono il modello meno sicuro per lo screening clinico primario.

**Random Forest** ⭐ — Miglior bilanciamento complessivo. Recall più alta (0.8879) con soli 12 Falsi Negativi: è il **modello raccomandato** per l'impiego clinico pratico, grazie alla capacità di catturare relazioni non lineari tra variabili eterogenee.

**SVM (kernel RBF)** — AUC più alta (0.9303), indicatore di superiore capacità discriminante sull'intero spettro delle probabilità, ma leggermente inferiore a Random Forest sulle metriche di soglia standard.

### Feature Importance (Random Forest)

Le feature più predittive identificate dal modello, in ordine di importanza:

```
ST_Slope_Up      ████████████████  ~0.155
MaxHR            ████████████      ~0.110
Oldpeak          ████████████      ~0.108
ST_Slope_Flat    ███████████       ~0.100
ExerciseAngina_Y ██████████        ~0.085
Age              █████████         ~0.080
Cholesterol      ███████           ~0.065
RestingBP        ██████            ~0.055
Sex_M            ████              ~0.035
ChestPainType_ATA███               ~0.030
```

> I segnali ECG (`ST_Slope`, `Oldpeak`) risultano più determinanti dell'età anagrafica, confermando che il modello si concentra su **indicatori clinici oggettivi** piuttosto che su dati demografici.

---

## Requisiti e Installazione

### Dipendenze Python

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### Installazione locale

```bash
git clone https://github.com/<username>/HealthDataAnalytics.git
cd HealthDataAnalytics

pip install pandas numpy matplotlib seaborn scikit-learn
```

### Esecuzione su Google Colab (consigliato)

Fare clic sul badge per aprire direttamente il notebook in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<username>/HealthDataAnalytics/blob/main/HealthDataAnalytics.ipynb)

---

## Come Eseguire il Progetto

**1. Scarica il dataset**  
Scarica `heart.csv` dalla [pagina Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) e posizionalo nella root del progetto.

**2. Apri il notebook**  
Esegui `HealthDataAnalytics.ipynb` seguendo le celle in ordine. Il notebook è suddiviso in sezioni commentate:

```
📂 Sezione 1 — Caricamento e analisi esplorativa
📂 Sezione 2 — Preprocessing (imputazione, encoding, scaling)
📂 Sezione 3 — Splitting del dataset
📂 Sezione 4 — Addestramento e valutazione dei modelli
📂 Sezione 5 — Confronto ROC e Feature Importance
📂 Sezione 6 — Esportazione del dataset processato
```

**3. Output generato**  
Al termine dell'esecuzione viene prodotto il file `heart_disease_processed.csv` contenente il dataset post-preprocessing, pronto per utilizzi futuri.

---

## Sviluppi Futuri

- **Integrazione di nuove feature** — variabili legate allo stile di vita (fumo, attività fisica, dieta) per aumentare il potere predittivo
- **Hyperparameter tuning** — ottimizzazione sistematica dei modelli tramite Grid Search o Bayesian Optimization
- **Cross-validation** — sostituzione dell'hold-out con k-fold CV per stime più robuste
- **Interfaccia grafica** — dashboard per l'inserimento dei parametri clinici e la visualizzazione dei risultati predetti
- **Explainability** — integrazione di SHAP o LIME per interpretabilità a livello di singola predizione

---

<div align="center">

*Progetto realizzato nell'ambito del corso di Machine Learning — A.A. 2025/2026*  
Carlomagno Elena · Cervino Rosaria

</div>
