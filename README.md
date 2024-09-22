
# **CineSage: Your Personalized Cinema Advisor**

CineSage is an advanced movie recommendation engine powered by machine learning. It helps users discover films based on their unique preferences by employing cutting-edge data processing techniques, feature engineering, and model training. The system also includes an intuitive web interface built with **Streamlit**.

---

## **Installation Guide**

To set up and run CineSage locally, follow these steps:

### 1. Clone the repository:

```bash
git clone https://github.com/your-username/CineSage.git
cd CineSage
```

### 2. Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset:

Follow the instructions in the [Dataset Preparation](#dataset-preparation) section to create the necessary `data.csv` file.

---

## **Usage Instructions**

After setting up the environment and preparing the dataset, follow these steps to run the application:

### 1. Run preprocessing and model training scripts:

- Execute the `data_preparation.ipynb` notebook located in the `Preprocessing` directory to generate the `data.csv` file.
- Run the `model_preparation.ipynb` notebook from the `ModelPreparation` directory to train models and create the `models_and_data.pkl` file.

### 2. Start the Streamlit web application:

```bash
streamlit run app.py
```

---

## **Project Structure**

Here is an overview of the project directory structure:

```
CineSage/
├── Preprocessing/
│   └── data_preparation.ipynb      # Preprocessing steps for dataset creation
├── ModelPreparation/
│   └── model_preparation.ipynb     # Model training and preparation script
├── app.py                          # Streamlit application file
├── requirements.txt                # Dependencies required for the project
└── README.md                       # Project documentation
```

---

## **Dataset Preparation**

The dataset used for CineSage is sourced from the **TMDB 5000 Movie Dataset** available on Kaggle. Follow these steps to prepare the dataset:

1. **Download the Dataset**: Get the dataset from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).

2. **Run the Preprocessing Script**:  
   Execute the `data_preparation.ipynb` notebook from the `Preprocessing` folder. This script will preprocess and merge the dataset, creating the `data.csv` file required for further steps.

---

## **Models and Features**

### Data Preprocessing

The `data_preparation.ipynb` notebook includes the following data transformations:

- **Missing Data Handling**: Numerical columns with missing values are filled with their mean, while text columns are replaced with empty strings.
- **Text Preprocessing**:
  - **Stemming**: The `overview` column undergoes stemming using NLTK’s `PorterStemmer`.
- **Feature Scaling**:
  - **Normalization**: Numeric columns are normalized using `MinMaxScaler`.
- **Feature Encoding**:
  - **MultiLabelBinarizer**: Genres are encoded for multi-label classification.
  - **CountVectorizer**: Cast and crew data are vectorized.
  - **TF-IDF Vectorizer**: The `overview` column is vectorized using Term Frequency-Inverse Document Frequency (TF-IDF).

### Model Training

The model training process is executed in the `model_preparation.ipynb` notebook, which includes the following models:

- **Singular Value Decomposition (SVD)**: Used for dimensionality reduction of feature vectors.
- **K-Nearest Neighbors (KNN)**: Leverages cosine similarity to find similar movies.
- **Random Forest Regressor**: Used for predicting movie similarities based on feature combinations.

The trained models and processed data are saved into the `models_and_data.pkl` file for use in the application.

---

## **Streamlit Application**

CineSage features an intuitive interface that allows users to seamlessly browse personalized movie recommendations.

### Key Features:

- **Movie Search**: Search for your favorite movies from the available dataset.
- **Recommendations**: Receive curated recommendations based on your selected movie.
- **Watchlist**: Add movies to your personal watchlist and access it anytime from the sidebar.

### Application Usage:

- **Select a Movie**: Use the dropdown menu to select a movie title.
- **Get Recommendations**: Click the 'Get Recommendations' button to receive movie suggestions.
- **Manage Watchlist**: Your selected watchlist will be displayed in the sidebar.

---

## **Contributing**

We warmly welcome contributions to improve CineSage. To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make the necessary changes.
4. Submit a pull request for review.

---

### **Important Note:**

CineSage relies on the TMDB API to fetch additional movie details such as posters and overviews. If you experience difficulties accessing the API due to regional restrictions (especially in India), consider using a VPN to bypass these restrictions.

---

**Enjoy discovering your next favorite movie with CineSage!**
"""

---

