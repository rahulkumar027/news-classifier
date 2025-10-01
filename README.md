📰 News Classifier

A machine learning–based web app that classifies news articles into categories (e.g., Business, Sports, Tech, Entertainment).
Built with Python, Flask, Scikit-learn, and Joblib.

🚀 Features

Train a model on news data (train_model.py).

Flask web app (app.py) for real-time text classification.

Pretrained model stored in model/news_classifier.joblib.

Web UI built with HTML templates.

Modular source code in src/.

📂 Project Structure
news-classifier/
│
├── app.py                  # Flask web app
├── train_model.py          # Train the ML model
├── requirements.txt        # Dependencies
├── Dockerfile              # (Optional) container setup
├── model/
│   └── news_classifier.joblib   # Saved ML model
├── src/
│   ├── preprocessing.py    # Data cleaning and prep
│   ├── train.py            # Training logic
│   ├── infer.py            # Prediction logic
│   └── utils.py            # Helper functions
├── templates/
│   └── index.html          # Web UI
└── data/
    └── news.csv (❌ ignored - too large for GitHub)

⚙️ Installation & Setup
1. Clone the repository
git clone https://github.com/rahulkumar027/news-classifier.git
cd news-classifier

2. Create and activate a virtual environment
# Windows (PowerShell)
python -m venv venv
venv\Scripts\activate

# Linux / Mac
python3 -m venv venv
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

📊 Dataset

This repo excludes the full dataset (data/news.csv) because it is too large (>100 MB).

👉 To run the project, place your dataset at:

news-classifier/data/news.csv


Example structure:

data/
  news.csv


If you don’t have a dataset, you can create a sample for quick testing:

echo "text,label" > data/news.csv
echo "Stocks rallied after the central bank announcement.,Business" >> data/news.csv
echo "The striker scored a hat-trick to win the derby.,Sports" >> data/news.csv

🏋️ Train the Model
python train_model.py


This will train a new classifier and save it to model/news_classifier.joblib.

🌐 Run the Web App
python app.py


Then open http://127.0.0.1:5000
 in your browser.

You’ll see a simple web form where you can input news text and get predictions.

🐳 Docker (Optional)

Build and run with Docker:

docker build -t news-classifier .
docker run -p 5000:5000 news-classifier

📸 Demo Screenshot

<img width="1913" height="1028" alt="image" src="https://github.com/user-attachments/assets/3d0e6c94-f053-49d9-acb0-d3379a330a76" />


🙌 Author

Rahul Kumar

GitHub: @rahulkumar027
