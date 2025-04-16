# 📈 Real-Time-Securities-Prediction-using-ARIMA-LSTM-LR-and-Sentiment-Analysis

Welcome to Dr Q. P. ! 

Stock Market Prediction Web App based on Machine Learning and Sentiment Analysis of Tweets (API keys included in code). The front end of the Web App is based on Flask. The App forecasts stock prices of the next seven days for any given stock under NASDAQ or NSE as input by the user. Predictions are made using ARIMA (Autoregressive Integrated Moving Average), LSTM (Long Short-Term Memory), Linear Regression, and Sentiment Analysis. The Web App combines the predicted prices of the next seven days with the sentiment analysis of tweets to give recommendation whether the price is going to rise or fall.

## 🌟 Features

- 🧠 Multi-model ML Securities Prediction using:
  - ARIMA
  - LSTM (deep learning)
  - Linear Regression
- 💬 Sentiment analysis from Twitter (optional, using `TextBlob`)
- 📊 Visualizations with Matplotlib
- Index page for searching stock ticker symbol
- Stock Prediction Results Page
- 🍪 “Remember me” and session managementPersistant Login Feature... 
- 🔐 User authentication (signup, login, logout)
- ✅ Email verification via secure token link
- 📩 Forgot password and reset functionality
- 🧹 Secure password hashing with Werkzeug
- 🗑️ Delete account option
- About Page
- Home Page
- Contact Us

## 🛠 Tech Stack

- **Backend**: Python, Flask, Flask-Login, Flask-Mail, SQLAlchemy
- **ML/AI**: scikit-learn, TensorFlow/Keras, statsmodels, yfinance
- **Sentiment Analysis**: TextBlob, Tweepy, preprocessor
- **Frontend**: HTML, CSS, Bootstrap (optional)
- **Database**: SQLite (can upgrade to PostgreSQL)

---

## 📦 Installation

1. Clone the repository
'''python
git clone https://github.com/yourusername/stock-prediction-app.git
cd stock-prediction-app
'''

3. Create virtual environment
'''python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate'''

4. Install dependencies
'''pip install -r requirements.txt'''

5. Set up environment variables (in .env or directly in config.py)
Flask-Mail, Twitter API keys, Secret key, etc.

6. Run the app
'''python main.py'''

---------------

## 🔐 Config (`config.py`)

Set your configuration values:

```python
class Config:
    SECRET_KEY = 'your_secret_key_here'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///database.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = 'youremail@gmail.com'
    MAIL_PASSWORD = 'your_email_password_or_app_password'
```
---
## 📸 Screenshots

> - Home Page
<img src='pics\home1.png'>

> - Search with the Stock ticker symbol
<img src='pics\Predict-search.png'>

> - Results page with predictions and graphs
<img src='pics\result.png'>

> - About us
<img src='pics\aboutus1.png'>

> - Contact us
<img src='pics\contactus.png'>

> - Signup/login page
<img src='pics\login.png'>

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Contributions

Pull requests are welcome. Feel free to fork and submit improvements or feature additions!

---



## Project is based on my research paper:

Autonomous Quantum Financial Omniscient Network: Self-Evolving, Multimodal, Quantum-Enhanced Graph Neural Framework For Adaptive, Risk-Aware Securities Prediction 
(ISBN No : 978-93-48733-42-9)

https://drive.google.com/file/d/1aei4qoXQzg36ATs3hNix6mrvRZm6l5a8/view?usp=sharing

https://drive.google.com/file/d/1yuhGEEqc-FelKFLdqaNysflez5UEYnoL/view?usp=sharing


## 🧑‍💻 Developed By

**Simon Anandan** 

cat("ML", "SWE", "datascience", sep=" ∩ ")
'# Changing the world one commit at a time'

[Linkedin](https://www.linkedin.com/in/simon-anandan/)

[Github](https://github.com/EricMaxwellnetizen?tab=repositories)





