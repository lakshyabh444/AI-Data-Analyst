# 📊 AI Data Analyst

An AI-powered data analysis application built with **Streamlit**, **Pandas**, **Plotly**, and the **Google Gemini API**. Upload a CSV dataset and let AI uncover key insights, trends, anomalies, and business recommendations.

An AI-powered data analysis application built with **Streamlit**, **Pandas**, **Plotly**, and the **Google Gemini API**. Upload a CSV dataset and let AI uncover key insights, trends, anomalies, and business recommendations.

🚀 **[View Live Demo](https://ai-data-analyst-dr83lgwke3wehef4m6z5rh.streamlit.app/)**
---

## ✨ Features

- **CSV Upload** – Easily upload any CSV file for analysis.
- **Dataset Preview** – View the first rows of your data in an interactive table.
- **Quick Metrics** – Instant summary cards showing rows, columns, numeric columns, and missing values.
- **Basic Statistics** – Descriptive statistics with missing data visualization.
- **Interactive Charts** – Plotly-powered line, bar, area charts, histograms, and box plots with column selection.
- **Correlation Heatmap** – Visual correlation matrix with Pearson, Spearman, or Kendall methods and strong-correlation highlighting.
- **Data Profiling** – Column-level profiling (types, nulls, unique values, min/max/mean/std) with categorical breakdowns and pie charts.
- **Data Filtering** – Filter rows by column values or text search.
- **AI-Powered Insights** – Send a dataset summary to Google Gemini and receive:
  - Key Insights
  - Trends
  - Anomalies
  - Business Recommendations
- **Chat with Data** – Ask follow-up questions about your dataset in a conversational AI chat.
- **Download Report** – Export AI-generated insights as a Markdown file.

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- A Google Gemini API key ([get one here](https://aistudio.google.com/app/apikey))

### Steps

1. **Clone the repository** (or download the project files):
   ```bash
   git clone <repository-url>
   cd AI_Data_Analyst
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS / Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API key**:
   ```bash
   cp .env.example .env
   # Edit .env and add your Gemini API key
   ```

---

## 🚀 How to Run

1. **Set your Gemini API key** (choose one method):

   - **Option A – `.env` file** (recommended):
     ```
     GEMINI_API_KEY=your-api-key-here
     ```

   - **Option B – Environment variable:**
     ```bash
     export GEMINI_API_KEY="your-api-key-here"       # macOS / Linux
     set GEMINI_API_KEY=your-api-key-here             # Windows CMD
     $env:GEMINI_API_KEY="your-api-key-here"          # Windows PowerShell
     ```

   - **Option C – Enter it in the app sidebar** after launching.

2. **Start the application**:
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** at `http://localhost:8501` (Streamlit will open it automatically).

4. **Upload a CSV file** and explore your data! (A `sample_data.csv` is included for testing.)

---

## 📂 Project Structure

```
AI_Data_Analyst/
│
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .gitignore           # Git ignore rules
├── sample_data.csv      # Sample dataset for testing
└── README.md            # Project documentation (this file)
```

---

## 📸 Usage

1. Launch the app with `streamlit run app.py`.
2. Upload a `.csv` file using the file uploader.
3. Browse the **Dataset Preview** and quick metric cards.
4. Explore 6 analysis tabs:
   - **Statistics** – Descriptive stats and missing data chart
   - **Visualizations** – Line/bar/area charts, histograms, box plots
   - **Correlations** – Heatmap with strong-correlation table
   - **Data Profiling** – Column profiling, categorical breakdowns, data filtering
   - **AI Insights** – One-click AI analysis with downloadable report
   - **Chat with Data** – Conversational Q&A about your dataset
5. Enter your Gemini API key in the sidebar (if not set via `.env`).
6. Click **"Generate AI Insights"** or use the **Chat** tab for interactive analysis.


---

## 🚀 Deployment to Render

This application is ready to be deployed to **Render** as a Web Service.

### Quick Setup (Recommended)

1. **Push your code to GitHub.**
2. **Create a new Blueprint Instance on Render:**
   - Go to [dashboard.render.com](https://dashboard.render.com).
   - Click **New +** and select **Blueprint**.
   - Connect your GitHub repository.
   - Render will automatically detect the `render.yaml` and set up the service.
3. **Configure Environment Variables:**
   - In the Render dashboard, go to your service's **Environment** tab.
   - Add `GEMINI_API_KEY` and/or `GROQ_API_KEY`.
4. **Wait for the build to complete.** Your app will be live at a `.onrender.com` URL!

### Manual Setup (Alternative)

If you prefer not to use Blueprints:
1. Create a **New Web Service**.
2. **Build Command:** `pip install -r requirements.txt`
3. **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. **Environment Variables:** Add your API keys.

---

## 📝 License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).
