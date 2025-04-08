📊 Dynamic Data Visualization & AI-Driven Insights Dashboard

A Data Science course project that brings your datasets to life with interactive visualizations and intelligent insights. This web-based dashboard allows users to upload datasets, explore dynamic Plotly-powered visualizations, and read AI-generated conclusions — all in a clean, modern interface inspired by Power BI and ChartHub.



 🚀 Features

- 📁 Dataset Upload – Upload any CSV dataset and explore it visually.
- 📊 Interactive Visualizations – Dynamic charts (Heatmap, Bar, Pie, Line, Scatter, Area, Bubble, Box, Radar) rendered using Plotly for real-time exploration.
- 🤖 AI-Generated Insights – Automated conclusions are generated for each chart to help interpret trends, correlations, and anomalies.
- 🎨 Modern UI Design – Smooth, Gen Z-inspired aesthetic with responsive tile layout and hover effects.
- 📈 Final Dataset Summary – Summarizes the dataset with key stats and patterns at the end of the dashboard.

---

 🧱 Built With

- Python (Flask) – Backend and routing logic
- Plotly – Dynamic and interactive visualizations
- Pandas & NumPy – Data wrangling and preprocessing
- Mistral AI – Generates chart conclusions based on data
- HTML, CSS, JavaScript (Bootstrap) – Frontend UI and layout
- Jinja2 – Templating for dynamic content


 🖼️ UI Preview



> A tile-style dashboard interface with zoom-out graph effects and hover-based AI insights.


 📂 How to Run Locally

1. Clone the Repository:**
   bash
   git clone https://github.com/yourusername/dynamic-viz-ai-dashboard.git
   cd dynamic-viz-ai-dashboard
   

2. Set up a virtual environment (optional but recommended):**
   bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   

3. Install dependencies:**
   bash
   pip install -r requirements.txt
   

4. Run the Flask App:**
   bash
   python test_aura_v2.py
 

5. Open in Browser:**
   
   http://127.0.0.1:5000/
   



## 🧪 Sample Dataset

You can try out the dashboard with any CSV file of your choice, or use the provided `sample_dataset.csv` in the repo.



 📌 Project Structure


📁 static/
📁 templates/
├── index.html
├── visualization.html
📄 test_aura_v2.py
📄 requirements.txt
📄 README.md




💡 Future Improvements

- Add support for time-series forecasting
- Export visualizations and reports
- User authentication and dataset history
- Dark/light mode toggle



📚 Course Context

This project was developed as part of a **Data Science course**, with a focus on real-world visualization techniques and interpretability using AI tools.




