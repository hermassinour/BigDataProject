
# Flight Price Prediction Dashboard

This repository contains a dashboard for exploring and predicting flight prices. It uses Python libraries such as Pandas, Dash, Matplotlib, and Scikit-learn.

---

## Prerequisites

- Python 3.8 or later
- pip (Python package manager)
- PySpark installed (tested with version 3.x)

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/hermassinour/BigDataProject.git
   ```

2. Navigate to the project directory:
   ```bash
   cd BigDataProject
   ```

3. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

4. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

5. Ensure PySpark is installed in your environment. If it isn't already, install it:
   ```bash
   pip install pyspark
   ```

---

## Running the Project

1. Ensure the dataset file `Data_Train.xlsx` is in the root directory of the project.

2. Activate the virtual environment (if not already active):
   ```bash
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Run the Dash application using PySpark:
   ```bash
   ./bin/spark-submit tourism_analysis_dashboard.py
   ```

4. Open a web browser and go to:
   ```
   http://127.0.0.1:8050/
   ```

---

## Features

- **Data Preprocessing**: Cleans and prepares the flight price dataset.
- **Visualization**: Interactive graphs and visual summaries of trends.
- **Prediction**: Predicts flight prices using a Linear Regression model.

---

## Folder Structure

```plaintext
BigDataProject/
├── tourism_analysis_dashboard.py       # Main application file
├── Data_Train.xlsx                     # Dataset
├── requirements.txt                    # List of required packages
├── README.md                           # Project documentation
```

---

## Contribution

This project was created and developed by **Nour Il Islem Hermassi**. Feel free to explore and modify for educational or personal use.