
# Flight Price Prediction Dashboard

This repository contains a dashboard for exploring and predicting flight prices. It uses Python libraries such as Pandas, Dash, PySpark, and Matplotlib.

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

3. Run the Dash application:
   ```bash
   python tourism_analysis_dashboard.py
   ```

4. Open a web browser and go to:
   ```
   http://127.0.0.1:8050/
   ```

---

## Features

- **Data Preprocessing**:  
   The data is cleaned and prepared for analysis using PySpark. It includes feature extraction, handling missing values, and encoding categorical variables.  

- **Visualization**:  
   The dashboard includes interactive and static visualizations for:  
   - Monthly spending trends  
   - Top 10 popular destinations  
   - Peak hours for flights  

- **Prediction**:  
   A Linear Regression model is used to predict flight prices based on features such as departure time, duration, and total stops.

- **Evaluation**:  
   Model performance is evaluated using metrics like RMSE (Root Mean Square Error) and RÂ² Score.

---

## Folder Structure

```plaintext
BigDataProject/
â”œâ”€â”€ tourism_analysis_dashboard.py       # Main application file
â”œâ”€â”€ Data_Train.xlsx                     # Dataset
â”œâ”€â”€ requirements.txt                    # List of required packages
â”œâ”€â”€ README.md                           # Project documentation
```

---

## Contribution

This project was created and developed by **Nour Il Islem Hermassi**. Feel free to explore and modify it for educational or personal use.
