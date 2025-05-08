# **Obesity - MSiD Project (Lab Part 1)**  

## **Project Overview**  
This project analyzes obesity levels based on eating habits and physical condition using data from the UCI Machine Learning Repository. The analysis includes statistical summaries, data visualization, and regression modeling to understand the relationships between different variables.  

Dataset: [Estimation of Obesity Levels Based On Eating Habits and Physical Condition](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)  

## **Installation**  
To set up the project, ensure you have Python installed (recommended version: **3.8 or higher**). Then, install the required dependencies using:  
```bash
pip install -r requirements.txt
```  

Alternatively, if using Jupyter Notebook, you may need to install Jupyter:  
```bash
pip install notebook
```  

## **Usage Instructions**  

### **Running the Data Analysis Program**

1. **Open a terminal or command prompt.**

2. **Navigate to the project directory** where your `data_analysis_main.py` is located:
   ```bash
   cd path/to/project
3. **Run the program using Python:**
   ```bash
   python data_analysis_main.py

### **Alternatively running the Jupyter Notebook**  
1. Open a terminal or command prompt.  
2. Navigate to the project directory:  
   ```bash
   cd path/to/project
   ```  
3. Launch Jupyter Notebook:  
   ```bash
   jupyter notebook
   ```  
4. Open the **`obesity_analysis.ipynb`** file in Jupyter Notebook.  
5. Run all the cells sequentially to perform data analysis and visualization.  

### **Dataset**  
- The dataset is provided in the project directory (`ObesityDataSet.csv`).  
- It contains information about individuals' eating habits, physical activity, and obesity levels.  

### **Project Structure**  
```
/project_root
│── ObesityDataSet.csv         # Data for project
│── data_analysis_main.py      # Python main to run
│── obesity_analysis.ipynb     # Jupyter Notebook with code
│── requirements.txt           # Required dependencies
│── README.md                  # Project documentation
```  
After running the script, the program will generate:  
- **Statistical summary files** containing key metrics.  
- **Folders** with visualizations.  


## Features & Analysis

### **1. Data Loading**
- A Python **function** loads the dataset using `pandas`.

### **2. Descriptive Statistics**
Computed and saved in CSV/text files:

- **Numerical Features**:
  - Mean, median, min, max, standard deviation.
  - 5th and 95th percentile.
  - Number of **missing values**.

- **Categorical Features**:
  - Number of **unique categories**.
  - Count of **missing values**.
  - **Proportion of each class**.

### **3. Data Visualization**
I use `matplotlib` and `seaborn` to create the following plots:

#### **- Boxplots**
#### **- Violinplots**
#### **- Error Bars**
#### **- Histograms**
#### **- Heatmaps**
#### **- Linear Regression Analysis**

## **Author**  
Jakub Porada  

