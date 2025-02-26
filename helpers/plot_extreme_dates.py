"""
Module for generating a histogram of extreme weather event counts per year.

### **Overview**
This script reads a file containing extreme event dates in the `YYYYMMDD` format, extracts the 
years, counts occurrences per year, and generates a histogram to visualize the data distribution.

### **Workflow**
1. **Load Data:**
   - Reads extreme weather event dates from `extreme_dates.txt`.
   - Filters only numeric date entries.

2. **Process Data:**
   - Extracts years from `YYYYMMDD` formatted dates.
   - Computes the count of extreme events per year.

3. **Generate and Save Histogram:**
   - Plots a bar chart showing the count of extreme weather events per year.
   - Saves the plot as an image file.

### **File Dependencies**
- **Input:** `../data/extreme_dates/extreme_dates.txt`
  - A text file containing extreme event dates, one per line in `YYYYMMDD` format.
- **Output:** `../data/extreme_dates/extreme_dates_histogram.png`
  - A histogram image showing the yearly count of extreme events.
"""
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "../data/extreme_dates/extreme_dates.txt"
OUTTPUT_FILE = "../data/extreme_dates/extreme_dates_histogram.png"

# Load the file
with open(INPUT_FILE, "r") as file:
    dates = [line.strip() for line in file if line.strip().isdigit()]

# Extract years from YYYYMMDD format
years = [date[:4] for date in dates]

# Convert to DataFrame
df = pd.DataFrame(years, columns=["Year"])

# Count occurrences per year
year_counts = df["Year"].value_counts().sort_index()

# Plot histogram
plt.figure(figsize=(12, 6))
plt.bar(year_counts.index, year_counts.values, color='skyblue', edgecolor='black')
plt.xlabel("Year")
plt.ylabel("Count")
plt.title("Histogram of Extreme Weather Date Counts per Year")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save and display the plot
plt.savefig(OUTTPUT_FILE)  # Saves the plot
