# ID3 Decision Tree Classifier (Streamlit App)

This project implements the **ID3 Decision Tree algorithm** in Python and provides an interactive interface using **Streamlit**.  
You can train a decision tree on a dataset, visualize the learned structure, and make predictions by selecting feature values.

---

## Features
- Upload your own CSV dataset or use the default sample dataset.
- Select the target column for classification.
- Train an ID3 decision tree interactively.
- View the generated decision tree in JSON format.
- Make predictions by selecting feature values from dropdowns.

---

## Requirements
Install dependencies using:

```bash
pip install -r requirements.txt
```
## Core Dependencies
- streamlit
- pandas
- numpy


## Run he app

streamlit run app.py

Example Dataset
The app includes a small sample dataset for the classic Play Tennis problem



Notes
- The target column should be categorical for proper training.
- If you upload a CSV, ensure it has at least one feature column and one target column.
- Predictions return "Unknown" if the input combination was not seen during training.

License
This project is released under the MIT License.



