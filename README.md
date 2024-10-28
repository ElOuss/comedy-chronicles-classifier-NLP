# 🎭 **Comedic Video Classifier** - NLP Text Classification Project

This repository presents a solution for classifying video titles as comical or non-comical. The focus is on developing a reproducible pipeline that not only classifies but also documents experimentation steps, ensuring a clear understanding of our model’s journey and performance.

## 📚 **Project Overview**

This assignment aims to create a model to predict if a video title relates to a comedy segment. The primary emphasis is on the methodological approach, exploring feature engineering, evaluating baseline models, and iterating to improve results. 

In compliance with task instructions:
1. **Problem Formulation**: Define the problem as text classification (i.e., `video_name` → `is_comic`).
2. **Baseline & Experimentation**: Establish a baseline, then iterate with different features and models.
3. **Experiment Tracking**: Document each experiment's rationale, methods, and conclusions in a structured report.
4. **Clean Codebase**: Ensure the codebase is organized, reproducible, and well-documented.

## 🚀 **Getting Started**

Clone the repository and install dependencies:


```bash
git clone https://github.com/ElOuss/comedy-chronicales-classifier-NLP.git
cd comedy-chronicales-classifier-NLP
pip install -r requirements.txt
```



### **Dataset**

The dataset consists of a CSV file with the following columns:
- **video_name**: Title of the video.
- **is_comic**: Binary label indicating if the video is comedy-related (`1`) or not (`0`).

## 🧩 **Pipeline Structure**

The `src/main.py` provides a main entry point for the pipeline with three essential functions:
1. **Train** - Prepares and trains the model.
2. **Predict** - Uses the trained model for predictions on new data.
3. **Evaluate** - Evaluates the model using cross-validation.

#### 1. Training the Model
```bash
python src/main.py train --input_filename=data/raw/train.csv --model_dump_filename=models/model.json
```

- **Loads the data**: Reads the input CSV for training.
- **Feature Engineering**: Transforms `video_name` with `CountVectorizer` into one-hot encoded features.
- **Model Training**: Fits a model (e.g., Random Forest or linear model) to classify the video as comedic or not.
- **Model Saving**: Dumps the trained model to `model_dump`.

#### 2. Prediction
```bash
python src/main.py predict --input_filename=data/raw/test.csv --model_dump_filename=models/model.json --output_filename=data/processed/prediction.csv
```

- **Loads the model**: Uses the trained model from `model_dump`.
- **Pre-processes New Data**: Ensures word index consistency with the training set for correct encoding.
- **Output**: Saves predictions to a specified output CSV.

#### 3. Evaluation
```bash
python src/main.py evaluate --input_filename=data/raw/train.csv
```

- **Cross-Validation**: Evaluates the model using cross-validation on the training set, outputting accuracy and other metrics.

## 🧠 **Tasks and Experimentation**

### Part 1: Text Classification
1. **Problem Definition**: Predict `is_comic` from `video_name` using non-neural methods.
2. **Baseline Model**: Establish a baseline with basic features (e.g., one-hot encoding) and a simple classifier.
3. **Feature Engineering**:
   - **Text Transformations**: Experiment with `NLTK` for stemming and stop-word removal.
   - **Vectorization**: Test different `CountVectorizer` parameters (e.g., `min_df`, `max_df`).
4. **Model Selection**: Try Naive Bayes, Decision Trees, and Random Forest models.

### Iterative Experiments
- Run multiple iterations to experiment with various feature sets and model configurations, each time recording insights and performance metrics.
- Track each experiment in a report, noting the goal, approach, and conclusions.

## 📜 **Summary Report**

The [Project Report](report.md) includes:
- Problem definition and approach.
- Description of each experiment.
- Key conclusions, challenges, and lessons learned.

## 🛠️ **Usage**

1. **Train the Model**:
   ```bash
   python src/main.py train --input_filename=data/raw/train.csv --model_dump_filename=models/model.json
   ```

2. **Predict with the Model**:
   ```bash
   python src/main.py predict --input_filename=data/raw/test.csv --model_dump_filename=models/model.json --output_filename=data/processed/prediction.csv
   ```

3. **Evaluate the Model**:
   ```bash
   python src/main.py evaluate --input_filename=data/raw/train.csv
   ```

## 👏 **Contributors**

Special thanks to the NLP course at ESGI for guidance and inspiration throughout this project.
@Smveer

## ⚖️ **License**

This project is licensed under the MIT License. Feel free to use, adapt, and experiment!
