import urllib.parse
import requests
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.metrics import confusion_matrix
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

np.random.seed(27)
torch.manual_seed(16)

print("Imports completed")

# Data Collection
class PlanetTerpDataCollector:

    BASE_URL = "https://api.planetterp.com/v1"

    def __init__(self):
        self.all_reviews = []

    # Collects a professors reviews and ratings
    def get_professor_reviews(self, name: str) -> List[Tuple[str, int]]:
      encoded = urllib.parse.quote(name)
      url = f"{self.BASE_URL}/professor?name={encoded}&reviews=true"
      response = requests.get(url)
      if response.status_code != 200:
        print("Error: Coudn't find professor")
        return []
      data = response.json()
      if 'reviews' in data and data['reviews']:
        reviews = []
        for resp in data["reviews"]:
          text = resp.get("review", "")
          rating = resp.get("rating", None)
          if text and rating is not None:
            reviews.append((text, rating))
        print(f"Found {len(reviews)} reviews for {name}")
        return reviews
      else:
        print("Error: No reviews found")
        return []

    # Make a dataframe of all 5 professor's reviews and ratings
    def get_all_reviews(self, prof_names: List[str]) -> pd.DataFrame:

      data = []
      for name in prof_names:
        reviews = self.get_professor_reviews(name)
        for text, rating in reviews:
          data.append({
              'professor' : name,
              'review' : text,
              'rating' : rating
          })

        time.sleep(.5)
      df = pd.DataFrame(data)
      return df

# These are the professors I used
PROFESSOR_NAMES = [
    "Justin Wyss-Gallifent",
    "Nelson Padua-Perez",
    "Larry Herman",
    "Ilchul Yoon",
    "Fawzi Emad"
]


scraper = PlanetTerpDataCollector()
reviews_df = scraper.get_all_reviews(PROFESSOR_NAMES)
print(f"\nRating distribution:\n{reviews_df['rating'].value_counts().sort_index()}")
print(f"Total Review: {len(reviews_df)}")

if len(reviews_df) == 0:
  print("No reviews found")
else:
  print("Reviews Collected")

# This is where I do data cleaning
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
  df = df.dropna(subset=['review', 'rating'])
  df = df[df['review'].str.strip().str.len() > 0]
  df['rating'] = df['rating'].astype(int)
  df = df.drop_duplicates(subset=['review'])
  return df

reviews_df = preprocess_data(reviews_df)
print("Data Cleaned")

train_df, temp_df = train_test_split(reviews_df, test_size=0.15, random_state=48, stratify=reviews_df['rating'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=48, stratify=temp_df['rating'])

print(f"\n Train set: {len(train_df)} reviews")
print(f" Validation set: {len(val_df)} reviews")
print(f" Test set: {len(test_df)} reviews")

# Get dataset ready for PyTorch
class ReviewDataset(Dataset):
  def __init__(self, reviews : List[str], ratings : List[int], tokenizer, max_length : int = 512):
    self.reviews = reviews
    self.ratings = ratings
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.reviews)

  # This is used to return a training example (review + rating) at an index
  def __getitem__(self, index):
    review = str(self.reviews[index])
    rating = self.ratings[index] - 1

    encoding = self.tokenizer(
        review,
        add_special_tokens=True,
        max_length=self.max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return {
        'input_ids' : encoding['input_ids'].flatten(),
        'attention_mask' : encoding['attention_mask'].flatten(),
        'labels' : torch.tensor(rating, dtype=torch.long)
    }

# This is where I initialize the tokenizer and model
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=5,
    problem_type="single_label_classification",
)
print("Model and Tokenizer Initialized")

# Organize the datasets for PyTorch
train_dataset = ReviewDataset(
    train_df['review'].tolist(),
    train_df['rating'].tolist(),
    tokenizer
)

val_dataset = ReviewDataset(
    val_df['review'].tolist(),
    val_df['rating'].tolist(),
    tokenizer
)

test_dataset = ReviewDataset(
    test_df['review'].tolist(),
    test_df['rating'].tolist(),
    tokenizer
)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# See how well the model is doing
def compute_metrics(eval_pred):
  predictions, labels = eval_pred
  predictions = np.argmax(predictions, axis=1)
  accuracy = accuracy_score(labels, predictions)
  mae = mean_absolute_error(labels, predictions)
  mse = mean_squared_error(labels, predictions)
  rmse = np.sqrt(mse)
  return {
      'accuracy' : accuracy,
      'mae' : mae,
      'rmse' : rmse
  }

# Initialize the trainer and train!
trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=val_dataset,
      compute_metrics=compute_metrics
)

print("Training Started...")
trainer.train()
print("Training Complete!")

# Evaluation and Results
test_results = trainer.evaluate(test_dataset)
print("\nTest Set Results:")
print(f"  Accuracy: {test_results['eval_accuracy']:.4f}")
print(f"  MAE: {test_results['eval_mae']:.4f}")
print(f"  RMSE: {test_results['eval_rmse']:.4f}")

predictions = trainer.predict(test_dataset)
predicted_ratings = np.argmax(predictions.predictions, axis=1) + 1
true_ratings = test_df['rating'].values

# SAMPLE PREDICTIONS
def show_sample_predictions(test_df, predicted_ratings, true_ratings, num_samples=10):
    print("\n")
    print(f"RANDOM SAMPLE PREDICTIONS: (n={num_samples})")


    # Get random indices
    sample_indices = np.random.choice(len(test_df), size=min(num_samples, len(test_df)), replace=False)

    for i, idx in enumerate(sample_indices, 1):
        # Get the actual dataframe index
        df_idx = test_df.index[idx]
        review = test_df.loc[df_idx, 'review']
        true_rating = true_ratings[idx]
        pred_rating = predicted_ratings[idx]

        print(f"\n--- Sample {i} ---")
        print(f"Review: {review}")
        print(f"True Rating: {true_rating} stars")
        print(f"Predicted Rating: {pred_rating} stars")
        print("-" * 80)

    print("\n" + "="*80 + "\n")

# Display sample predictions
show_sample_predictions(test_df, predicted_ratings, true_ratings, num_samples=10)

#Validation Accuracy Over Time
print("Generating Validation Accuracy Plot")
history = trainer.state.log_history
eval_accuracy = [x['eval_accuracy'] for x in history if 'eval_accuracy' in x]

plt.figure(figsize=(10, 6))
plt.plot(eval_accuracy, label='Validation Accuracy', marker='o', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Confusion Matrix
print("Generating Confusion Matrix")
conf_matrix = confusion_matrix(true_ratings, predicted_ratings)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(1, 6), yticklabels=range(1, 6))
plt.xlabel('Predicted Ratings')
plt.ylabel('True Ratings')
plt.title('Confusion Matrix: True Vs Predicted Ratings')
plt.tight_layout()
plt.show()
