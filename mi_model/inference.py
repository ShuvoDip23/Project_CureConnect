import torch
from transformers import BertTokenizer, BertModel
import os

# Dictionary mapping numbers to disease names
disease_dict = {
    0: 'Psoriasis',
    1: 'Varicose Veins',
    2: 'Typhoid',
    3: 'Chicken pox',
    4: 'Impetigo',
    5: 'Dengue',
    6: 'Fungal infection',
    7: 'Common Cold',
    8: 'Pneumonia',
    9: 'Dimorphic Hemorrhoids',
    10: 'Arthritis',
    11: 'Acne',
    12: 'Bronchial Asthma',
    13: 'Hypertension',
    14: 'Migraine',
    15: 'Cervical spondylosis',
    16: 'Jaundice',
    17: 'Malaria',
    18: 'Urinary tract infection',
    19: 'Allergy',
    20: 'Gastroesophageal reflux disease',
    21: 'Drug reaction',
    22: 'Peptic ulcer disease',
    23: 'Diabetes'
}

# Function to get disease name from number
def get_disease_name(disease_number):
    return disease_dict.get(disease_number, "Invalid number")  # Returns "Invalid number" if not found

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model_and_tokenizer", "model.pth")
TOKENIZER_PATH = os.path.join(BASE_DIR, "model_and_tokenizer", "tokenizer")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)

# Define the model class
class CustomBERTClassifier(torch.nn.Module):
    def __init__(self, bert_model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext", num_labels=24):
        super(CustomBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)  # Removed output_attentions=True
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.bert.config.hidden_size, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, num_labels)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        avg_hidden_state = torch.mean(outputs.last_hidden_state, dim=1)
        logits = self.classifier(avg_hidden_state)
        return logits

# Load the model
model = CustomBERTClassifier()

# Load model weights correctly
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
print("Model loaded successfully")
model.eval()  # Set to evaluation mode

# Function to predict a condition
def predict_condition(text):
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        predicted_class = torch.argmax(logits, dim=1).item()
        predicted_class = int(predicted_class)
    return get_disease_name(predicted_class)

