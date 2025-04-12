import torch
from transformers import BertModel

class CustomBERTClassifier(torch.nn.Module):
    def __init__(self, bert_model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext", num_labels=24):
        super(CustomBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name, output_attentions=True)
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
