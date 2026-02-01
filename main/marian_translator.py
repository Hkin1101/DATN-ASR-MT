import torch
from transformers import MarianMTModel, MarianTokenizer

class MarianTranslator:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-vi", device=None):
        print(f"Dang load model MT: {model_name}...")
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        
        if device:
            self.model.to(device)
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)

    def translate(self, text_list, batch_size=8):
        translated_results = []
        
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i : i + batch_size]
            
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            with torch.no_grad():
                translated = self.model.generate(**inputs)
            
            batch_preds = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]
            translated_results.extend(batch_preds)
            
        return translated_results