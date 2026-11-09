"""Small, reproducible head-only fine-tune on 30 Food-101 examples."""
from pathlib import Path
import sys, json, torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL = 'aspis/swin-finetuned-food101'
OUT = Path('mlops-data/finetuned-swin-food101')
def main():
    ds = load_dataset('ethz/food101', split='validation', streaming=True)
    rows = []
    for i, item in enumerate(ds):
        if i >= 100: break
        rows.append({'image': item['image'].convert('RGB'), 'label': int(item['label'])})
    train, test = rows[:30], rows[30:]
    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModelForImageClassification.from_pretrained(MODEL)
    for p in model.parameters(): p.requires_grad = False
    head = getattr(model, 'classifier', None) or getattr(model, 'head', None)
    for p in head.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    model.train()
    for _ in range(3):
        for start in range(0, len(train), 5):
            batch = train[start:start+5]
            inputs = processor([x['image'] for x in batch], return_tensors='pt')
            loss = model(**inputs, labels=torch.tensor([x['label'] for x in batch])).loss
            loss.backward(); opt.step(); opt.zero_grad()
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT); processor.save_pretrained(OUT)
    Path('mlops-data/finetune-manifest.json').write_text(json.dumps({'model': MODEL, 'tuning': 30, 'evaluation': 70}), encoding='utf-8')
    print('saved', OUT)
if __name__ == '__main__': main()
