"""Evaluate the production classifier without mixing tuning and test data."""
from pathlib import Path
import argparse, json, tempfile, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import load_dataset
from PIL import Image
from model import classify, classify_tta, classify_multicrop

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='evaluation-results.json')
    p.add_argument('--tta', action='store_true')
    p.add_argument('--multicrop', action='store_true')
    p.add_argument('--tuning-count', type=int, default=20)
    p.add_argument('--evaluation-count', type=int, default=50)
    args = p.parse_args()
    ds = load_dataset('ethz/food101', split='validation', streaming=True)
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for i, item in enumerate(ds):
            if i >= args.tuning_count + args.evaluation_count: break
            path = Path(td) / f'{i}.jpg'
            image = item['image'].convert('RGB')
            image.save(path, quality=95)
            result = classify_multicrop(path) if args.multicrop else (classify_tta(path) if args.tta else classify(path))
            top = result['predictions'][0]['label'].lower().replace('_', ' ')
            truth = ds.features['label'].int2str(item['label']).lower().replace('_', ' ')
            rows.append({'index': i, 'split': 'tuning' if i < args.tuning_count else 'evaluation',
                         'truth': truth, 'prediction': top,
                         'correct': top == truth, 'model_version': result['model_version']})
    test = [r for r in rows if r['split'] == 'evaluation']
    report = {'dataset': 'ethz/food101', 'license_note': 'research/non-commercial use approved by project owner',
              'tuning_count': args.tuning_count, 'evaluation_count': args.evaluation_count,
              'accuracy': sum(r['correct'] for r in test) / len(test), 'rows': rows}
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('dataset','tuning_count','evaluation_count','accuracy')}, ensure_ascii=False))

if __name__ == '__main__': main()
