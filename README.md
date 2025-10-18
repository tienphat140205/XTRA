# XTRA: Cross-lingual Topic Modeling

Cross-lingual topic modeling framework using document embeddings and clustering.

## Requirements

```bash
pip install torch numpy scipy scikit-learn pyyaml
```

## Usage

### Run all datasets
```bash
bash run.sh
```
## Available Datasets

- `Amazon_Review` (English-Chinese)
- `ECNews` (English-Chinese) 
- `Rakuten_Amazon` (English-Japanese)

## Output

Results saved to `output/XTRA/{dataset}/{timestamp}/`:
- `T15_en.txt`, `T15_cn.txt` - Topic words
- `rst.mat` - Topic distributions and embeddings