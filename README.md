# AGVS-MNER: Anchor-Guided Visual Selection for Fine-Grained Multimodal Named Entity Recognition with Multiple Images

This repository contains the official implementation of **AGVS-MNER**, an
anchor-guided framework for multimodal named entity recognition (MNER) with
multiple images. AGVS-MNER models weak-order dependencies across images,
constructs a text-guided global visual anchor, and selects entity-relevant
visual patches for token-to-patch interaction.

## Method Overview

AGVS-MNER has three stages:

1. **Weak-Order Multi-Image Context Modeling**: image-level ViT `[CLS]`
   features are augmented with learnable slot embeddings and contextualized by
   self-attention with relative position bias.
2. **Text-Guided Global Visual Aggregation**: a text-conditioned query and a
   shared learnable query aggregate image representations into a global visual
   anchor.
3. **Anchor-Guided Visual Selection and Token-to-Patch Fusion**: channel-wise
   patch calibration and anchor-guided top-K selection filter local evidence;
   image-text cross-attention and a vector gate then inject visual information
   into textual tokens before CRF decoding.

## Environment Setup

1. Use Python 3.9 or later and create an isolated environment if desired.

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download or prepare the pretrained models used by the paper:

   - `bert-base-uncased` for the textual encoder;
   - `ViT-B/16` for the visual encoder.

   By default, the training script expects them at:

   ```text
   ../pretrained_models/bert-base-uncased/
   ../pretrained_models/ViTB-16/
   ```

   You may instead pass local paths or Hugging Face model identifiers through
   `--bert-model` and `--vit-model`.

## Project Structure

```text
AGVS-MNER/
|-- dataset/
|   |-- text/                         # MNER-MI / MNER-MI-Plus split files
|   |-- images/                       # MNER-MI images
|   `-- twitter2017_images/           # Twitter-2017 images used by MNER-MI-Plus
|-- models/
|   `-- agvs_mner.py                  # AGVS-MNER model definition
|-- modules/
|   |-- dataset.py                    # Text, label, and image preprocessing
|   `-- trainer.py                    # Training, validation, and test loop
|-- ner_evaluate.py                   # Entity-level NER evaluation
|-- run.py                            # Training entry point
|-- requirements.txt
`-- README.md
```


## Data Preparation

Place the line-delimited dataset files under `dataset/text/`:

```text
MNER-MI_train.txt
MNER-MI_val.txt
MNER-MI_test.txt
MNER-UNI_train.txt
MNER-UNI_val.txt
MNER-UNI_test.txt
```

Each line is a dictionary with the following fields:

```python
{
    "text": ["Token", "sequence"],
    "label": ["O", "B-PER"],
    "images": ["image_1.jpg", "image_2.jpg"]
}
```

Use `--dataset MI` for MNER-MI and `--dataset UNI` for MNER-MI-Plus. Image
files referenced by MNER-MI records belong in `dataset/images/`; references to
Twitter-2017 images belong in `dataset/twitter2017_images/`.

## Training and Evaluation

Run training with:

```bash
python run.py \
  --dataset UNI \
  --data-path ./dataset/text \
  --image-path ./dataset/images \
  --twitter2017-image-path ./dataset/twitter2017_images \
  --bert-model ../pretrained_models/bert-base-uncased \
  --vit-model ../pretrained_models/ViTB-16 \
  --lr 2e-5 \
  --bert-lr 1e-5 \
  --vit-lr 5e-6 \
  --batch-size 8 \
  --num-epochs 15 \
  --topk-img-patches 12
```

On a CPU-only machine, add `--device cpu`. The script stores the configuration,
per-epoch metrics, best validation checkpoint, and training log in a timestamped
subdirectory of `outputs/`.

## Main Results

The paper reports the following entity-level performance (P / R / F1):

| Model | MNER-MI | MNER-MI-Plus |
| --- | ---: | ---: |
| AGVS-MNER | 77.73 / 80.43 / **79.06** | 84.52 / 83.80 / **84.16** |

## Reproducibility Notes

- The text encoder is BERT-base-uncased (hidden size 768); all its layers are
  fine-tuned.
- The visual encoder is ViT-B/16 (196 patch features per image); only its upper
  encoder layers are fine-tuned.
- The original initialization order, model parameter layout, image fallback,
  and evaluation rule are retained in this release so that the AGVS naming
  update does not change the original model behavior.

## Citation

If you use this code, please cite the AGVS-MNER paper:

```text
AGVS-MNER: Anchor-Guided Visual Selection for Fine-Grained Multimodal Named
Entity Recognition with Multiple Images.
```

Please replace this entry with the final author list, venue, and bibliographic
record once the paper is published.
