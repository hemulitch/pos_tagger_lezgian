## Part of Speech (POS) Tagger for Lezgian language

This is a web application for part-of-speech tagging of sentences in Lezgian, powered by a transformer-based sequence tagger trained with Flair. It provides a simple browser interface where users can enter a sentence and view predicted POS tags for each word.

#### How to use?

1. Train your model
   
`train/pos_data/` contains annotated data

Run:
```
cd train
python3 train.py
```

The trained model will be saved to `model`

2. Run the web app

Install dependencies and start the server:

```
pip install -r requirements.txt
uvicorn app:app --reload
```

3. Docker support

Build and run the app with Docker:

```
docker build -t pos_tagger_lezgian .
docker run -p 8000:8000 pos_tagger_lezgian
```

#### Example

Enter a sentence in the web form, e.g., _За мад геждач._ (translation: "I won't be late again/anymore").

See a list of word–POS tag pairs, e.g.:

За - PRN

мад - ADV

геждач - VERB

. - PUNCT

#### Requirements

See `requirements.txt` for details.
