# Original neural model (optional)

The original `chatbotApp.py`, `ecomChatbot.py`, model and intent files are preserved.
The new CRM imports the same trained model through `bot_engine.neural_answer` only
when you explicitly select **Original neural model**. Demo mode needs no model packages.

Use a separate Python 3.11 environment for the original TensorFlow 2.16 model:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-crm.txt tensorflow==2.16.1 nltk==3.8.1
python -m nltk.downloader punkt wordnet
python -m streamlit run app.py --server.address 127.0.0.1
```

Select **Original neural model** in the enquiry page. The model's 0.25 threshold
comes from the original code and has not been calibrated. Softmax scores are not
validated probabilities of correctness. The original intent data contains repeated
labels and overlapping patterns. Do not claim general accuracy from demo results.

The neural path has not been run in this build: TensorFlow was not installed in the
available environment. It may require dependency or model compatibility fixes.
Do not present a successful demo-mode run as verification of the neural model.

