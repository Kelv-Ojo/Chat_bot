# E-Commerce Chatbot

A Python chatbot with a Streamlit interface, NLTK text processing, and a TensorFlow/Keras intent classifier. It matches messages to predefined intents and selects a response from the corresponding response list.

## How it works

1. Tokenize and lemmatize the user's message.
2. Encode it as a bag-of-words vector.
3. Predict intent probabilities with the trained neural network.
4. Select a predefined response for the highest-ranked intent above the 0.25 threshold.
5. Return a fallback message when no intent passes the threshold.

Responses come from `intents.json`; the app does not generate free-form answers or connect to a store backend.

## Run locally

The original project specifies Python 3.9.7. Dependency installation and the bundled model have not been verified across operating systems.

```bash
git clone https://github.com/Kelv-Ojo/Chat_bot.git
cd Chat_bot
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies and launch the app from the repository root:

```bash
python -m pip install -r requirements.txt
python -m streamlit run chatbotApp.py
```

The repository includes a trained model and its vocabulary/class files, so training is optional for an initial launch. The scripts download NLTK resources on startup and may need internet access on first use.

## Train or customize

Edit the patterns and responses in [intents.json](intents.json), then run:

```bash
python ecomChatbot.py
```

Training runs for 200 epochs and overwrites `chatbot_model.h5`, `words.pkl`, and `classes.pkl`. Keep these three generated files together and restart Streamlit after retraining.

Model architecture and training parameters are defined in `ecomChatbot.py`. Prediction and interface behavior are defined in `chatbotApp.py`.

## Repository contents

| File | Purpose |
| --- | --- |
| [chatbotApp.py](chatbotApp.py) | Streamlit interface and intent prediction |
| [ecomChatbot.py](ecomChatbot.py) | Training pipeline |
| [intents.json](intents.json) | Intent tags, example patterns, and responses |
| chatbot_model.h5 | Saved neural network |
| words.pkl / classes.pkl | Vocabulary and intent labels |
| [requirements.txt](requirements.txt) | Pinned Python dependencies |
| user.png / aiChatbot.png | Chat interface icons |

## Troubleshooting

- **Missing model, vocabulary, or images:** Launch from the repository root; the scripts use relative paths.
- **Dependency installation fails:** The requirements include platform-specific TensorFlow packages. Compatibility needs checking for your operating system and Python version; a clean install has not yet been validated.
- **NLTK resource download fails:** Check network access and the error's resource name.
- **Unexpected responses:** Review intent examples and retrain after changing patterns. No held-out accuracy or benchmark is documented.

## Current limitations

The chatbot demonstrates intent classification and a conversational interface. It does not provide live order tracking, payments, inventory integration, or a measured production-quality accuracy guarantee.

## Related work

- [Sales Overview Dashboard](https://github.com/Kelv-Ojo/Dashboard-sales)
- [Heart Disease Data Analysis](https://github.com/Kelv-Ojo/Heart-disease)
