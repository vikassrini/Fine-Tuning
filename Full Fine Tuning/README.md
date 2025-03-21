#  Fully Fine-Tuning Bert for Question Answering 

This project demonstrates how to **fine-tune a BERT model (bert-base-uncased)** on a custom **Question Answering (QA)** dataset using **Hugging Face Transformers**, **PyTorch**, and **Trainer API**.

The dataset should contain questions, contexts, and character-level start and end indices for the answers. After training, the model can answer domain-specific questions accurately.

---

## Dependencies

Make sure to install all necessary packages:

```bash
pip install transformers datasets peft accelerate torch
```

---

## Dataset Format & Source

The dataset used for fine-tuning is a custom-built **Question Answering (QA)** dataset in JSON format:

```json
[
  {
    "question": "What is the capital of the UK?",
    "context": "London is the capital of the United Kingdom...",
    "answer_text": "guided tour",
    "answer_start_index": 0,
    "answer_end_index": 6
  },
  ...
]
```

Each entry includes:
- `question`: The question to be answered.
- `context`: A paragraph containing the answer.
- `answer_text`: The actual answer string (used during preprocessing).
- `answer_start_index`: The character index where the answer begins in the context.
- `answer_end_index`: The character index where the answer ends in the context.

### 📦 Dataset Source

This dataset was **custom curated** by scraping activity descriptions from **Tripadvisor's London activities page**:  
👉 [Tripadvisor London Activities](https://www.tripadvisor.in/Attractions-g186338-Activities-London_England.html)

Using a Python script, the scraped activity data was **converted into QA pairs** to create training examples suitable for fine-tuning BERT on a domain-specific task like **tourism and city activities**.


---

## Training Pipeline Overview

1. **Tokenizer Initialization:** Using `BertTokenizerFast` for tokenizing context-question pairs with offset mapping.
2. **Preprocessing:** Convert character-level answer spans to token indices.
3. **Model Setup:** Load `BertForQuestionAnswering` and push to CUDA if available.
4. **Training:** Fine-tune all parameters using `Trainer` and `TrainingArguments`.
5. **Saving Model:** Save the trained model and tokenizer for future inference.
6. **Inference:** Load the model using `pipeline` and run question-answering on new data.

---

## Result Example

```json
{
  "score": 0.9995,
  "start": 67,
  "end": 78,
  "answer": "guided tour"
}
```

---

## Notes

- Ensure your input dataset has **accurate character-level answer spans**.
- The pipeline uses **full fine-tuning**, updating all model weights.
- This example uses `bert-base-uncased`, but any transformer model with a QA head can be used.

---
