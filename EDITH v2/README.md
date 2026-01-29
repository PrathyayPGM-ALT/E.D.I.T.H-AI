# E.D.I.T.H - AI Assistant with RAG

**Even Dead, I'm The Hero**

A PyTorch-based AI assistant that can answer ANY question while maintaining Tony Stark's tactical E.D.I.T.H personality.

## Quick Start

```bash
# Run E.D.I.T.H
python edith_rag_v2.py

# Or double-click
START_EDITH.bat
```

## What Makes This Special?

### ❌ What We Tried First (Failed)
- Training from scratch → Not enough data
- Fine-tuning GPT-2 → Produced garbage ("d3c d3c d3c")
- Hardcoded patterns → Limited to pre-written responses

### ✅ Final Solution: RAG System

**RAG = Retrieval Augmented Generation**

Combines:
1. **Your knowledge files** (E.D.I.T.H-specific info)
2. **Pre-trained FLAN-T5** (general knowledge about everything)
3. **Smart prompting** (maintains tactical personality)

Result: Handles unlimited questions with accurate responses.

## How It Works

```
Question → Search knowledge files → Build prompt with context
       → Pre-trained model generates answer → Response
```

### Example Questions It Can Handle

**E.D.I.T.H-Specific:**
- "What are your capabilities?"
- "Who created you?"
- "What can you do?"

**General Knowledge:**
- "How does photosynthesis work?"
- "Tell me about World War 2"
- "How do I cook pasta?"
- "What is quantum physics?"
- "Which programming language should I learn?"

**Mixed:**
- "Can you help me understand DNA?"
- "What tactical advice do you have for productivity?"

## Files

### Main System
- `edith_rag_v2.py` - **USE THIS** - FLAN-T5 based RAG (instruction-tuned)
- `START_EDITH.bat` - Launch script

### Knowledge Base
- `knowledge/` - Your E.D.I.T.H knowledge files
  - capabilities.md
  - faq.md
  - conversations.md
  - identity.md
  - important.md

### Documentation
- `RAG_EXPLANATION.md` - Detailed explanation of how RAG works
- `README.md` - This file

### Alternative/Legacy
- `edith_rag.py` - GPT-2 medium version (slower, okay quality)
- `edith_rag_fast.py` - Failed GPT-2 attempt (not instruction-tuned)
- `edith_chat.py` - Hardcoded pattern matching (limited)
- `train.py`, `model.py` - Training scripts (not needed for RAG)

## Requirements

```bash
pip install torch transformers
```

## Model Options

Edit `edith_rag_v2.py` line 18 to change models:

```python
# Fast but lower quality
model_name="google/flan-t5-small"  # 80M params

# Balanced (default)
model_name="google/flan-t5-base"   # 250M params

# Best quality but slower
model_name="google/flan-t5-large"  # 780M params
```

## Why RAG Instead of Training?

| Approach | Data Needed | Quality | Can Answer Anything? |
|----------|-------------|---------|---------------------|
| Train from scratch | Gigabytes | Poor (not enough data) | ❌ No |
| Fine-tune GPT-2 | Gigabytes | Poor (not enough data) | ❌ No |
| Hardcoded patterns | Manual work | Good for known topics | ❌ No |
| **RAG** | **Your files only** | **Excellent** | **✅ YES** |

## How To Use

1. Run `python edith_rag_v2.py`
2. Ask ANY question
3. Get accurate responses in E.D.I.T.H's tactical style

```
You: What are your capabilities?
E.D.I.T.H: [Uses capabilities.md] I provide tactical intelligence,
threat assessment, real-time data analysis...

You: How does gravity work?
E.D.I.T.H: [Uses FLAN-T5's knowledge] Gravity is the fundamental
force of attraction between masses...

You: How do I make coffee?
E.D.I.T.H: [Uses FLAN-T5's knowledge + tactical style] Coffee
preparation protocol: boil water, add grounds at 1:15 ratio...
```

## Technical Details

- **Model**: FLAN-T5 (instruction-tuned seq2seq model from Google)
- **Framework**: PyTorch + Hugging Face Transformers
- **Retrieval**: Keyword-based search through knowledge chunks
- **Generation**: Conditional generation with personality prompting
- **No Training Required**: Uses pre-trained weights

## Advantages

✅ **Unlimited knowledge** - Answers any question
✅ **No training needed** - Works immediately
✅ **Uses your files** - E.D.I.T.H-specific when needed
✅ **Maintains character** - Always tactical style
✅ **Accurate** - Instruction-tuned model
✅ **Offline** - Runs locally once downloaded

## What Happened to Training?

We discovered that training/fine-tuning with small datasets produces gibberish. RAG sidesteps this by using a pre-trained model that already knows everything, and just guides its personality through prompting.

This is how modern AI assistants work - they don't train on your data, they retrieve it and use it as context.

---

**E.D.I.T.H is ready to assist with any mission.**
