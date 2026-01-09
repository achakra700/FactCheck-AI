# 🕵️ FactCheck-AI: Narrative Consistency System

**Track A Submission | KDSH 2025**
implementation of the Narrative Consistency and Causal Reasoning system for Track A using LLM-driven analysis and vector retrieval.

## 🎯 Features

- **Memory State Propagation**: Tracks character belief evolution across EARLY/MID/LATE phases to detect impossible jumps.
- **Cross-Claim Dependency Checks**: Identifies logical "infection" where a contradiction in a root claim invalidates dependent motivations.
- **Counterfactual Reasoning**: Simulates "If-Then" behavioral expectations to verify causal consistency.
- **Contradiction Scoring**: Weighted confidence calculation (0.0-1.0) replacing binary logic.
- **LLM-Powered Analysis**: Uses Hugging Face Inference API (free tier) with Llama-3-8B
- **Vector Retrieval**: Pathway-based semantic search with open-source sentence-transformers (all-MiniLM-L6-v2).
- **6-Phase Reasoning**:
  1. **Claim Extraction & Dependency Mapping**: LLM extracts structured claims and identifies logical prerequisites.
  2. **Importance Classification**: Weights claims as MAJOR or MINOR.
  3. **Evidence Retrieval**: Finds relevant passages using vector search.
  4. **Temporal & Memory Analysis**: Tracks belief evolution and temporal constraints.
  5. **Counterfactual Simulation**: Verifies behavioral expectations.
  6. **Logical Validation & Scoring**: Final cross-claim validation and weighted scoring.

## 📁 Project Structure

```
kds_hackathon/
├── data/                    # Novel and backstory files
│   ├── story_1.txt
│   ├── backstory_1.txt
│   └── ...
├── pipeline.py              # Main orchestration script
├── llm_hf.py               # Hugging Face LLM integration
├── retrieval.py            # Vector-based evidence retrieval
├── reasoning.py            # LLM-driven reasoning pipeline
├── csv_writer.py           # Submission CSV formatter
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
└── README.md              # This file
```

## 🚀 How to Run

### 1️⃣ Set Hugging Face Token
```bash
# Linux/macOS
export HF_API_TOKEN="your_token_here"

# Windows (PowerShell)
$env:HF_API_TOKEN="your_token_here"
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run Pipeline

**Debug Mode (Recommended for testing)**
Includes confidence scores (0.0-1.0) and detailed logs.
```bash
python pipeline.py
```

**Submission Mode (Mandatory for final turn-in)**
Stripped CSV format (`story_id,prediction,rationale`) as per hackathon rules.
```bash
python pipeline.py --submission
```

---

## 🔬 Advanced Reasoning Architecture
This system implements a production-grade analysis pipeline:
- **Reasoning Graph**: Maps logical prerequisites between claims.
- **Temporal Constraint Mapping**: Validates beliefs across EARLY, MID, and LATE phases.
- **Causal Evaluation**: Distinguishes between soft tensions and fatal logical contradictions.

## 🚀 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Get a free Hugging Face API key from: https://huggingface.co/settings/tokens

Create a `.env` file:
```bash
cp .env.example .env
```

Edit `.env` and add your API key:
```
HUGGINGFACE_API_KEY=hf_your_actual_key_here
```

Alternatively, export it directly:
```bash
export HUGGINGFACE_API_KEY=hf_your_actual_key_here
```

### 3. Prepare Data

Place your story files in the `data/` directory:
- `story_1.txt`, `story_2.txt`, etc.
- `backstory_1.txt`, `backstory_2.txt`, etc.

## 🎬 Usage

### Run Full Pipeline

Process all stories in the data directory:

```bash
python pipeline.py
```

Output will be saved to `results.csv` in the submission format:
```csv
story_id,prediction,rationale
1,0,"Character's later actions violate core backstory claims without causal justification"
2,1,"Backstory aligns with all major narrative events"
```

### Environment Variables

- `HUGGINGFACE_API_KEY` or `HF_TOKEN`: Your Hugging Face API key (required)

## 📊 Pipeline Steps Explained

### Step 1: Claim Extraction
The LLM analyzes the backstory and extracts structured claims:
- Early-life events
- Formative experiences  
- Beliefs and values
- Fears and psychological traits
- Ambitions and goals
- Assumptions about the world

### Step 2: Query Generation
For each claim, the LLM generates optimized search queries to find relevant evidence in the novel.

### Step 3: Evidence Retrieval
Uses Pathway vector indexing (with keyword fallback) to retrieve the top 5 most relevant passages for each query.

### Step 4: Causal Reasoning
The LLM evaluates each claim against its evidence, checking for:
- Timeline conflicts
- Causal chain violations
- Character development inconsistencies
- World rule contradictions

### Step 5: Final Decision
Aggregates all claim evaluations into a binary decision:
- **0 (Contradict)**: If any major claim contradicts the narrative
- **1 (Consistent)**: If all claims are consistent or underdetermined

## 🔧 Customization

### Change LLM Model

Edit `llm_hf.py`:
```python
def call_llm(prompt: str, model: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
    # Change model to any supported HF model
```

### Adjust Retrieval Parameters

Edit `pipeline.py`:
```python
evidence = retrieve_evidence(queries, index, top_k=5)  # Change top_k
```

### Modify Prompt Templates

Edit the prompt strings in `reasoning.py` to customize LLM instructions.

## 🐛 Troubleshooting

**Issue**: `HUGGINGFACE_API_KEY not set`
- **Solution**: Set the environment variable or create a `.env` file

**Issue**: `Pathway not installed`
- **Solution**: Run `pip install pathway` or use the keyword fallback

**Issue**: API timeout or rate limiting
- **Solution**: Add delays between requests or reduce the number of stories processed in batch

## 📝 Output Format

The system generates `results.csv` with three columns:
- `story_id`: Story identifier
- `prediction`: Binary label (0 = Contradict, 1 = Consistent)
- `rationale`: Brief explanation (1-2 lines)

## 🎓 Academic Note

This implementation follows the principles outlined in the KDS Hackathon Track A:
- Global narrative consistency checking
- Causal and temporal reasoning
- Evidence-based decision making
- Distinction between surface plausibility and logical compatibility

## 📄 License

MIT License - Free to use and modify for the hackathon and beyond.
