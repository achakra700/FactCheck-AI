# Evidence Re-Ranking Implementation

## Purpose
Improves the quality of retrieved evidence by using LLM-based re-ranking to prioritize passages with the strongest causal relevance, temporal constraints, and logical incompatibilities.

## How It Works

### Input
- **Claim**: A specific backstory claim to evaluate
- **Retrieved Chunks**: Top-K passages from vector search (e.g., 5 passages)

### Re-Ranking Criteria
The LLM ranks chunks based on:
1. **Direct Causal Evidence**: Does the passage show cause-effect relationships?
2. **Temporal Constraints**: Does it constrain what could have happened earlier?
3. **Logical Incompatibility**: Does it directly contradict the claim?

### Output
Chunks reordered from most to least relevant for consistency evaluation.

## Example

**Claim**: "He never trusted anyone in a position of power."

**Before Re-Ranking** (vector similarity order):
1. Geographic description of town
2. Handshake with CEO
3. Father's work history
4. Cooperation with management
5. Direct statement of trust

**After Re-Ranking** (causal relevance order):
1. Direct statement of trust (strongest contradiction)
2. Cooperation with management (causal constraint)
3. Handshake with CEO (behavioral contradiction)
4. Father's work history (weak context)
5. Geographic description (irrelevant)

## Integration

The re-ranking happens automatically in Step 3A if `use_reranking=True`:

```python
evidence = retrieve_evidence(queries, index, top_k=5, use_reranking=True)
```

## Performance Notes

- Adds one LLM call per claim (with retrieved evidence)
- Increases API usage but significantly improves evidence quality
- Can be disabled by setting `use_reranking=False` for faster processing

## Benefits

1. **Better Evidence Quality**: Most relevant passages appear first
2. **Improved Accuracy**: Temporal analysis uses more meaningful evidence
3. **Clearer Rationales**: Final decisions cite the strongest contradictions
