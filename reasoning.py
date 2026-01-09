"""
Advanced Reasoning with Importance Weighting + Temporal Constraints.
Streamlined for production use.
"""

from typing import Dict, List
from llm_hf import call_llm
from retrieval import search_index


# STEP 1: Extract Claims
def extract_claims(backstory_text: str) -> str:
    """Extract structured claims from backstory."""
    prompt = f"""Extract all explicit and implicit claims from the backstory.

Organize into:
1. Early-life events
2. Formative experiences
3. Beliefs and values
4. Fears and psychological traits
5. Ambitions and goals
6. Assumptions about how the world works

Rules:
- Numbered bullet points
- Each must be a factual claim
- Do not invent

Backstory:
{backstory_text}
"""
    return call_llm(prompt, max_new_tokens=800)


# STEP 2: Classify Claim Importance
def classify_importance(claims_text: str) -> str:
    """Classify claims as MAJOR or MINOR."""
    prompt = f"""Classify each claim as MAJOR or MINOR.

MAJOR:
- Shapes identity, values, motivations
- Affects causal chains
- Influences multiple future events

MINOR:
- Descriptive, stylistic, incidental

Output:
Claim ID
Importance: MAJOR/MINOR
Reason: 1 line

Claims:
{claims_text}
"""
    return call_llm(prompt, max_new_tokens=1000)


# STEP 2.5: Map Claim Dependencies
def map_claim_dependencies(claims_text: str) -> str:
    """
    Identify logical dependencies between claims.
    (e.g., Claim 5 is a motivation based on the event in Claim 2).
    """
    prompt = f"""You are mapping logical dependencies between character backstory claims.

For each claim, identify if it depends on another claim (a "Prerequisite").
A dependency exists if:
- Claim B is a motivation/belief caused by event Claim A.
- Claim B is a goal intended to resolve trauma in Claim A.
- Claim B is an assumption based on experience in Claim A.

Output format:
Claim ID -> Prerequisite ID (Reason)
...

Claims:
{claims_text}
"""
    return call_llm(prompt, max_new_tokens=1000)


# STEP 3: Generate Retrieval Queries
def generate_queries(claims_text: str) -> str:
    """Generate search queries for each claim."""
    prompt = f"""For each claim, generate a short search query
to retrieve relevant evidence from the novel.

Output:
Claim 1: <query>
Claim 2: <query>
...

Claims:
{claims_text}
"""
    return call_llm(prompt, max_new_tokens=800)


# STEP 4: Retrieve Evidence from Index
def retrieve_evidence(query_text: str, index, top_k: int = 5) -> Dict[str, List[Dict]]:
    """Retrieve evidence using the vector index."""
    evidence = {}
    
    for line in query_text.split("\n"):
        if ":" in line and line.strip():
            try:
                cid, query = line.split(":", 1)
                cid = cid.strip()
                query = query.strip()
                
                if query and len(query) > 2:
                    results = search_index(index, query, k=top_k)
                    evidence[cid] = results
            except Exception as e:
                print(f"Error processing query '{line}': {e}")
                continue
    
    return evidence


# STEP 5: Temporal Constraint Tracking
def temporal_analysis(claims_text: str, evidence_dict: Dict[str, List[Dict]]) -> str:
    """Analyze claims across EARLY/MID/LATE narrative phases."""
    formatted_evidence = ""
    for cid, passages in evidence_dict.items():
        formatted_evidence += f"\n{cid} Evidence:\n"
        for p in passages[:5]:
            formatted_evidence += f"- {p['text'][:400]}...\n"

    prompt = f"""Perform temporal constraint mapping.

For each claim, analyze:
- EARLY: introduction/background
- MID: turning points/conflicts
- LATE: final outcomes

For each phase:
Status: Supports / Contradicts / Constrains
Explanation: 1 line

Output format:
Claim ID

EARLY:
Status:
Explanation:

MID:
Status:
Explanation:

LATE:
Status:
Explanation:

Claims:
{claims_text}

Evidence:
{formatted_evidence}
"""
    return call_llm(prompt, max_new_tokens=3000)


# STEP 6: Causal Evaluation with Importance Weighting
def causal_evaluation(claims_text: str, importance_text: str, temporal_text: str) -> str:
    """Evaluate consistency with importance and temporal weighting."""
    prompt = f"""Evaluate narrative consistency with importance weighting.

For each claim:
- Use IMPORTANCE (MAJOR / MINOR)
- Use EARLY/MID/LATE analysis
- Classify as:
  • FATAL CONTRADICTION (only for MAJOR claims)
  • SOFT CONTRADICTION
  • CONSISTENT

Output:
Claim ID
Importance:
Result:
Explanation: 1–2 lines

Claims:
{claims_text}

Importance:
{importance_text}

Temporal Analysis:
{temporal_text}
"""
    return call_llm(prompt, max_new_tokens=3000)


# STEP 6.5: Memory State Propagation
def memory_state_propagation(claims_text: str, temporal_text: str, importance_text: str) -> str:
    """
    Track mental state evolution across EARLY/MID/LATE phases.
    Detect impossible belief transitions vs. plausible character development.
    """
    prompt = f"""You are tracking a character's mental state evolution across a narrative.

For each MAJOR claim:
1. Identify the character's belief/trait in EARLY, MID, and LATE phases
2. Determine if state transitions are:
   - GRADUAL: Plausible evolution with triggering events
   - SUDDEN: Unexplained rapid shift
   - IMPOSSIBLE: Direct reversal without causal justification

3. For SUDDEN or IMPOSSIBLE transitions, identify:
   - What trigger event would be needed
   - Whether such an event exists in the evidence

Output format:
Claim ID
Importance: (from importance analysis)
EARLY State: <belief/trait value>
MID State: <belief/trait value>  
LATE State: <belief/trait value>
Transition Type: GRADUAL / SUDDEN / IMPOSSIBLE
Trigger Events: <list or NONE FOUND>
Analysis: 1-2 lines on causal plausibility

Claims:
{claims_text}

Temporal Analysis:
{temporal_text}

Importance:
{importance_text}
"""
    return call_llm(prompt, max_new_tokens=3000)


# STEP 6.7: Cross-Claim Dependency Validation
def cross_claim_validation(claims_text: str, causal_analysis: str, dependencies: str) -> str:
    """
    Validate if contradictions in root claims logically invalidate dependent claims.
    """
    prompt = f"""You are performing cross-claim dependency validation for narrative consistency.

Given:
1. Backstory Claims
2. Causal Consistency Analysis
3. Logical Dependencies (Map of which claims depend on which prerequisites)

Your Task:
Identify "Causally Orphaned" claims. A claim is orphaned if:
- It is a dependent claim (e.g., a goal or belief).
- Its prerequisite "Root Claim" (the event it's based on) has a FATAL CONTRADICTION.

If a root claim is false, the character's motivation for the dependent claim no longer exists in the narrative context.

Output format:
Claim ID
Status: CAUSALLY ORPHANED / VALID
Explanation: 1-2 lines on why the contradiction in the root claim invalidates this goal/belief.

Claims:
{claims_text}

Causal Analysis:
{causal_analysis}

Dependencies:
{dependencies}
"""
    return call_llm(prompt, max_new_tokens=2000)


# STEP 6.8: Counterfactual Verification
def verify_counterfactual_expectations(claims_text: str, evidence_dict: Dict[str, List[Dict]], importance_text: str) -> str:
    """
    Simulate "If-Then" expectations for MAJOR claims and verify against text.
    """
    formatted_evidence = ""
    for cid, passages in evidence_dict.items():
        formatted_evidence += f"\n{cid} Evidence:\n"
        for p in passages[:5]:
            formatted_evidence += f"- {p['text'][:400]}...\n"

    prompt = f"""Perform Counterfactual Reasoning (Simulation).

For each MAJOR claim, generate an "If-Then" expectation and verify it.

Logic:
1. Hypothesis: "If [Claim] is true..."
2. Expectation: "...then the character MUST [Reaction/Action] or MUST NOT [Reaction/Action] in relevant situations."
3. Verification: Does the evidence show the OPPOSITE behavior?

Example:
Claim: "Pacifist"
Expectation: Must not fight back in ambush.
Evidence: "He punched the guard."
Result: VIOLATED

Output format:
Claim ID
Hypothesis:
Expectation:
Verification Result: SATISFIED / VIOLATED / UNKNOWN
Explanation: 1 line

Claims:
{claims_text}

Importance:
{importance_text}

Evidence:
{formatted_evidence}
"""
    return call_llm(prompt, max_new_tokens=2500)


# STEP 7: Calculate Contradiction Score
def calculate_contradiction_score(weighted_analysis: str, memory_states: str, dependency_analysis: str, counterfactual_analysis: str) -> str:
    """
    Calculate weighted contradiction score (0.0 - 1.0) instead of binary decision.
    
    Score Formula:
    - 1.0 = Fully consistent
    - 0.7-0.9 = Minor issues
    - 0.4-0.6 = Mixed/soft contradictions
    - 0.0-0.3 = Severe contradictions
    """
    prompt = f"""You are calculating a weighted contradiction score for narrative consistency.

Scoring Rules:
- MAJOR claim with FATAL CONTRADICTION: -1.0 points
- MAJOR claim with SOFT CONTRADICTION: -0.3 points
- MINOR claim contradiction (any): -0.2 points
- IMPOSSIBLE state transition (from memory analysis): -0.5 points
- CAUSALLY ORPHANED claim (from dependency analysis): -0.4 points
- VIOLATED EXPECTATION (from counterfactuals): -0.5 points
- CONSISTENT claims: 0.0 points

Calculate:
1. Total penalty from all claims
2. Total possible penalty (# of MAJOR claims × 1.0)
3. Final Score = 1.0 - (Total Penalty / Total Possible)

Also consider:
- Memory state transitions (IMPOSSIBLE transitions increase penalty)
- Dependency violations (CAUSALLY ORPHANED claims increase penalty)
- Counterfactual violations (VIOLATED expectations increase penalty)
- Temporal consistency across EARLY/MID/LATE

Output format:
Contradiction Score: <0.0 to 1.0>
Breakdown:
- MAJOR FATAL: <count> claims (-<penalty>)
- MAJOR SOFT: <count> claims (-<penalty>)
- MINOR issues: <count> claims (-<penalty>)
- Memory violations: <count> (-<penalty>)
- Dependency violations: <count> (-<penalty>)
- Counterfactual violations: <count> (-<penalty>)
Total Penalty: <value>
Final Score: <0.0-1.0>

Weighted Analysis:
{weighted_analysis}

Memory States:
{memory_states}

Dependency Analysis:
{dependency_analysis}

Counterfactual Analysis:
{counterfactual_analysis}
"""
    return call_llm(prompt, max_new_tokens=1000)


# STEP 8: Final Decision with Scoring
def final_decision_with_score(weighted_analysis: str, memory_states: str, score_analysis: str, dependency_analysis: str, counterfactual_analysis: str) -> str:
    """
    Make final decision using contradiction score instead of binary logic.
    
    Decision Rules:
    - Score < 0.4 → Prediction: 0 (Contradict)
    - Score >= 0.4 → Prediction: 1 (Consistent)
    """
    prompt = f"""Make the final consistency judgment using the contradiction score.

Decision Rules:
- Contradiction Score < 0.4 → Output 0 (Contradict)
- Contradiction Score >= 0.4 → Output 1 (Consistent)

Rationale Requirements:
- Must cite the specific logical failure (e.g. "Root Claim X contradicted").
- MUST include a SHORT VERBATIM QUOTE from the novel that proves the contradiction.
- Keep it under 3 lines for CSV formatting.

Output format:
Prediction: <0 or 1>
Confidence Score: <extract from score analysis>
Rationale: <Specific failure> + "Direct Quote from text"

Weighted Analysis:
{weighted_analysis}

Memory States:
{memory_states}

Score Analysis:
{score_analysis}

Dependency Analysis:
{dependency_analysis}

Counterfactual Analysis:
{counterfactual_analysis}
"""
    return call_llm(prompt, max_new_tokens=600)


# LEGACY: Keep old final_decision for backward compatibility
def final_decision(weighted_analysis: str) -> str:
    """Make final binary decision with rationale."""
    prompt = f"""Make the final consistency judgment.

Rules:
- If ANY MAJOR claim has FATAL CONTRADICTION → Output 0
- If only MINOR or SOFT contradictions → Output 1

Output:
Prediction: <0 or 1>
Rationale: 1–2 lines citing strongest MAJOR evidence

Analysis:
{weighted_analysis}
"""
    return call_llm(prompt, max_new_tokens=400)
