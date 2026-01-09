"""
Main Pipeline Orchestration.
Implements the complete 7-step analysis with open-source components.
"""

from retrieval import build_vector_index
from reasoning import (
    extract_claims, classify_importance, generate_queries,
    retrieve_evidence, temporal_analysis, causal_evaluation,
    memory_state_propagation, calculate_contradiction_score,
    final_decision_with_score, map_claim_dependencies,
    cross_claim_validation, verify_counterfactual_expectations
)
from csv_writer import write_results
import argparse

DATA_DIR = "data"


def load_text(path: str) -> str:
    """Load text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_decision(decision_output: str) -> tuple:
    """
    Parse LLM decision output with confidence score support.
    
    Returns:
        (prediction: int, confidence_score: float, rationale: str)
    """
    prediction = 1  # Default
    if "Prediction: 0" in decision_output or "Prediction:0" in decision_output:
        prediction = 0
    
    # Extract confidence score
    confidence_score = None
    # import re already at top
    score_match = re.search(r'Confidence Score:\s*(0?\.\d+|1\.0)', decision_output)
    if score_match:
        try:
            confidence_score = float(score_match.group(1))
        except:
            pass
    
    # If no confidence score found, use default based on prediction
    if confidence_score is None:
        confidence_score = 0.2 if prediction == 0 else 0.8
    
    rationale = "No rationale provided"
    if "Rationale:" in decision_output:
        rationale = decision_output.split("Rationale:")[-1].strip()
        rationale = rationale.split("\n")[0].strip()
        rationale = rationale.strip('"').strip("'")
    
    return prediction, confidence_score, rationale


def run_pipeline(submission_mode: bool = False):
    """Run the complete narrative consistency pipeline."""
    results = []

    # Find all story files
    story_files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("story_")])
    
    if not story_files:
        print(f"⚠️  No story files found in {DATA_DIR}/")
        return
    
    print(f"📚 Found {len(story_files)} stories to process")
    if submission_mode:
        print("🚀 RUNNING IN SUBMISSION MODE (Strict CSV format)")
    else:
        print("🛠️  RUNNING IN DEBUG MODE (Includes confidence scores)")
    print("\n")

    for fname in story_files:
        story_id = fname.replace("story_", "").replace(".txt", "")
        novel_path = os.path.join(DATA_DIR, f"story_{story_id}.txt")
        backstory_path = os.path.join(DATA_DIR, f"backstory_{story_id}.txt")

        if not os.path.exists(backstory_path):
            print(f"⚠️  Skipping story {story_id}: No matching backstory")
            continue

        try:
            print(f"\n{'='*60}")
            print(f"📖 Processing Story {story_id}")
            print(f"{'='*60}")
            
            novel_text = load_text(novel_path)
            backstory_text = load_text(backstory_path)

            # Build vector index with open-source embeddings
            print("🔍 Building vector index...")
            index = build_vector_index(novel_text)

            # STEP 1: Extract claims
            print("📋 Step 1: Extracting claims...")
            claims = extract_claims(backstory_text)
            print("✅ Claims extracted")

            # STEP 2: Importance classification
            print("⭐ Step 2: Classifying importance...")
            importance = classify_importance(claims)
            print("✅ Importance classified")

            # STEP 2.5: Map claim dependencies
            print("🔗 Step 2.5: Mapping claim dependencies (Reasoning Graph)...")
            dependencies = map_claim_dependencies(claims)
            print("✅ Dependencies mapped")

            # STEP 3: Generate queries
            print("🔎 Step 3: Generating queries...")
            queries = generate_queries(claims)
            print("✅ Queries generated")

            # STEP 4: Retrieve evidence
            print("📚 Step 4: Retrieving evidence...")
            evidence = retrieve_evidence(queries, index, top_k=5)
            print(f"✅ Evidence retrieved for {len(evidence)} claims")

            # STEP 5: Temporal analysis
            print("⏱️  Step 5: Temporal analysis (EARLY/MID/LATE)...")
            temporal = temporal_analysis(claims, evidence)
            print("✅ Temporal analysis complete")

            # STEP 6: Causal evaluation
            print("🧠 Step 6: Causal evaluation with weighting...")
            weighted = causal_evaluation(claims, importance, temporal)
            print("✅ Causal evaluation complete")

            # STEP 6.5: Memory state propagation
            print("🔄 Step 6.5: Memory state propagation...")
            memory_states = memory_state_propagation(claims, temporal, importance)
            print("✅ Memory states tracked")

            # STEP 6.7: Cross-Claim Dependency Validation
            print("🖇️  Step 6.7: Cross-Claim Dependency Validation...")
            dependency_analysis = cross_claim_validation(claims, weighted, dependencies)
            print("✅ Dependency validation complete")

            # STEP 6.8: Counterfactual Verification
            print("🎭 Step 6.8: Counterfactual Verification (Simulation)...")
            counterfactuals = verify_counterfactual_expectations(claims, evidence, importance)
            print("✅ Counterfactuals verified")

            # STEP 7: Calculate contradiction score
            print("📊 Step 7: Calculating contradiction score...")
            score_analysis = calculate_contradiction_score(weighted, memory_states, dependency_analysis, counterfactuals)
            print("✅ Contradiction score calculated")

            # STEP 8: Final decision with scoring
            print("⚖️  Step 8: Final decision with confidence...")
            decision = final_decision_with_score(weighted, memory_states, score_analysis, dependency_analysis, counterfactuals)
            print("✅ Final decision made")

            # Parse result
            prediction, confidence_score, rationale = parse_decision(decision)
            
            print(f"\n{'='*60}")
            print(f"📊 RESULT for Story {story_id}")
            print(f"{'='*60}")
            print(f"Prediction: {prediction}")
            print(f"Confidence Score: {confidence_score}")
            print(f"Rationale: {rationale}")
            print(f"{'='*60}\n")

            if submission_mode:
                # Strictly: story_id, prediction, rationale
                results.append([story_id, prediction, rationale])
            else:
                # Include confidence for debugging
                results.append([story_id, prediction, confidence_score, rationale])

        except Exception as e:
            print(f"❌ Error processing story {story_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Write CSV
    if results:
        output_file = "results.csv"
        # csv_writer detects column count automatically
        write_results(results, output_path=output_file, include_confidence=not submission_mode)
        print(f"\n🎉 Pipeline complete! Results saved to {output_file}")
    else:
        print("\n⚠️  No results to write.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Narrative Consistency Pipeline")
    parser.add_argument(
        "--submission", 
        action="store_true", 
        help="Run in submission mode (strips confidence scores from CSV)"
    )
    args = parser.parse_args()

    print("🚀 Starting Narrative Consistency Pipeline")
    print("🔧 Using: Pathway + SentenceTransformers + Hugging Face LLM")
    print("="*60)
    
    run_pipeline(submission_mode=args.submission)
