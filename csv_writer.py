"""
CSV Writer for official submission format.
"""

import pandas as pd
import os


def write_results(rows, output_path="results.csv", include_confidence=True):
    """
    Write results to CSV with optional confidence scores.
    
    Args:
        rows: List of [story_id, prediction, rationale] or 
              [story_id, prediction, confidence_score, rationale]
        output_path: Output file path
        include_confidence: Whether rows include confidence scores
    """
    if include_confidence and rows and len(rows[0]) == 4:
        columns = ["story_id", "prediction", "confidence_score", "rationale"]
    else:
        columns = ["story_id", "prediction", "rationale"]
    
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(rows)} results to {output_path}")


def append_result(story_id: str, prediction: int, rationale: str, output_path: str = "results.csv"):
    """
    Append a single result to the CSV file.
    
    Args:
        story_id: Story identifier
        prediction: Binary prediction (0 or 1)
        rationale: Explanation for the decision
        output_path: Output CSV file path
    """
    file_exists = os.path.isfile(output_path)
    
    df = pd.DataFrame([[story_id, prediction, rationale]], 
                      columns=["story_id", "prediction", "rationale"])
    
    if file_exists:
        df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        df.to_csv(output_path, index=False)
    
    print(f"✅ Appended result for story {story_id} to {output_path}")
