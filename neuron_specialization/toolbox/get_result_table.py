import os
import json
import pandas as pd
import re
import numpy as np

def extract_results(result_dir):
    """
    Extract BLEU, chrF++ and COMET scores from result files.
    
    Args:
        result_dir: Path to the directory containing language pair results
        
    Returns:
        DataFrames with results organized by direction and language group
    """
    # Define language groups
    languages = ["de", "nl", "fr", "es", "ru", "cs", "hi", "bn", "ar", "he",
                "sv", "da", "it", "pt", "pl", "bg", "kn", "mr", "ha", "mt",
                "af", "lb", "ro", "oc", "uk", "sr", "sd", "gu", "ti", "am"]
    
    high = ["de", "nl", "fr", "es", "ru", "cs", "hi", "bn", "ar", "he"]
    med = ["sv", "da", "it", "pt", "pl", "bg", "kn", "mr", "ha", "mt"]
    low = ["af", "lb", "ro", "oc", "uk", "sr", "sd", "gu", "ti", "am"]
    
    # Prepare data collection
    results = []
    
    # Process each language
    for lang in languages:
        # Process en-xx (O2M)
        o2m_dir = os.path.join(result_dir, f"en-{lang}")
        o2m_bleu, o2m_chrf, o2m_comet = extract_metrics(o2m_dir)
        
        # Process xx-en (M2O)
        m2o_dir = os.path.join(result_dir, f"{lang}-en")
        m2o_bleu, m2o_chrf, m2o_comet = extract_metrics(m2o_dir)
        
        # Determine language group
        if lang in high:
            group = "high"
        elif lang in med:
            group = "med"
        else:
            group = "low"
        
        # Add results to collection
        results.append({
            "language": lang,
            "group": group,
            "BLEU_O2M": o2m_bleu,
            "BLEU_M2O": m2o_bleu,
            "chrF++_O2M": o2m_chrf,
            "chrF++_M2O": m2o_chrf,
            "COMET_O2M": o2m_comet,
            "COMET_M2O": m2o_comet
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create summary DataFrames for each metric
    bleu_summary = create_summary_table(df, "BLEU")
    chrf_summary = create_summary_table(df, "chrF++")
    comet_summary = create_summary_table(df, "COMET")
    
    return df, bleu_summary, chrf_summary, comet_summary

def extract_metrics(dir_path):
    """Extract BLEU, chrF++ and COMET scores from files in a directory"""
    bleu_score = None
    chrf_score = None
    comet_score = None
    
    # Skip if directory doesn't exist
    if not os.path.exists(dir_path):
        return bleu_score, chrf_score, comet_score
    
    # Extract BLEU score
    bleu_file = os.path.join(dir_path, "test_detok_bleu.txt")
    if os.path.exists(bleu_file):
        with open(bleu_file, 'r') as f:
            try:
                bleu_data = json.load(f)
                bleu_score = bleu_data.get("score")
            except json.JSONDecodeError:
                pass
    
    # Extract chrF++ score
    chrf_file = os.path.join(dir_path, "test_detok_chrfpp.txt")
    if os.path.exists(chrf_file):
        with open(chrf_file, 'r') as f:
            try:
                chrf_data = json.load(f)
                chrf_score = chrf_data.get("score")
            except json.JSONDecodeError:
                pass
    
    # Extract COMET score
    comet_file = os.path.join(dir_path, "test_comet.txt")
    if os.path.exists(comet_file):
        with open(comet_file, 'r') as f:
            comet_text = f.read().strip()
            # Extract score using regex
            match = re.search(r'score: ([-+]?\d*\.\d+|\d+)', comet_text)
            if match:
                comet_score = float(match.group(1))
    
    return bleu_score, chrf_score, comet_score

def create_summary_table(df, metric):
    """Create a summary table for a specific metric"""
    # Calculate group statistics
    summary = {}
    
    # Calculate averages for each group and direction
    for group in ["high", "med", "low"]:
        group_df = df[df["group"] == group]
        o2m_avg = group_df[f"{metric}_O2M"].mean()
        m2o_avg = group_df[f"{metric}_M2O"].mean()
        avg = (o2m_avg + m2o_avg) / 2 if not (pd.isna(o2m_avg) or pd.isna(m2o_avg)) else None
        
        summary[group] = {
            "O2M": o2m_avg,
            "M2O": m2o_avg,
            "Avg": avg
        }
    
    # Calculate overall averages
    o2m_all = df[f"{metric}_O2M"].mean()
    m2o_all = df[f"{metric}_M2O"].mean()
    avg_all = (o2m_all + m2o_all) / 2 if not (pd.isna(o2m_all) or pd.isna(m2o_all)) else None
    
    summary["all"] = {
        "O2M": o2m_all,
        "M2O": m2o_all,
        "Avg": avg_all
    }
    
    return summary

def print_table(metric_name, summary):
    """Print a table for a specific metric"""
    # Round values based on metric type
    if metric_name == "COMET":
        digits = 3
    else:
        digits = 1
    
    print(f"{metric_name} Scores:")
    
    # Header
    print(f"{'High (5M)':<20} {'Med (1M)':<20} {'Low (100K)':<20} | {'All (61M)'}")
    print(f"{'O2M':^6} {'M2O':^6} {'Avg':^6} {'O2M':^6} {'M2O':^6} {'Avg':^6} {'O2M':^6} {'M2O':^6} {'Avg':^6} | {'O2M':^6} {'M2O':^6} {'Avg':^6}")
    print("-" * 75)
    
    # Get all direction values in single row
    o2m_row = []
    m2o_row = []
    avg_row = []
    
    for group in ["high", "med", "low", "all"]:
        group_data = summary[group]
        # Format O2M
        val = group_data["O2M"]
        o2m_row.append(f"{val:.{digits}f}" if not pd.isna(val) else "N/A")
        
        # Format M2O
        val = group_data["M2O"]
        m2o_row.append(f"{val:.{digits}f}" if not pd.isna(val) else "N/A")
        
        # Format Avg
        val = group_data["Avg"]
        avg_row.append(f"{val:.{digits}f}" if not pd.isna(val) else "N/A")
    
    # Print rows with correct alignment
    o2m_str = f"{o2m_row[0]:^6} {m2o_row[0]:^6} {avg_row[0]:^6} {o2m_row[1]:^6} {m2o_row[1]:^6} {avg_row[1]:^6} {o2m_row[2]:^6} {m2o_row[2]:^6} {avg_row[2]:^6} | {o2m_row[3]:^6} {m2o_row[3]:^6} {avg_row[3]:^6}"
    print(o2m_str)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract MT evaluation results")
    parser.add_argument("--result_dir", type=str, required=True, help="Path to the result directory")
    #parser.add_argument("--output_dir", type=str, required=True, help="Output CSV file path")
    args = parser.parse_args()
    
    # Extract results
    results_df, bleu_summary, chrf_summary, comet_summary = extract_results(args.result_dir)
    
    # Print tables for each metric
    print_table("BLEU", bleu_summary)
    print()
    print_table("chrF++", chrf_summary)
    print()
    print_table("COMET", comet_summary)
    
    # save it if you want
    # Save summary in tabular format
    #with open(f"{args.output_dir}/summary_tables.txt", "w") as f:
    #    # Redirect print output to file
    #    import sys
    #    original_stdout = sys.stdout
    #    sys.stdout = f
    #    
    #    print_table("BLEU", bleu_summary)
    #    print()
    #    print_table("chrF++", chrf_summary)
    #    print()
    #    print_table("COMET", comet_summary)
    #    
    #    sys.stdout = original_stdout
    #
    #print(f"\nSummary tables saved to {args.output_dir}/summary_tables.txt")

if __name__ == "__main__":
    main()