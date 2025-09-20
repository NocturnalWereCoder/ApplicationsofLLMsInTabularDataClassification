import json
from collections import Counter
from pathlib import Path

def summarize_log(jsonl_path, log_path, output_path):
    """
    Summarizes the .jsonl and .log files from Llama classification run and saves to a file.
    
    Args:
        jsonl_path (str or Path): Path to the detailed .jsonl log file.
        log_path (str or Path): Path to the general .log file.
        output_path (str or Path): Path to save the summary text file.
    """
    # Counters
    events_counter = Counter()
    correct_predictions = 0
    incorrect_predictions = 0
    parsing_failures = 0
    prompt_count = 0
    response_count = 0

    # === Process detailed .jsonl file ===
    with open(jsonl_path, "r") as f:
        for line in f:
            try:
                log_entry = json.loads(line)
                message = log_entry.get("message", {})
                event = message.get("event")

                if event:
                    events_counter[event] += 1
                    if event == "prediction_accuracy":
                        correct = message.get("correct")
                        if correct is True:
                            correct_predictions += 1
                        elif correct is False:
                            incorrect_predictions += 1
                else:
                    if isinstance(message, dict) and message.get("event") == "prompt":
                        prompt_count += 1
                    if isinstance(message, dict) and message.get("event") == "response":
                        response_count += 1
            except Exception:
                continue  # Ignore broken lines

    # === Process general .log file for parsing failures ===
    with open(log_path, "r") as f:
        for line in f:
            if "Max retries reached" in line:
                parsing_failures += 1

    total_predictions = correct_predictions + incorrect_predictions + parsing_failures
    failure_rate = (parsing_failures / total_predictions * 100) if total_predictions > 0 else 0

    # === Summary Text ===
    summary_lines = [
        "========== PROMPT SUMMARY ==========",
        f"Detailed log file : {jsonl_path}",
        f"General log file  : {log_path}",
        "",
        f"Total Prompts Issued     : {prompt_count}",
        f"Total Responses Received : {response_count}",
        f"Total Predictions Made   : {total_predictions}",
        f"  ├── Correct Predictions : {correct_predictions}",
        f"  ├── Incorrect Predictions: {incorrect_predictions}",
        f"  └── Parsing Failures    : {parsing_failures}",
        f"Parsing Failure Rate    : {failure_rate:.2f}%",
        "",
        f"Other Events Logged     : {dict(events_counter)}",
        "====================================="
    ]

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(summary_lines))

    # Also print
    print("\n".join(summary_lines))
    print(f"\nSummary saved to → {output_path}\n")

if __name__ == "__main__":
    # === CHANGE THESE ===
    dataset = "wine"
    detailed_jsonl = f"Log_Testing_withMax500_and_10fold_wine.jsonl"
    general_log = f"Log_Testing_withMax500_and_10fold_wine.log"
    output_file = "prompt_summary.txt"


    summarize_log(detailed_jsonl, general_log, output_file)
