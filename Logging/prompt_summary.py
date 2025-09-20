import json
import re
from collections import Counter
from pathlib import Path

def summarize_logs(jsonl_path=None, log_path=None, output_path="summary.txt"):
    """
    Summarizes successful parses and failures from Llama classification .jsonl and .log files.

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
    accuracies = []

    # === Process detailed .jsonl file ===
    if jsonl_path:
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
                    if isinstance(message, dict) and message.get("event") == "prompt":
                        prompt_count += 1
                    if isinstance(message, dict) and message.get("event") == "response":
                        response_count += 1
                except Exception:
                    continue  # Skip broken lines

    # === Process general .log file ===
    if log_path:
        with open(log_path, "r") as f:
            for line in f:
                # Count successful parses via accuracy lines
                if "Accuracy:" in line:
                    match = re.search(r"Accuracy:\s([0-9.]+)", line)
                    if match:
                        accuracies.append(float(match.group(1)))

                # Count parsing failures
                if "Max retries reached" in line or "parsing failure" in line.lower():
                    parsing_failures += 1

    # === Compute Statistics ===
    total_predictions = correct_predictions + incorrect_predictions + parsing_failures
    failure_rate = (parsing_failures / total_predictions * 100) if total_predictions > 0 else 0
    success_rate = (len(accuracies) / (len(accuracies) + parsing_failures) * 100) if (len(accuracies) + parsing_failures) > 0 else 0

    # === Summary Text ===
    summary_lines = [
        "========== PARSE & PREDICTION SUMMARY ==========",
        f"Detailed log file (.jsonl) : {jsonl_path if jsonl_path else 'Not provided'}",
        f"General log file  (.log)   : {log_path if log_path else 'Not provided'}",
        "",
    ]

    if jsonl_path:
        summary_lines.extend([
            "---- From Detailed JSONL ----",
            f"Total Prompts Issued     : {prompt_count}",
            f"Total Responses Received : {response_count}",
            f"Total Predictions Made   : {total_predictions}",
            f"  ├── Correct Predictions : {correct_predictions}",
            f"  ├── Incorrect Predictions: {incorrect_predictions}",
            f"  └── Parsing Failures    : {parsing_failures}",
            f"Parsing Failure Rate     : {failure_rate:.2f}%",
            f"Other Events Logged      : {dict(events_counter)}",
            "",
        ])

    if log_path:
        summary_lines.extend([
            "---- From General LOG ----",
            f"Successful Parses (Accuracy reported): {len(accuracies)}",
            f"Parsing Failures (logged)            : {parsing_failures}",
            f"Success Rate                         : {success_rate:.2f}%",
            f"Accuracies Captured                  : {accuracies}",
            f"Mean Accuracy                        : {sum(accuracies)/len(accuracies):.4f}" if accuracies else "No accuracy entries found.",
            "",
        ])

    summary_lines.append("===============================================")

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(summary_lines))

    # Print summary
    print("\n".join(summary_lines))
    print(f"\n✅ Summary saved to → {output_path}\n")


if __name__ == "__main__":
    # === CHANGE THESE ===
    dataset = "student-portuguese_hotencoded_50PassFail"
    detailed_jsonl = f"Log_Testing_withMax500_and_10fold_{dataset}.jsonl"
    general_log = f"Log_Testing_withMax500_and_10fold_{dataset}.log"
    output_file = f"parse_summary_{dataset}.txt"

    summarize_logs(detailed_jsonl, general_log, output_file)
