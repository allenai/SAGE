import json
import sys
from typing import List, Dict, Any
from sage.src.api.gpt import GPT
from tqdm import tqdm
from collections import Counter
import statistics
import os
from sage.src.functions.utils.temporal import get_video_duration


def count_tool_calls(entry: Dict) -> tuple[int, Counter]:
    """
    Count the number of tool calls and their types in an entry.
    Args:
        entry: Dictionary containing `complete_results` (dict or list[dict])
        
    Returns:
        Tuple of (total_tool_calls, tool_type_counter)
    """
    tool_call_count = 0
    tool_type_counter = Counter()

    cr = entry.get('complete_results', None)

    # Normalize to a list of complete_result dicts
    if isinstance(cr, dict):
        cr_list = [cr]
    else:
        cr_list = []

    # Check iterative_reasoner for tool calls across all attempts
    for cr_item in cr_list:
        iterative_reasoner = cr_item.get('iterative_reasoner')

        if iterative_reasoner and len(iterative_reasoner) > 0:
            for iterative_reasoner_result in iterative_reasoner:
                if 'tool_calls' in iterative_reasoner_result:
                    for tool_call in iterative_reasoner_result['tool_calls']:
                        tool_call_count += 1
                        # Extract tool name and track type
                        if isinstance(tool_call, dict):
                            tool_name = tool_call.get('name', 'unknown')
                        else:
                            tool_name = str(tool_call)

                        if tool_name:
                            # Extract base tool name (remove _#<call_num> suffix)
                            base_tool_name = tool_name.split('_#')[0] if '_#' in tool_name else tool_name
                            tool_type_counter[base_tool_name] += 1
    return tool_call_count, tool_type_counter


def has_tool_calls(entry: Dict) -> bool:
    """Check if the entry involved any tool calls."""
    tool_call_count, _ = count_tool_calls(entry)
    return tool_call_count > 0


def evaluate_answers(
    jsonl_path: str,
) -> tuple[float, int, int, float, float, Dict, list]:
    """
    Evaluate answers from a JSONL file using GPT-4o to check if they match the correct answers.

    Args:
        jsonl_path: Path to JSONL file containing answers and correct answers

    Returns:
        Tuple containing:
        - Overall accuracy score between 0 and 1
        - Number of direct answers
        - Number of iterative reasoner occasions
        - Accuracy for direct answers
        - Accuracy for iterative reasoner occasions
        - Dictionary with comprehensive statistics
        - List of per-sample evaluation results
    """
    
    gpt = GPT()

    # Load and parse JSONL file
    answers = []
    direct_answers = 0
    iterative_reasoner_occasions = 0
    direct_answers_list = []
    iterative_reasoner_list = []
    direct_answer_indices = []  # Store indices of direct answer entries
    iterative_reasoner_indices = []     # Store indices of iterative reasoner entries
    
    # Statistics tracking
    total_tool_calls = 0
    tool_type_counter = Counter()
    none_predictions = 0
    could_not_produce_count = 0
    list_answers_count = 0
    total_entries = 0

    # For per-type accuracy tracking
    ques_type_stats = {}  # Track accuracy per question type
    difficulty_stats = {}  # Track accuracy per difficulty level
    modality_stats = {}    # Track accuracy per modality

    # For per-sample stats
    per_sample_stats = []

    with open(jsonl_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            total_entries += 1
            
            # Track entry types for accurate result extraction
            entry_index = len(answers)
            
            answers.append(entry)
            
            # Track None predictions for statistics
            if entry.get('answer') is None:
                none_predictions += 1
            
            # Track "could not produce an answer" responses
            answer_text = entry.get('answer', '')
            if isinstance(answer_text, str) and "could not produce an answer" in answer_text.lower():
                could_not_produce_count += 1
            
            # Track list answers
            if isinstance(answer_text, list):
                list_answers_count += 1
            
            # Track type information for per-type accuracy computation
            ques_type = entry.get('ques_type')
            difficulty = entry.get('difficulty')
            modality = entry.get('modality')
            
            # Initialize type tracking if not already present
            if ques_type and ques_type not in ques_type_stats:
                ques_type_stats[ques_type] = {'correct': 0, 'total': 0}
            if difficulty and difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = {'correct': 0, 'total': 0}
            if modality and modality not in modality_stats:
                modality_stats[modality] = {'correct': 0, 'total': 0}
            
            # Count tool calls
            tool_call_count, entry_tool_types = count_tool_calls(entry)
            total_tool_calls += tool_call_count
            tool_type_counter.update(entry_tool_types)
            
            # Count iterative reasoner occasions
            has_iterative_reasoner = False
            cr = entry.get("complete_results", None)

            if isinstance(cr, list):
                for cr_item in cr:
                    if isinstance(cr_item, dict) and cr_item.get("iterative_reasoner"):
                        if len(cr_item["iterative_reasoner"]) > 0:
                            has_iterative_reasoner = True
                            break
            elif isinstance(cr, dict):
                if cr.get("iterative_reasoner") and len(cr["iterative_reasoner"]) > 0:
                    has_iterative_reasoner = True

            if has_iterative_reasoner:
                iterative_reasoner_occasions += 1
                iterative_reasoner_list.append(entry)
                iterative_reasoner_indices.append(entry_index)
            else:
                # Count direct answers (when no iterative_reasoner was used)
                direct_answers += 1
                direct_answers_list.append(entry)
                direct_answer_indices.append(entry_index)

    total = len(answers)
    
    # Initialize results
    accuracy = 0.0
    direct_accuracy = 0.0
    iterative_reasoner_accuracy = 0.0

    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Per-sample correctness placeholder
    per_sample_correct = [None] * total
    if total > 0:
        # Initialize thread pool
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def evaluate_single_answer(entry):
            # Handle None predictions
            if entry.get('answer') is None:
                # For None predictions, we need to check if the correct answer is also None/empty
                # or if the model should have provided an answer
                correct_answer = entry.get('correct_answer', '')
                full_answer = entry.get('full_answer', '')
                
                # If correct answer is also None/empty, then None prediction is correct
                if not correct_answer and not full_answer:
                    return True
                else:
                    # If there should be an answer but model predicted None, it's wrong
                    return False
            
            correct_answer = str(entry.get('correct_answer', ''))
            gt_answer = entry.get('full_answer', '')
            if correct_answer is not None and len(correct_answer) > 0:
                gt_answer = f"option: {correct_answer}; answer_text: {gt_answer}"
            
            # Handle list answers by converting to string
            model_answer = entry.get('answer', '')
            if isinstance(model_answer, list):
                model_answer = str(model_answer)
            
            prompt = f"""Compare the model prediction and the ground truth and determine if they convey the same meaning for the question:

Question: {entry['question']}

Model Prediction: {model_answer}
Ground Truth: {gt_answer}

You MUST respond with the verdict as 'True' if they match semantically or 'False' if they don't match.
Answer in the following format:
Reasoning: <Reasoning for the verdict>
Verdict: <True/False>
"""

            response = gpt.get_response(prompt)[0]
            if response is None:
                return (False, None)
            
            # Extract reasoning and verdict
            reasoning = None
            verdict = None
            if response:
                if "Reasoning:" in response and "Verdict:" in response:
                    reasoning_part = response.split("Reasoning: ")[1].split("Verdict: ")[0].strip()
                    verdict_part = response.split("Verdict: ")[1].split("\n")[0].strip()
                    reasoning = reasoning_part
                    verdict = verdict_part
                else:
                    # Fallback: try to extract just the verdict
                    if "Verdict:" in response:
                        verdict = response.split("Verdict: ")[1].split("\n")[0].strip()
            
            correct = verdict is not None and "true" in verdict.strip().lower()
            return (correct, reasoning)

        # Process all answers in parallel
        results = [None] * total
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = {executor.submit(evaluate_single_answer, entry): i for i, entry in enumerate(answers)}
            for future in tqdm(as_completed(futures), total=len(answers), desc="Evaluating answers"):
                idx = futures[future]
                results[idx] = future.result()

        # Convert results to boolean values for overall accuracy calculation
        bool_results = []
        evaluation_reasonings = []  # Store reasoning for each evaluation
        
        for i, result in enumerate(results):
            if isinstance(result, tuple) and len(result) == 2:
                # Result format: (is_correct, reasoning)
                bool_results.append(result[0])
                evaluation_reasonings.append(result[1])  # reasoning
            else:
                # Regular answer result: boolean
                bool_results.append(result)
                evaluation_reasonings.append(None)

        # Calculate overall accuracy
        correct = sum(bool_results)
        accuracy = correct / total

        # Calculate accuracy for direct answers
        direct_results = [bool_results[i] for i in direct_answer_indices]
        direct_correct = sum(direct_results)
        direct_accuracy = direct_correct / direct_answers if direct_answers > 0 else 0.0

        # Calculate accuracy for iterative reasoner occasions
        iterative_reasoner_results = [bool_results[i] for i in iterative_reasoner_indices]
        iterative_reasoner_correct = sum(iterative_reasoner_results)
        iterative_reasoner_accuracy = iterative_reasoner_correct / iterative_reasoner_occasions if iterative_reasoner_occasions > 0 else 0.0

        # Calculate per-type accuracies
        for i, entry in enumerate(answers):
            is_correct = bool_results[i]
            
            # Update ques_type statistics
            ques_type = entry.get('ques_type')
            if ques_type:
                ques_type_stats[ques_type]['total'] += 1
                if is_correct:
                    ques_type_stats[ques_type]['correct'] += 1
            
            # Update difficulty statistics
            difficulty = entry.get('difficulty')
            if difficulty:
                difficulty_stats[difficulty]['total'] += 1
                if is_correct:
                    difficulty_stats[difficulty]['correct'] += 1
            
            # Update modality statistics
            modality = entry.get('modality')
            if modality:
                modality_stats[modality]['total'] += 1
                if is_correct:
                    modality_stats[modality]['correct'] += 1


        # Fill per-sample stats
        for i, entry in enumerate(answers):
            # Extract duration information
            duration = None
            video_path = entry.get('path') or entry.get('video_path')
            
            # Try to get duration from entry first, then from video file
            if entry.get('duration') is not None:
                duration = entry.get('duration')
            elif video_path and os.path.exists(video_path):
                try:
                    duration = get_video_duration(video_path)
                except Exception as e:
                    # Silently fail and continue without duration
                    pass
            
            sample_stat = {
                'index': i,
                'question': entry.get('question'),
                'answer': entry.get('answer'),
                'correct_answer': entry.get('correct_answer'),
                'full_answer': entry.get('full_answer'),
                'is_correct': bool_results[i],
                'evaluation_reasoning': evaluation_reasonings[i] if i < len(evaluation_reasonings) else None,
                'none_prediction': entry.get('answer') is None,
                'could_not_produce': isinstance(entry.get('answer'), str) and "could not produce an answer" in entry.get('answer', '').lower(),
                'is_list_answer': isinstance(entry.get('answer'), list),
                'used_tool_calls': has_tool_calls(entry),
                'tool_call_types': dict(count_tool_calls(entry)[1]),
                'ques_type': entry.get('ques_type'),
                'difficulty': entry.get('difficulty'),
                'modality': entry.get('modality'),
                'duration': duration,
                'video_path': video_path,
                'num_attempts': entry.get('num_attempts'),
            }
            
            per_sample_stats.append(sample_stat)
    else:
        # If not evaluating, just fill with None for is_correct
        for i, entry in enumerate(answers):
            # Extract duration information
            duration = None
            video_path = entry.get('path') or entry.get('video_path')
            
            # Try to get duration from entry first, then from video file
            if entry.get('duration') is not None:
                duration = entry.get('duration')
            elif video_path and os.path.exists(video_path):
                try:
                    duration = get_video_duration(video_path)
                except Exception as e:
                    # Silently fail and continue without duration
                    pass
            
            sample_stat = {
                'index': i,
                'question': entry.get('question'),
                'answer': entry.get('answer'),
                'correct_answer': entry.get('correct_answer'),
                'full_answer': entry.get('full_answer'),
                'is_correct': None,
                'evaluation_reasoning': None,
                'none_prediction': entry.get('answer') is None,
                'could_not_produce': isinstance(entry.get('answer'), str) and "could not produce an answer" in entry.get('answer', '').lower(),
                'is_list_answer': isinstance(entry.get('answer'), list),
                'used_tool_calls': has_tool_calls(entry),
                'tool_call_types': dict(count_tool_calls(entry)[1]),
                'ques_type': entry.get('ques_type'),
                'difficulty': entry.get('difficulty'),
                'modality': entry.get('modality'),
                'duration': duration,
                'video_path': video_path,
                'num_attempts': entry.get('num_attempts'),
            }
            
            per_sample_stats.append(sample_stat)

    num_attempts_values = [entry.get('num_attempts') for entry in answers if entry.get('num_attempts') is not None and isinstance(entry.get('num_attempts'), int)]
    avg_num_attempts = statistics.mean(num_attempts_values) if num_attempts_values else None
    num_attempts_count = len(num_attempts_values)
    
    # Compile comprehensive statistics
    stats = {
        'total_entries': total_entries,
        'valid_entries': total,
        'none_predictions': none_predictions,
        'could_not_produce_count': could_not_produce_count,
        'could_not_produce_percentage': (could_not_produce_count / total_entries * 100) if total_entries > 0 else 0,
        'list_answers_count': list_answers_count,
        'list_answers_percentage': (list_answers_count / total_entries * 100) if total_entries > 0 else 0,
        'num_attempts_statistics': {
            'average_num_attempts': avg_num_attempts,
            'entries_with_num_attempts': num_attempts_count,
            'num_attempts_percentage': (num_attempts_count / total_entries * 100) if total_entries > 0 else 0,
        },
        'tool_call_statistics': {
            'total_tool_calls': total_tool_calls,
            'entries_with_tool_calls': sum(1 for entry in answers if has_tool_calls(entry)),
            'tool_call_percentage': (sum(1 for entry in answers if has_tool_calls(entry)) / total * 100) if total > 0 else 0,
            'average_tool_calls_per_entry': total_tool_calls / total if total > 0 else 0,
            'tool_type_distribution': dict(tool_type_counter),
            'most_common_tools': tool_type_counter.most_common(10)
        },
        'response_statistics': {
            'direct_answers': direct_answers,
            'iterative_reasoner_occasions': iterative_reasoner_occasions,
        },
        'per_type_statistics': {
            'ques_type_accuracies': {k: {'accuracy': v['correct'] / v['total'], 'correct': v['correct'], 'total': v['total']} 
                                    for k, v in ques_type_stats.items() if v['total'] > 0},
            'difficulty_accuracies': {k: {'accuracy': v['correct'] / v['total'], 'correct': v['correct'], 'total': v['total']} 
                                     for k, v in difficulty_stats.items() if v['total'] > 0},
            'modality_accuracies': {k: {'accuracy': v['correct'] / v['total'], 'correct': v['correct'], 'total': v['total']} 
                                   for k, v in modality_stats.items() if v['total'] > 0}
        },
    }

    return (
        accuracy,
        direct_answers,
        iterative_reasoner_occasions,
        direct_accuracy,
        iterative_reasoner_accuracy,
        stats,
        per_sample_stats,
    )

def calculate_duration_wise_accuracy(per_sample_stats: List[Dict]) -> Dict[str, Dict]:
    """
    Calculate accuracy statistics grouped by duration buckets.
    
    Args:
        per_sample_stats: List of per-sample statistics containing duration and is_correct
        
    Returns:
        Dictionary with duration bucket statistics
    """
    # Filter samples with valid duration and is_correct values
    valid_samples = [
        s for s in per_sample_stats 
        if s.get('duration') is not None 
        and s.get('is_correct') is not None
        and isinstance(s.get('duration'), (int, float))
        and s.get('duration') > 0
    ]
    
    if not valid_samples:
        return {}
    
    # Define specific duration buckets
    duration_buckets = [
        (0, 60, "0-60s"),
        (60, 180, "60-180s"),
        (180, 300, "180-300s"),
        (300, 600, "300-600s"),
        (600, 1200, "600-1200s"),
        (1200, 2400, "1200-2400s"),
        (2400, float('inf'), "2400+s")
    ]
    
    # Create duration buckets
    bucket_data = {}
    for bucket_start, bucket_end, bucket_name in duration_buckets:
        bucket_data[bucket_name] = {
            'total': 0,
            'correct': 0,
            'incorrect': 0,
            'durations': [],
            'samples': []
        }
    
    for sample in valid_samples:
        duration = sample['duration']
        
        # Find the appropriate bucket
        bucket_name = None
        for bucket_start, bucket_end, name in duration_buckets:
            if bucket_start <= duration < bucket_end:
                bucket_name = name
                break
        
        if bucket_name is None:
            continue  # Skip if duration doesn't fit any bucket
        
        bucket_data[bucket_name]['total'] += 1
        bucket_data[bucket_name]['durations'].append(duration)
        bucket_data[bucket_name]['samples'].append(sample)
        
        if sample['is_correct']:
            bucket_data[bucket_name]['correct'] += 1
        else:
            bucket_data[bucket_name]['incorrect'] += 1
    
    # Calculate statistics for each bucket
    stats = {}
    for bucket_name, bucket_info in bucket_data.items():
        if bucket_info['total'] > 0:  # Only include buckets with samples
            accuracy = (bucket_info['correct'] / bucket_info['total']) * 100
            avg_duration = statistics.mean(bucket_info['durations']) if bucket_info['durations'] else 0
            
            # Calculate average tool calls per bucket and track tool type distribution
            total_tool_calls = 0
            bucket_tool_type_counter = Counter()
            for sample in bucket_info['samples']:
                tool_call_types = sample.get('tool_call_types', {})
                # Sum up all tool call counts for this sample
                sample_tool_calls = sum(tool_call_types.values()) if isinstance(tool_call_types, dict) else 0
                total_tool_calls += sample_tool_calls
                # Track tool type distribution
                if isinstance(tool_call_types, dict):
                    bucket_tool_type_counter.update(tool_call_types)
            
            avg_tool_calls = total_tool_calls / bucket_info['total'] if bucket_info['total'] > 0 else 0
            
            num_attempts_values = [
                sample.get('num_attempts') for sample in bucket_info['samples']
                if sample.get('num_attempts') is not None and isinstance(sample.get('num_attempts'), int)
            ]
            avg_num_attempts = statistics.mean(num_attempts_values) if num_attempts_values else None
            num_attempts_count = len(num_attempts_values)
            
            stats[bucket_name] = {
                'total_samples': bucket_info['total'],
                'correct': bucket_info['correct'],
                'incorrect': bucket_info['incorrect'],
                'accuracy': accuracy,
                'avg_duration': avg_duration,
                'min_duration': min(bucket_info['durations']) if bucket_info['durations'] else 0,
                'max_duration': max(bucket_info['durations']) if bucket_info['durations'] else 0,
                'avg_tool_calls': avg_tool_calls,
                'total_tool_calls': total_tool_calls,
                'tool_type_distribution': dict(bucket_tool_type_counter),
                'avg_num_attempts': avg_num_attempts,
                'entries_with_num_attempts': num_attempts_count
            }
    
    return stats


def check_args_equal(args1: Dict[str, Any], args2: Dict[str, Any]) -> bool:
    """Check if two arguments are equal."""
    # check the keys in args1 are present in args2 and vice versa
    if set(args1.keys()) != set(args2.keys()):
        return False
    for key in args1.keys():
        if args1[key] != args2.get(key, None):
            return False
    print("FILTERED DUE TO SAME ARGS")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_responses.py <jsonl_path>")
        sys.exit(1)
    
    jsonl_path = sys.argv[1]

    (
        accuracy,
        direct_answers,
        iterative_reasoner_occasions,
        direct_accuracy,
        iterative_reasoner_accuracy,
        stats,
        per_sample_stats,
    ) = evaluate_answers(jsonl_path)
    
    # Print basic results
    print(f"\n=== Evaluation Results ===")
    print(f"Total entries: {stats['total_entries']}")
    print(f"Could not produce an answer: {stats['could_not_produce_count']} ({stats['could_not_produce_percentage']:.1f}%)")
    
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Direct Answers: {direct_answers} (Accuracy: {direct_accuracy:.2%})")
    if iterative_reasoner_occasions > 0:
        print(f"Iterative Reasoner Occasions: {iterative_reasoner_occasions} (Accuracy: {iterative_reasoner_accuracy:.2%})")
    else:
        print("No iterative reasoner occasions found")
    
    # Print per-type accuracy statistics
    print(f"\n=== Per-Type Accuracy Statistics ===")
    
    # Question type accuracies
    if stats['per_type_statistics']['ques_type_accuracies']:
        print(f"\nQuestion Type Accuracies:")
        for ques_type, type_stats in stats['per_type_statistics']['ques_type_accuracies'].items():
            print(f"  {ques_type}: {type_stats['accuracy']:.2%} ({type_stats['correct']}/{type_stats['total']})")
    
    # Difficulty accuracies
    if stats['per_type_statistics']['difficulty_accuracies']:
        print(f"\nDifficulty Level Accuracies:")
        for difficulty, type_stats in stats['per_type_statistics']['difficulty_accuracies'].items():
            print(f"  {difficulty}: {type_stats['accuracy']:.2%} ({type_stats['correct']}/{type_stats['total']})")
    
    # Modality accuracies
    if stats['per_type_statistics']['modality_accuracies']:
        print(f"\nModality Accuracies:")
        for modality, type_stats in stats['per_type_statistics']['modality_accuracies'].items():
            print(f"  {modality}: {type_stats['accuracy']:.2%} ({type_stats['correct']}/{type_stats['total']})")
    
    # Print tool call statistics
    print(f"\n=== Tool Call Statistics ===")
    tool_stats = stats['tool_call_statistics']
    print(f"Total tool calls: {tool_stats['total_tool_calls']}")
    print(f"Entries with tool calls: {tool_stats['entries_with_tool_calls']} ({tool_stats['tool_call_percentage']:.1f}%)")
    print(f"Average tool calls per entry: {tool_stats['average_tool_calls_per_entry']:.2f}")
    
    if tool_stats['tool_type_distribution']:
        print(f"\nTool type distribution:")
        for tool_type, count in tool_stats['most_common_tools']:
            print(f"  {tool_type}: {count}")
    
    # Print num_attempts statistics
    print(f"\n=== Num Attempts Statistics ===")
    num_attempts_stats = stats['num_attempts_statistics']
    if num_attempts_stats['average_num_attempts'] is not None:
        print(f"Average num attempts: {num_attempts_stats['average_num_attempts']:.2f}")
        print(f"Entries with num attempts: {num_attempts_stats['entries_with_num_attempts']} ({num_attempts_stats['num_attempts_percentage']:.1f}%)")
    else:
        print("No num_attempts data available")

    # Save per-sample stats to a JSON file with the same name as the input but _gpt.json extension
    import os
    base, _ = os.path.splitext(jsonl_path)
    per_sample_json_path = base + ".json"
    with open(per_sample_json_path, 'w') as f:
        json.dump(per_sample_stats, f, indent=2)
    print(f"\nPer-sample evaluation results saved to {per_sample_json_path}")
    
    # Print a summary of reasoning storage
    reasoning_count = sum(1 for sample in per_sample_stats if sample.get('evaluation_reasoning') is not None)
    print(f"Evaluation reasoning stored for {reasoning_count}/{len(per_sample_stats)} samples")