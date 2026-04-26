import os
import argparse
from embeddings_db import VoucherVisionEmbeddingTest

def test_same(path_to_files): 
    version = "Individual"
    ground_truth_dir = os.path.join(path_to_files, "Individual")
    llm_output_dir = os.path.join(path_to_files, version)
    mean_similarity, max_diff, similarities, mean_key_similarity, max_diff_key, key_similarities = evaluate_LLM(ground_truth_dir, llm_output_dir, "hkunlp/instructor-xl")
    report_results(version, mean_similarity, max_diff, similarities, mean_key_similarity, max_diff_key, key_similarities)


def test_one_text_error(path_to_files):
    version = "Individual_Smudge_OneTextError"
    ground_truth_dir = os.path.join(path_to_files, "Individual")
    llm_output_dir = os.path.join(path_to_files, version)
    mean_similarity, max_diff, similarities, mean_key_similarity, max_diff_key, key_similarities = evaluate_LLM(ground_truth_dir, llm_output_dir, "hkunlp/instructor-xl")
    report_results(version, mean_similarity, max_diff, similarities, mean_key_similarity, max_diff_key, key_similarities)


def test_one_key_error(path_to_files):
    version = "Individual_Smudge_OneKeyError"
    ground_truth_dir = os.path.join(path_to_files, "Individual")
    llm_output_dir = os.path.join(path_to_files, version)
    mean_similarity, max_diff, similarities, mean_key_similarity, max_diff_key, key_similarities = evaluate_LLM(ground_truth_dir, llm_output_dir, "hkunlp/instructor-xl")
    report_results(version, mean_similarity, max_diff, similarities, mean_key_similarity, max_diff_key, key_similarities)

    
def test_text(path_to_files):
    version = "Individual_Smudge_Text"
    ground_truth_dir = os.path.join(path_to_files, "Individual")
    llm_output_dir = os.path.join(path_to_files, version)
    mean_similarity, max_diff, similarities, mean_key_similarity, max_diff_key, key_similarities = evaluate_LLM(ground_truth_dir, llm_output_dir, "hkunlp/instructor-xl")
    report_results(version, mean_similarity, max_diff, similarities, mean_key_similarity, max_diff_key, key_similarities)


def test_keys(path_to_files):
    version = "Individual_Smudge_Keys"
    ground_truth_dir = os.path.join(path_to_files, "Individual")
    llm_output_dir = os.path.join(path_to_files, version)
    mean_similarity, max_diff, similarities, mean_key_similarity, max_diff_key, key_similarities = evaluate_LLM(ground_truth_dir, llm_output_dir, "hkunlp/instructor-xl")
    report_results(version, mean_similarity, max_diff, similarities, mean_key_similarity, max_diff_key, key_similarities)


def report_results(version, mean_similarity, max_diff, similarities, mean_key_similarity, max_diff_key, key_similarities):
    print(f"\n\nTesting --- {version}")
    print(f'    Mean Total Similarity: {mean_similarity}')
    print(f'    Max Total Disimilarity: {max_diff}\n')
    print(f'    Total Similarities: {similarities}\n')
    print(f'    Mean Key Similarity: {mean_key_similarity}')
    print(f'    Max Key Similarity: {max_diff_key}\n')
    print(f'    Key Similarities: {key_similarities}\n\n')


def evaluate_LLM(ground_truth_dir, llm_output_dir, model_name):
    # Create an instance of VoucherVisionEmbeddingTest
    tester = VoucherVisionEmbeddingTest(ground_truth_dir, llm_output_dir, model_name)
    # Evaluate the model
    mean_similarity, max_diff, similarities, mean_key_similarity, max_diff_key, key_similarities = tester.evaluate()

    return mean_similarity, max_diff, similarities, mean_key_similarity, max_diff_key, key_similarities


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='Evaluate LLM')
        parser.add_argument('--ground_truth_dir', required=True, help='Directory containing ground truth files')
        parser.add_argument('--llm_output_dir', required=True, help='Directory containing LLM output files')
        parser.add_argument('--model_name', default="hkunlp/instructor-xl", help='Name of the model to use for embeddings')

        args = parser.parse_args()

        mean_similarity, max_diff, similarities, mean_key_similarity, max_diff_key, key_similarities = evaluate_LLM(args.ground_truth_dir, args.llm_output_dir, args.model_name)
    except:
        path_to_files = 'D:/Dropbox/VoucherVision/HLT_Demo'
        test_same(path_to_files)
        test_one_text_error(path_to_files)
        test_one_key_error(path_to_files)
        test_text(path_to_files)
        test_keys(path_to_files)
# python evaluate_LLM_predictions.py --ground_truth_dir path/to/ground_truth --llm_output_dir path/to/llm_output
