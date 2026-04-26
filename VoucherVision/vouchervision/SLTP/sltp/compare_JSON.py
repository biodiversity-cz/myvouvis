import json
import Levenshtein
from sklearn.metrics.pairwise import cosine_similarity
import re
import matplotlib.pyplot as plt
from collections import OrderedDict

class JSONComparator:
    def __init__(self, json_gt_path, json_llm_path, json_llm_edited_path):
        with open(json_gt_path, 'r') as f:
            self.json_gt = json.load(f, object_pairs_hook=OrderedDict)
        with open(json_llm_path, 'r') as f:
            self.json_llm = json.load(f, object_pairs_hook=OrderedDict)
        with open(json_llm_edited_path, 'r') as f:
            self.json_llm_edited = json.load(f, object_pairs_hook=OrderedDict)
        self.keys = list(self.json_gt.keys())

        self.ignore = ['catalogNumber']

    def extract_words(self, obj):
        words = set()
        for key, value in obj.items():
            if isinstance(value, str):
                words.update(re.findall(r'\w+', key.lower()))  # Extract words from keys
                words.update(re.findall(r'\w+', value.lower()))  # Extract words from values
        return words

    def exact_match_percentage(self):
        total_keys = len(self.keys)
        exact_matches = sum(1 for key in self.keys if self.ground_truth[key] == self.generated.get(key))
        return (exact_matches / total_keys) * 100 if total_keys != 0 else 0

    def levenshtein_distance(self, gt_value, gen_value):
        if len(str(gt_value)) == 0 and len(str(gen_value)) == 0:
            return 1  # Both values are empty, return 1 as the score
        elif len(str(gt_value)) == 0 and len(str(gen_value)) > 0:
            return 2  # Both values are empty, return 1 as the score
        elif gt_value == "" or gt_value is None or gen_value == "" or gen_value is None:
            return 0  # Handle when either value is empty or None
        else:
            pass

        gt_len = len(str(gt_value))
        gen_len = len(str(gen_value))

        # Adjust score based on relative length
        if gen_len > gt_len:
            levenshtein_score = 1 + (Levenshtein.distance(str(gt_value), str(gen_value)) / max(gt_len, gen_len))
            return levenshtein_score
        elif gen_len < gt_len:
            levenshtein_score = 1 - (Levenshtein.distance(str(gt_value), str(gen_value)) / max(gt_len, gen_len))
            return levenshtein_score
        else:
            levenshtein_score = 1 - (Levenshtein.distance(str(gt_value), str(gen_value)) / max(gt_len, gen_len))
            return levenshtein_score
        
    def overall_levenshtein_score(self, json_gt, json_gen):
        total_distance = 0
        total_possible_distance = 0
        for key in self.keys:
            if key not in self.ignore:
                gt_value = json_gt[key]
                gen_value = json_gen.get(key)
                if gt_value is not None and gen_value is not None:
                    gt_len = len(str(gt_value))
                    gen_len = len(str(gen_value))
                    total_distance += Levenshtein.distance(str(gt_value), str(gen_value))
                    total_possible_distance += max(gt_len, gen_len)
        return 1 - (total_distance / total_possible_distance) if total_possible_distance != 0 else 0
    
    def compare_json(self, json_gt, json_gen):
        results = OrderedDict()
        for key in self.keys:
            if key not in self.ignore:
                results[key] = {
                    'levenshtein_distance': self.levenshtein_distance(json_gt[key], json_gen.get(key)),
                }
        return results
    
    def plot_comparison(self):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        
        # Comparison between json_gt and json_llm
        results_llm = self.compare_json(self.json_gt, self.json_llm)
        overall_score_llm = self.overall_levenshtein_score(self.json_gt, self.json_llm)
        overall_score_show_llm = round(overall_score_llm, 4)
        
        ax = axes[0]
        test = 'levenshtein_distance'
        scores_llm = [metric[test] - 1 for metric in results_llm.values()]  # Subtract 1 to make scores relative to zero
        keys_llm = list(results_llm.keys())[::-1]  # Reverse the order of keys
        colors_llm = ['green' if score == 0 else (0.5, 0.5, 1) if score < 0 else (1, 0.5, 0.5) for score in scores_llm]

        ax.barh(keys_llm, scores_llm[::-1], color=colors_llm[::-1])  # Reverse the order of scores and colors
        ax.set_title(f"LLM transcription\noverall score = {overall_score_show_llm}")
        ax.set_xlabel('Score')
        ax.tick_params(axis='y', rotation=0)
        ax.set_xlim([-1, 1])  # Adjust x-axis limits
        ax.axvline(0, color='black', linewidth=0.5)  # Add a vertical line at zero
        
        # Comparison between json_gt and json_llm_edited
        results_llm_edited = self.compare_json(self.json_gt, self.json_llm_edited)
        overall_score_llm_edited = self.overall_levenshtein_score(self.json_gt, self.json_llm_edited)
        overall_score_show_llm_edited = round(overall_score_llm_edited, 4)
        
        ax = axes[1]
        test = 'levenshtein_distance'
        scores_llm_edited = [metric[test] - 1 for metric in results_llm_edited.values()]  # Subtract 1 to make scores relative to zero
        keys_llm_edited = list(results_llm_edited.keys())[::-1]  # Reverse the order of keys
        colors_llm_edited = ['green' if score == 0 else (0.5, 0.5, 1) if score < 0 else (1, 0.5, 0.5) for score in scores_llm_edited]

        ax.barh(keys_llm_edited, scores_llm_edited[::-1], color=colors_llm_edited[::-1])  # Reverse the order of scores and colors
        ax.set_title(f"Edited LLM transcription\noverall score = {overall_score_show_llm_edited}")
        ax.set_xlabel('Score')
        ax.tick_params(axis='y', rotation=0)
        ax.set_xlim([-1, 1])  # Adjust x-axis limits
        ax.axvline(0, color='black', linewidth=0.5)  # Add a vertical line at zero
        
        plt.tight_layout()
        plt.show()



    def compare_words(self):
        gt_words = self.extract_words(self.json_gt)
        llm_words = self.extract_words(self.json_llm)
        new_words_llm = llm_words - gt_words
        missing_words_llm = gt_words - llm_words

        llm_edited_words = self.extract_words(self.json_llm_edited)
        new_words_llm_edited = llm_edited_words - gt_words
        missing_words_llm_edited = gt_words - llm_edited_words
        return new_words_llm, missing_words_llm, new_words_llm_edited, missing_words_llm_edited

    # def run_tests_by_key(self):
    #     results = OrderedDict()
    #     for key in self.keys:
    #         if key not in self.ignore:
    #             # gt_values = set(self.ground_truth[key]) if isinstance(self.ground_truth[key], list) else {self.ground_truth[key]}
    #             # gen_values = set(self.generated.get(key)) if isinstance(self.generated.get(key), list) else {self.generated.get(key)}
    #             results[key] = {
    #                 # 'exact_match': self.ground_truth[key] == self.generated.get(key),
    #                 # 'jaccard_similarity': self.jaccard_similarity(gt_values, gen_values),
    #                 'levenshtein_distance': self.levenshtein_distance(self.ground_truth[key], self.generated.get(key)),
    #             }
    #     return results

    # def plot_results(self, results):
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))  # Single horizontal plot for Levenshtein distance
        # overall_score = self.overall_levenshtein_score()
        # overall_score_show = round(overall_score, 4)
        
        # test = 'levenshtein_distance'
        # scores = [metric[test] for metric in results.values()]
        # colors = []
        # for score in scores:
        #     if score == 1:
        #         colors.append('green')
        #     elif score < 1:
        #         colors.append((0.5, 0.5, 1))  # Gray gradient for scores less than 1
        #     else:
        #         colors.append((1, 0.5, 0.5))  # Red gradient for scores greater than 1

        # ax.barh(list(results.keys()), scores, color=colors)  # Use barh for horizontal bar chart
        # ax.set_title(f"{test} - overall score = {overall_score_show}")
        # ax.set_xlabel('Score')  # Change xlabel to reflect score orientation
        # ax.tick_params(axis='y', rotation=0)  # Rotate y-axis labels for better visibility
        # ax.set_xlim([0, 2.0])  # Adjust x-axis limits for better visualization
        
        # plt.tight_layout()
        # plt.show()




if __name__ == '__main__':
    # comparator = JSONComparator('./demo/1_gt.json', './demo/1_gt.json')
    # comparator = JSONComparator('./demo/1_gt.json', './demo/1_one_extra_letter.json')
    # comparator = JSONComparator('./demo/1_gt.json', './demo/1_one_extra_letter_minus_period.json')
    # comparator = JSONComparator('./demo/1_gt.json', './demo/1_one_word_different.json')
    # comparator = JSONComparator('./demo/1_gt.json', './demo/1_different_species.json')
    # comparator = JSONComparator('./demo/1_gt.json', './demo/1_different_species_and_more_words.json')
    # comparator = JSONComparator('./demo/1_gt.json', './demo/1_new_word.json')
    # comparator = JSONComparator('./demo/1_gt.json', './demo/1_new_word.json','./demo/1_one_extra_letter.json')
    comparator = JSONComparator('./demo/1_gt.json', './demo/1_new_word.json','./demo/1_with_list_notation.json')

    # results = comparator.compare_json()
    # print(results)

    # print("Exact Match Percentage:", comparator.exact_match_percentage())
    # print("Jaccard Similarity:", comparator.jaccard_similarity())
    # print("Levenshtein Distance:", comparator.levenshtein_distance())

    new_words_llm, missing_words_llm, new_words_llm_edited, missing_words_llm_edited = comparator.compare_words()
    print("New Words in Generated JSON [new_words_llm]:", new_words_llm)
    print("Missing Words in Generated JSON [missing_words_llm]:", missing_words_llm)
    print("New Words in Generated JSON [new_words_llm_edited]:", new_words_llm_edited)
    print("Missing Words in Generated JSON [missing_words_llm_edited]:", missing_words_llm_edited)

    comparator.plot_comparison()
