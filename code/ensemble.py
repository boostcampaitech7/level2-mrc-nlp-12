import argparse
import json
import os
from collections import Counter, defaultdict


class VotingEnsemble:
    def __init__(self, predictions_dir, output_dir, method="hard"):
        self.predictions_dir = predictions_dir
        self.output_dir = output_dir
        self.method = method
        self.prediction_files = []
        self.votes = defaultdict(list)

    def load_predictions(self):
        # predictions_dir에서 모든 json 파일을 불러옴
        self.prediction_files = [
            f for f in os.listdir(self.predictions_dir) if f.endswith(".json")
        ]

        for file in self.prediction_files:
            file_path = os.path.join(self.predictions_dir, file)
            with open(file_path, "r") as f:
                predictions = json.load(f)
                for id, pred in predictions.items():
                    self.votes[id].append(pred)

    def vote(self):
        if self.method == "hard":
            return self.__hard_vote()
        elif self.method == "soft":
            return self.__soft_vote()

    def __hard_vote(self):
        final_predictions = {}
        for id, preds in self.votes.items():
            final_predictions[id] = max(set(preds), key=preds.count)
        return final_predictions

    def __soft_vote(self):
        final_predictions = {}
        for id, preds in self.votes.items():
            prob_dict = defaultdict(float)
            for pred in preds:
                for choice in pred:
                    prob_dict[choice["text"]] += choice["probability"]
            final_predictions[id] = max(prob_dict, key=prob_dict.get)
        return final_predictions

    def save_results(self, final_predictions):
        os.makedirs(self.output_dir, exist_ok=True)

        ensemble_number = (
            sum(1 for entry in os.scandir(self.output_dir) if entry.is_dir()) + 1
        )
        ensemble_dir = os.path.join(self.output_dir, f"ensemble_{ensemble_number}")
        os.makedirs(ensemble_dir)

        predictions_path = os.path.join(ensemble_dir, "predictions.json")
        with open(predictions_path, "w", encoding="utf-8") as f:
            json.dump(final_predictions, f, indent=4, ensure_ascii=False)

        used_predictions_path = os.path.join(ensemble_dir, "used_predictions.txt")
        with open(used_predictions_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.prediction_files))

        print(f"Results saved to {ensemble_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voting Ensemble for predictions.")
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default="predictions_for_ensemble",
        help="Directory containing prediction JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ensemble_outputs",
        help="Directory to save ensemble results.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["hard", "soft"],
        default="hard",
        help="Voting method: hard or soft.",
    )

    args = parser.parse_args()

    print(f"Using predictions from: {args.predictions_dir}")
    print(f"Saving results to: {args.output_dir}")
    print(f"Using voting method: {args.method}")

    ensemble = VotingEnsemble(args.predictions_dir, args.output_dir, args.method)

    # load json files
    ensemble.load_predictions()

    # get final predictions
    final_predictions = ensemble.vote()

    # save results
    ensemble.save_results(final_predictions)
