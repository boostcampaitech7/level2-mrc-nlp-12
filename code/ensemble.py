import os
import json
from collections import Counter, defaultdict




class HardVotingEnsemble:
    def __init__(self, predictions_dir, output_dir):
        self.predictions_dir = predictions_dir
        self.output_dir = output_dir
        self.prediction_files = []
        self.votes = defaultdict(list)

    def load_predictions(self):
        #predictions_dir에서 모든 json 파일을 불러옴
        self.prediction_files = [f for f in os.listdir(self.predictions_dir) if f.endswith(".json")]

        for file in self.prediction_files:
            file_path = os.path.join(predictions_dir, file)
            with open(file_path, 'r') as f:
                predictions = json.load(f)
                for id, pred in predictions.items():
                    self.votes[id].append(pred)

    def hard_vote(self):
        final_predictions = {}
        for id, preds in self.votes.items():
            final_predictions[id] = max(set(preds), key=preds.count)
        return final_predictions

    def save_results(self):
        os.makedirs(self.output_dir, exist_ok=True)
        
        prediction_file = os.path.join(self.output_dir, "ensemble")
        ensemble_number = sum(1 for entry in os.scandir(self.output_dir) if entry.is_dir()) + 1
        ensemble_dir = os.path.join(self.output_dir, f"ensemble_{ensemble_number}")
        os.makedirs(ensemble_dir)

        final_predictions = self.hard_vote()
        predictions_path = os.path.join(ensemble_dir, "predictions.json")
        with open(predictions_path, 'w', encoding="utf-8") as f:
            json.dump(final_predictions, f, indent=4, ensure_ascii=False)

        used_predictions_path = os.path.join(ensemble_dir, "used_predictions.txt")
        with open(used_predictions_path, 'w', encoding="utf-8") as f:
            f.write("\n".join(self.prediction_files))

        print(f"Results saved to {ensemble_dir}")
        

if __name__ == "__main__":
    predictions_dir = "predictions_for_ensemble"
    output_dir = "ensemble_outputs"

    ensemble = HardVotingEnsemble(predictions_dir, output_dir)
    ensemble.load_predictions()
    ensemble.save_results()