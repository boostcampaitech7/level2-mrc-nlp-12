import os
import json
from collections import Counter, defaultdict




class HardVotingEnsemble:
    def __init__(self, predictions_dir, output_dir):
        self.predictions_dir = predictions_dir
        self.output_dir = output_dir
        self.votes = defaultdict(list)

    def load_predictions(self):
        for file in os.listdir(predictions_dir):
            if file.endswith(".json"):
                file_path = os.path.join(predictions_dir, file)
                with open(file_path, 'r') as f:
                    predictions = json.load(f)
                    for id, pred in predictions.items():
                        votes[id].append(pred)

