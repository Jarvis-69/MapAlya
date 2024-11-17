import torch
import json

class EDI_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, targets):
        self.encodings = encodings
        self.targets = targets

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.targets["input_ids"][idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

def load_data_with_errors(filename="synthetic_edifact_data_with_errors.json"):
    with open(filename, "r") as f:
        data = json.load(f)

    erroneous_segments = [d["erroneous_segment"] for d in data]
    corrected_segments = [d["original_segment"] for d in data]
    labels = [d["label"] for d in data]  # Optionnel
    return erroneous_segments, corrected_segments, labels