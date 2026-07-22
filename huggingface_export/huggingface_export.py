"""
Uploads models, model cards, preprocess.py, and modeling.py to all 4 published
chimp-vocalization Hugging Face repos.

Run from the folder containing: preprocess.py, modeling.py, and the 4
README_*.md files (e.g. huggingface_export/).
"""
from bioacoustics.classifier.model.cnn10_torch import CNN10Model
from bioacoustics.classifier.model.cnn12_torch import CNN12Model
from huggingface_export.modeling import CNN10Hub, CNN12Hub



from huggingface_hub import HfApi
api = HfApi()
info = api.whoami()
for org in info["orgs"]:
    print(org["name"], "-> role:", org.get("roleInOrg"))

# each entry: (training class, hub wrapper class, model_dir, hub repo name)
exports = [
    (CNN10Model, CNN10Hub, "output/results/predictions/cnn10/all", "utrechtuniversity/chimp-vocalization-cnn10-synthetic"),
    (CNN10Model, CNN10Hub, "output/results/predictions/cnn10/mefou",  "utrechtuniversity/chimp-vocalization-cnn10-sanctuary"),
    (CNN12Model, CNN12Hub, "output/results/predictions/cnn12/all", "utrechtuniversity/chimp-vocalization-cnn12-synthetic"),
    (CNN12Model, CNN12Hub, "output/results/predictions/cnn12/mefou",  "utrechtuniversity/chimp-vocalization-cnn12-sanctuary"),
]

for model_cls, hub_cls, model_dir, repo_id in exports:
    print(f"Exporting {repo_id} from {model_dir} ...")

    trained = model_cls(num_channels=3, dropout_rate=0.5, model_dir=model_dir)
    trained._load_model()

    hub_model = hub_cls(num_channels=3, num_labels=2, dropout_rate=0.5)
    hub_model.acoustic_model.load_state_dict(trained.acoustic_model.state_dict())
    hub_model.eval()

    hub_model.push_to_hub(repo_id)
    print(f"Pushed {repo_id}")


repos = [
    ("model_cards/README_cnn10_synthetic.md", "utrechtuniversity/chimp-vocalization-cnn10-synthetic"),
    ("model_cards/README_cnn10_sanctuary.md", "utrechtuniversity/chimp-vocalization-cnn10-sanctuary"),
    ("model_cards/README_cnn12_synthetic.md", "utrechtuniversity/chimp-vocalization-cnn12-synthetic"),
    ("model_cards/README_cnn12_sanctuary.md", "utrechtuniversity/chimp-vocalization-cnn12-sanctuary"),
]

for readme_file, repo_id in repos:
    print(f"Uploading to {repo_id} ...")

    api.upload_file(
        path_or_fileobj=readme_file,
        path_in_repo="README.md",
        repo_id=repo_id,
    )
    api.upload_file(
        path_or_fileobj="preprocess.py",
        path_in_repo="preprocess.py",
        repo_id=repo_id,
    )
    api.upload_file(
        path_or_fileobj="modeling.py",
        path_in_repo="modeling.py",
        repo_id=repo_id,
    )

    print(f"Done: {repo_id}")

print("All repos updated.")
