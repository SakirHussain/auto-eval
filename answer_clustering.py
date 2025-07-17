import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer


def debug(msg: str):
    """Simple debug logger."""
    print(f"[DEBUG] {msg}")


class StudentAnswerClustering:
    def __init__(
        self,
        ideal_answers_path: str,
        student_answers_path: str,
        threshold: float = 0.80,
    ):
        debug("Initializing StudentAnswerClustering")
        debug(f"Ideal path={ideal_answers_path}, Student path={student_answers_path}, threshold={threshold}")
        self.ideal_answers_path = ideal_answers_path
        self.student_answers_path = student_answers_path
        self.threshold = threshold
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Raw entries
        self.ideal_entries = []   # list of dicts from qnia.json
        self.ideal_keys = []      # corresponding keys (e.g., 'answer_1')
        self.ideal_splits = []    # list of split_by_rubric lists

        self.student_entries = [] # list of dicts from student_answers.json

        # Simplified lists for embedding
        self.ideal_texts = []     # full_text strings
        self.student_texts = []   # full_answer strings

        # Clustering outputs
        self.student_to_ideal_map = {}
        self.unclustered = []
        self.clusters = {}

    def load_data(self):
        # Load ideal answers
        debug(f"Loading ideal answers from '{self.ideal_answers_path}'")
        with open(self.ideal_answers_path, 'r') as f:
            data = json.load(f)

        if 'ideal_answers' in data and isinstance(data['ideal_answers'], dict):
            entries = data['ideal_answers']
            self.ideal_keys = list(entries.keys())
            self.ideal_entries = [entries[k] for k in self.ideal_keys]
            debug(f"Loaded {len(self.ideal_entries)} ideal answer entries: {self.ideal_keys}")
        else:
            raise KeyError("Could not find 'ideal_answers' in ideal answers JSON.")

        # Extract text & rubric splits for each ideal answer
        for entry in self.ideal_entries:
            txt = entry.get('full_text') or entry.get('answer', '')
            self.ideal_texts.append(txt)
            # Pull out the split_by_rubric list (or default to empty list)
            self.ideal_splits.append(entry.get('split_by_rubric', []))
        debug(f"Extracted {len(self.ideal_texts)} ideal texts and splits")

        # Load student answers
        debug(f"Loading student answers from '{self.student_answers_path}'")
        with open(self.student_answers_path, 'r') as f:
            sdata = json.load(f)

        students = sdata.get('student_answers')
        if isinstance(students, list):
            self.student_entries = students
            debug(f"Loaded {len(self.student_entries)} student answer entries")
        else:
            raise KeyError("Could not find 'student_answers' list in student answers JSON.")

        # Extract text for embedding
        for se in self.student_entries:
            txt = se.get('full_answer') or se.get('answer', '')
            self.student_texts.append(txt)
        debug(f"Extracted {len(self.student_texts)} student texts for embedding")

    def cluster_answers(self):
        # Compute embeddings
        debug("Encoding ideal answer embeddings...")
        ideal_emb = self.model.encode(self.ideal_texts)
        debug("Encoding student answer embeddings...")
        student_emb = self.model.encode(self.student_texts)

        n_ideal = len(self.ideal_texts)
        debug(f"Preparing clusters for {n_ideal} ideals")
        self.clusters = {key: [] for key in self.ideal_keys}
        self.student_to_ideal_map.clear()
        self.unclustered.clear()

        # Assign each student answer
        for idx, emb in enumerate(student_emb):
            sims = cosine_similarity([emb], ideal_emb)[0]
            max_i = int(np.argmax(sims))
            max_val = float(sims[max_i])
            ideal_key = self.ideal_keys[max_i]
            debug(f"Student #{idx}: top match '{ideal_key}' with cosine={max_val:.4f}")

            if max_val >= self.threshold:
                self.clusters[ideal_key].append(idx)
                self.student_to_ideal_map[idx] = {
                    'ideal_key': ideal_key,
                    'similarity': max_val,
                    'student_text': self.student_texts[idx],
                    'student_split_by_rubric': self.student_entries[idx].get('split_by_rubric', []),
                    # Now pulling from our pre-extracted list:
                    'ideal_split_by_rubric': self.ideal_splits[max_i],
                    'ideal_full_text': self.ideal_texts[max_i]
                }
                debug(f"--> Assigned to cluster '{ideal_key}' with ideal splits: {self.ideal_splits[max_i]}")
            else:
                self.unclustered.append(idx)
                debug(f"--> Unclustered (cosine {max_val:.4f} < threshold {self.threshold})")

    def get_clustered_ideal(self, student_id: int):
        debug(f"Retrieving mapping for student_id={student_id}")
        mapping = self.student_to_ideal_map.get(student_id)
        if mapping:
            debug(f"--> Found: ideal={mapping['ideal_key']}, sim={mapping['similarity']:.4f}")
            return mapping
        else:
            debug("--> No mapping found; unclustered")
            return {'message': 'Unclustered: needs manual grading'}

    def print_unclustered(self):
        debug("Listing unclustered student answers:")
        for idx in self.unclustered:
            print(f"\n[UNCLUSTERED] Student #{idx}:")
            print(self.student_texts[idx])

    def visualize_clusters(self):
        # PCA 2D for visualization only
        debug("Starting 2D PCA visualization")
        ideal_emb = self.model.encode(self.ideal_texts)
        student_emb = self.model.encode(self.student_texts)
        all_emb = np.vstack([ideal_emb, student_emb])
        debug(f"Reducing {all_emb.shape[0]} embeddings to 2D")
        reduced = PCA(n_components=2).fit_transform(all_emb)
        ideal_2d = reduced[:len(self.ideal_texts)]
        student_2d = reduced[len(self.ideal_texts):]

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = ['red','blue','green','purple','orange','cyan']

        # Plot ideal points
        for i, (x, y) in enumerate(ideal_2d):
            ax.scatter(x, y,
                       color=colors[i % len(colors)],
                       marker='X', s=200,
                       label=f"Ideal {self.ideal_keys[i]}")

        # Plot student points
        for i, (x, y) in enumerate(student_2d):
            if i in self.student_to_ideal_map:
                key = self.student_to_ideal_map[i]['ideal_key']
                col = colors[self.ideal_keys.index(key) % len(colors)]
            else:
                col = 'gray'
            ax.scatter(x, y, color=col, alpha=0.6)

        ax.set_title("Student Answer Clusters (2D PCA)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    clustering = StudentAnswerClustering(
        ideal_answers_path='qnia.json',
        student_answers_path='student_answers.json',
        threshold=0.80
    )
    clustering.load_data()
    clustering.cluster_answers()

    # Example retrieval of broken-down structures
    print("\n=== Example mapping for student #0 ===")
    print(clustering.get_clustered_ideal(0))

    # List any that need manual review
    clustering.print_unclustered()

    # Show clusters in 2D
    clustering.visualize_clusters()
