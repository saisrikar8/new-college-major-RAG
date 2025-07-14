import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

class RAGEvaluator:
    def __init__(self, data_path="student_data_extended - Training.csv"):
        self.df = pd.read_csv(data_path).dropna(subset=['gpa all']) # Ensure GPA exists
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.db_metadata = []

    def _create_student_profile(self, student_series):
        """Creates a text chunk from a student's data row."""
        profile_lines = []
        # Create profile from all columns except uid, gpa, and major for the query
        query_cols = [col for col in student_series.index if col not in ['uid', 'gpa all', 'major']]
        for col in query_cols:
            if pd.notna(student_series[col]):
                profile_lines.append(f"- {col.replace('_', ' ').title()}: {student_series[col]}")
        return "\n".join(profile_lines)

    def setup_evaluation(self, test_size=0.2):
        """Splits data and builds the vector DB from the training set."""
        train_df, test_df = train_test_split(self.df, test_size=test_size, random_state=42)
        
        # Build the DB using the training data
        print(f"Building vector DB with {len(train_df)} students...")
        profile_texts = []
        for _, row in train_df.iterrows():
            # The profile stored in the DB can contain everything
            full_profile = self._create_student_profile(row) + f"\n- GPA: {row['gpa all']}"
            profile_texts.append(full_profile)
            self.db_metadata.append({"uid": row['uid'], "gpa": float(row['gpa all'])})

        embeddings = self.embedder.encode(profile_texts, convert_to_numpy=True, show_progress_bar=True)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        return test_df

    def evaluate(self, validation_df, k=5):
        """Runs the evaluation and returns the GPA Mean Absolute Error."""
        actual_gpas = []
        predicted_gpas = []

        print(f"\nRunning evaluation on {len(validation_df)} validation students...")
        for _, student in validation_df.iterrows():
            # The query profile should not contain the answer (GPA)
            query_profile = self._create_student_profile(student)
            
            query_vec = self.embedder.encode([query_profile]).astype("float32")
            _, indices = self.index.search(query_vec, k)
            
            # Get GPAs of retrieved students
            retrieved_gpas = [self.db_metadata[i]["gpa"] for i in indices[0]]
            
            if retrieved_gpas:
                avg_retrieved_gpa = np.mean(retrieved_gpas)
                predicted_gpas.append(avg_retrieved_gpa)
                actual_gpas.append(student['gpa all'])

        # Calculate the validation metric
        mae = mean_absolute_error(actual_gpas, predicted_gpas)
        return mae

# main exec
if __name__ == "__main__":
    evaluator = RAGEvaluator()
    
    # 1. Split data and build the DB with the training portion
    validation_set = evaluator.setup_evaluation(test_size=0.2)
    
    # 2. Run evaluation
    gpa_mae = evaluator.evaluate(validation_set, k=5)
    
    print("\n--- RAG RETRIEVAL EVALUATION RESULTS ---")
    print(f"Validation Metric: Mean Absolute Error (GPA)")
    print(f" MAE: {gpa_mae:.4f}")
    print("\nInterpretation: This value represents the average absolute difference between a student's actual GPA and the average GPA of the profiles the RAG model retrieved. A lower number indicates the model is better at finding academically similar peers.")