import torch
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
import pickle

class GroupingLookup:
    """
    A class for grouping similar stack traces together.

    Attributes:
        model_path (str): The path to the pre-trained sentence transformer model.
        chunk_size (int): Size of each text chunk for processing.
        step_size (int): Step size for chunking text.
        batch_size (int): Batch size for processing embeddings.
        device (str): The device (CPU or CUDA) to use for computations.
        model (SentenceTransformer): The loaded sentence transformer model.
        embeddings_matrix (Tensor): The pre-loaded embeddings matrix.
        index_mapping (list): The pre-loaded index mapping.
        data (DataFrame): The pre-loaded preprocessed data.
    """

    def __init__(
        self,
        model_path,
        embeddings_path,
        index_map_path,
        data_path,
        chunk_size=2000,
        step_size=500,
        batch_size=1,
    ):
        """
        Initializes the GroupingLookup object with the specified parameters.

        Parameters:
            model_path (str): The path to the sentence transformer model to use.
            embeddings_path (str): The path to the pre-loaded embeddings matrix.
            index_map_path (str): The path to the pre-loaded index mapping.
            data_path (str): The path to the pre-loaded preprocessed data.
            chunk_size (int): Size of each text chunk for processing.
            step_size (int): Step size for chunking text.
            batch_size (int): Batch size for processing embeddings.
        """
        self.chunk_size = chunk_size
        self.step_size = step_size
        self.batch_size = batch_size
        self.model = SentenceTransformer(
            model_path,
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        )
        self.embeddings_matrix = torch.load(embeddings_path)
        with open(index_map_path, "rb") as f:
            self.index_mapping = pickle.load(f)
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)


    def chunk_text(self, text):
        """
        Chunks the given text into smaller parts based on the predefined chunk and step sizes.

        Parameters:
            text (str): The text to be chunked.

        Returns:
            list: A list of text chunks.
        """
        if len(text) <= self.chunk_size or self.chunk_size == 0:
            return [text]
        return [text[i:i + self.chunk_size] for i in range(0, len(text) - self.chunk_size + 1, self.step_size)]

    def encode_texts(self, texts):
        """
        Encodes a list of texts into embeddings using the pre-loaded model.

        Parameters:
            texts (list of str): The texts to be encoded.

        Returns:
            Tensor: A tensor of embeddings.
        """
        if not texts:
            print("No texts to encode.")
            return torch.Tensor() 
        embeddings = []
        for start_index in range(0, len(texts), self.batch_size):
            end_index = min(start_index + self.batch_size, len(texts))
            batch_texts = texts[start_index:end_index]
            with torch.no_grad():
                batch_embeddings = self.model.encode(batch_texts, convert_to_tensor=True).to(self.device)
                embeddings.append(batch_embeddings)
        return torch.cat(embeddings) if embeddings else torch.Tensor()

    def find_top_candidates(self, embedding, top_k=5):
        """
        Finds top candidate group_ids based on similarity to first embedding chunk.

        Parameters:
            embedding (Tensor): The embedding of the text to compare.
            top_k (int): Number of top candidates to return.

        Returns:
            list: List of top candidate group IDs.
        """
        similarities = cosine_similarity(embedding, self.embeddings_matrix, dim=1)
        top_indices = torch.argsort(similarities, descending=True)[:top_k]

        candidate_group_ids = {self.data.iloc[self.index_mapping[idx.item()]]['group_id'] for idx in top_indices}
        return list(candidate_group_ids)
    
    def weighted_cosine_similarity(self, embeddings1, embeddings2):
        """
        Calculates the weighted cosine similarity between two sets of embeddings. This is done
        by iterating through the embedding chunks and calculating the cosine similarity between
        each chunk. The similarity score is weighted by a decay factor to give more weight to
        the first chunk.

        Parameters:
            embeddings1 (Tensor): First set of embeddings.
            embeddings2 (Tensor): Second set of embeddings.

        Returns:
            float: The weighted cosine similarity score.
        """
        if embeddings1.nelement() == 0 or embeddings2.nelement() == 0:
            return 0

        total_similarity, total_weight = 0.0, 0.0
        decay_factor = 0.9
        num_comparisons = min(len(embeddings1), len(embeddings2))

        for i in range(num_comparisons):
            weight = decay_factor ** i
            similarity = cosine_similarity(embeddings1[i].unsqueeze(0), embeddings2[i].unsqueeze(0)).item()
            total_similarity += weight * similarity
            total_weight += weight

        return total_similarity / total_weight if total_weight else 0

    def find_nearest_stacktrace(self, new_stacktrace):
        """
        Finds the nearest stack trace based on the cosine similarity of embeddings.
        Steps:
        1. The first chunk of the new stacktrace is compared to the embeddings matrix to find the top candidates.
        2. Weighted cosine similarity is used to compare the new stacktrace to the top candidates.

        Parameters:
            new_stacktrace (str): The new stacktrace to compare.

        Returns:
            dict: Information about the most similar stacktrace.
        """
        first_chunk_embedding = self.encode_texts([self.chunk_text(new_stacktrace)[0]])
        if first_chunk_embedding.nelement() == 0:
            print("No valid embedding generated for the first chunk of the input stacktrace.")
            return None

        candidates = self.find_top_candidates(first_chunk_embedding, top_k=5)

        best_similarity = -1
        best_group_id = None
        best_stacktrace = None
        new_embeddings = self.encode_texts(self.chunk_text(new_stacktrace))

        for group_id in candidates:
            group_indices = [i for i, x in enumerate(self.index_mapping) if self.data.iloc[x]['group_id'] == group_id]
            candidate_embeddings = self.embeddings_matrix[group_indices]
            similarity = self.weighted_cosine_similarity(new_embeddings, candidate_embeddings)

            if similarity > best_similarity:
                best_similarity = similarity
                best_group_id = group_id
                best_stacktrace = self.data[self.data['group_id'] == group_id]['stacktrace'].iloc[0]

        return {
            "group_id": best_group_id,
            "stacktrace": best_stacktrace,
            "similarity": best_similarity
        }
