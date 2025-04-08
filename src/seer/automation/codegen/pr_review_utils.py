import logging


from seer.automation.agent.embeddings import GoogleProviderEmbeddings
from seer.db import DbReviewCommentEmbedding, Session

SIMILARITY_THRESHOLD = 0.7
COMMENT_COMPARISON_LIMIT = 5

logger = logging.getLogger(__name__)


class PrReviewUtils:

    @staticmethod
    def is_positive_comment(comment_body: str, owner: str) -> bool:
        """
        Determines if a comment is positive by comparing it with stored review comment patterns.
        Returns True if the comment matches more positive patterns than negative ones.

        Args:
            comment_body: The body text of the comment to evaluate
            owner: The repository owner to filter relevant comment patterns

        Returns:
            bool: True if there are more similar positive patterns than negative ones
                 or if insufficient patterns are found for comparison
        """

        try:
            # Generate embedding using same model as in PrClosedStep
            model = GoogleProviderEmbeddings.model(
                "text-embedding-005", task_type="SEMANTIC_SIMILARITY"
            )
            # encode() returns a 2D array when given a list, so we need to extract the first embedding
            comment_embedding = model.encode([comment_body])[0]
            with Session() as session:
                # Find 5 most similar comments using cosine similarity
                similar_comments = (
                    session.query(DbReviewCommentEmbedding)
                    .where(
                        DbReviewCommentEmbedding.embedding.cosine_distance(comment_embedding)
                        <= (1 - SIMILARITY_THRESHOLD),
                        DbReviewCommentEmbedding.owner == owner,
                    )
                    .order_by(DbReviewCommentEmbedding.embedding.cosine_distance(comment_embedding))
                    .limit(COMMENT_COMPARISON_LIMIT)
                ).all()
                if not similar_comments or len(similar_comments) < COMMENT_COMPARISON_LIMIT:
                    return True  # Default to True if not enough similar comments found

                positive_patterns = sum(
                    1 for comment in similar_comments if comment.is_good_pattern
                )

                # Return True if majority of similar patterns are positive
                return positive_patterns > len(similar_comments) / 2

        except Exception as e:
            logger.warning(f"Error checking comment positivity: {e}")
            return True  # Default to True on error to be conservative
