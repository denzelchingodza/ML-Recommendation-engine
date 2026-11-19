"""Which models the gate knows, and which of them can explain their picks exactly (rule 3)."""

from recommender.evaluation.seasons import MODEL_NAMES

KNOWN_MODELS = frozenset(MODEL_NAMES)

# Rule 3: a champion must come with an exact explainer that passes the Phase 6 checks.
# Add a model here only together with its explainer and its explanation tests.
EXPLAINABLE = frozenset({"label_similarity", "label_similarity_full"})
