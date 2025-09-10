# # # # # # # # # # # # # # # # # # # # # # # #
# Alpaca instruction dataset feature store    #
# showcasing Ray offline store and compute     #
# engine capabilities for NLP features        #
# # # # # # # # # # # # # # # # # # # # # # # #

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureService, FeatureView, Field, ValueType
from feast.infra.offline_stores.file_source import FileSource
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64, String

# Constants related to the generated data sets
CURRENT_DIR = Path(__file__).parent

# Entity definitions for alpaca dataset
instruction = Entity(
    name="instruction",
    description="instruction sample id",
    value_type=ValueType.INT64,
    join_keys=["instruction_id"],
)

# Data sources - Ray offline store works with FileSource
# These will be processed by Ray for efficient distributed data access
alpaca_instructions_source = FileSource(
    name="alpaca_instructions",
    path=f"{CURRENT_DIR}/data/alpaca_data.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

# Feature Views - These leverage Ray compute engine for distributed processing
alpaca_instructions_view = FeatureView(
    name="alpaca_instructions",
    entities=[instruction],
    ttl=timedelta(days=30),
    schema=[
        Field(name="instruction", dtype=String),
        Field(name="input", dtype=String),
        Field(name="output", dtype=String),
        Field(name="text", dtype=String),
    ],
    online=True,
    source=alpaca_instructions_source,
    tags={"team": "nlp", "processing": "ray", "dataset": "alpaca"},
)


# On-demand feature view showcasing Ray compute engine capabilities for NLP
# This demonstrates real-time text feature transformations using Ray
@on_demand_feature_view(
    sources=[alpaca_instructions_view],
    schema=[
        Field(name="instruction_length", dtype=Int64),
        Field(name="output_length", dtype=Int64),
        Field(name="input_length", dtype=Int64),
        Field(name="text_complexity_score", dtype=Float64),
        Field(name="has_input", dtype=Int64),
    ],
)
def text_analytics_features(inputs: dict):
    """
    On-demand NLP feature transformations processed by Ray compute engine.
    These calculations happen in real-time and can leverage Ray's
    distributed processing capabilities for text analysis.
    """
    import pandas as pd
    import re

    instruction_text = inputs["instruction"]
    input_text = inputs["input"]
    output_text = inputs["output"]

    # Calculate text lengths
    instruction_length = instruction_text.str.len()
    output_length = output_text.str.len()
    input_length = input_text.str.len()
    
    # Binary feature for whether input is provided
    has_input = (input_text.str.len() > 0).astype(int)
    
    # Simple complexity score based on sentence count and word count
    def complexity_score(text_series):
        # Count sentences (rough approximation)
        sentence_count = text_series.str.count(r'[.!?]+')
        # Count words
        word_count = text_series.str.split().str.len()
        # Simple complexity metric
        return (sentence_count * 2 + word_count) / (text_series.str.len() + 1)
    
    text_complexity_score = complexity_score(instruction_text) + complexity_score(output_text)

    return pd.DataFrame(
        {
            "instruction_length": instruction_length,
            "output_length": output_length,
            "input_length": input_length,
            "text_complexity_score": text_complexity_score,
            "has_input": has_input,
        }
    )


# Feature Service - Groups related features for serving
# Ray compute engine optimizes the retrieval of these feature combinations
alpaca_basic_features = FeatureService(
    name="alpaca_basic_features",
    features=[
        alpaca_instructions_view,
    ],
    tags={"version": "v1", "compute_engine": "ray", "dataset": "alpaca"},
)

alpaca_enhanced_features = FeatureService(
    name="alpaca_enhanced_features",
    features=[
        alpaca_instructions_view,
        text_analytics_features,  # Includes on-demand NLP transformations
    ],
    tags={"version": "v2", "compute_engine": "ray", "transformations": "on_demand", "dataset": "alpaca"},
)
