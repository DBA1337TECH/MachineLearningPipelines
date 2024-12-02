import os
from tfx.components import (
    ExampleGen,
    Transform,
    Trainer,
    Evaluator,
    Pusher,
)
from tfx.proto import trainer_pb2
from tfx.orchestration import pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, Examples, Schema
from tfx.utils.dsl_utils import external_input

# Define paths
_pipeline_name = "steganography_pipeline"
_pipeline_root = os.path.join("pipelines", _pipeline_name)
_data_root = "data/steganography"  # Path to your dataset
_module_file = "steganography_trainer.py"  # Path to the trainer module
_serving_model_dir = os.path.join("serving_models", _pipeline_name)

def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_root: str,
    module_file: str,
    serving_model_dir: str,
):
    """Creates a TFX pipeline for the steganography project."""

    # Input data
    example_gen = ExampleGen(input=external_input(data_root))

    # Transform component (add feature engineering if needed)
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=Channel(type=Schema),
        module_file=module_file,
    )

    # Trainer component
    trainer = Trainer(
        module_file=module_file,
        examples=transform.outputs["transformed_examples"],
        schema=transform.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(num_steps=200),
        eval_args=trainer_pb2.EvalArgs(num_steps=50),
    )

    # Evaluator
    evaluator = Evaluator(
        examples=transform.outputs["transformed_examples"],
        model=trainer.outputs["model"],
    )

    # Pusher component
    pusher = Pusher(
        model=trainer.outputs["model"],
        push_destination=trainer_pb2.PushDestination(
            filesystem=trainer_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[example_gen, transform, trainer, evaluator, pusher],
        enable_cache=True,
    )

if __name__ == "__main__":
    # Define the pipeline
    steganography_pipeline = create_pipeline(
        pipeline_name=_pipeline_name,
        pipeline_root=_pipeline_root,
        data_root=_data_root,
        module_file=_module_file,
        serving_model_dir=_serving_model_dir,
    )

    # Run the pipeline locally
    LocalDagRunner().run(steganography_pipeline)

