import os
import asyncio

from aviary.env import TaskDataset
from ldp.agent import SimpleAgent
from ldp.alg.callbacks import MeanMetricsCallback
from ldp.alg.runners import Evaluator, EvaluatorConfig

from paperqa import QueryRequest, Settings
from paperqa.agents.task import TASK_DATASET_NAME

async def evaluate(settings: Settings) -> None:
    base_query = QueryRequest(
        settings=settings
    )
    dataset = TaskDataset.from_name(TASK_DATASET_NAME, base_query=base_query)
    metrics_callback = MeanMetricsCallback(eval_dataset=dataset)

    evaluator = Evaluator(
        config=EvaluatorConfig(batch_size=3),
        agent=settings.agent,
        dataset=dataset,
        callbacks=[metrics_callback],
    )
    await evaluator.evaluate()

    print(metrics_callback.eval_means)