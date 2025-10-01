import logging
from abc import ABC, abstractmethod
from pandas import DataFrame

class Step(ABC):
    def __init__(self, step_name: str):
        self.step_name = step_name

    @abstractmethod
    def run(self, data: DataFrame) -> DataFrame:
        """
        Process the input data and return transformed data.

        Args:
            data: Input DataFrame containing the data to process

        Returns:
            DataFrame: Transformed data
        """
        logging.info(f"Running {self.step_name} step...")
        pass

class DataProcessingPipeline:
    def __init__(self, steps: list[Step]):
        self.steps = steps

    def run(self, df: DataFrame) -> DataFrame:
        for step in self.steps:
            df = step.run(df)
        return df
