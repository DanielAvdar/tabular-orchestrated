import dataclasses
from typing import Tuple

from ml_orchestrator import artifacts
from ml_orchestrator.artifacts import Input, Output
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from tabular_orchestrated.tab_comp import TabComponent


@dataclasses.dataclass
class DataSplitter(TabComponent):
    extra_packages = ["spliter"]
    dataset: Input[artifacts.Dataset] = None
    train_dataset: Output[artifacts.Dataset] = None
    test_dataset: Output[artifacts.Dataset] = None
    test_size: float = 0.2
    random_state: int = 42
    shuffle: bool = True

    def execute(self) -> None:
        df: DataFrame = self.load_df(self.dataset)
        df1, df2 = self.split_dataframe(df)
        self.save_df(df1, self.train_dataset)
        self.save_df(df2, self.test_dataset)

    def split_dataframe(self, df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        df1, df2 = train_test_split(df, test_size=self.test_size, random_state=self.random_state, shuffle=self.shuffle)
        return df1, df2
