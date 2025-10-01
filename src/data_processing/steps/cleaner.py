import re
from pandas import DataFrame, Series
from src.data_processing.processing_pipeline import Step

# Pre-compile patterns once
url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F]{2}))+')
www_pattern = re.compile(r'www\.(?:[a-zA-Z0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F]{2}))+')
special_pattern = re.compile(r'[^a-zA-Z0-9\s\.,!?\'-]')

class Cleaner(Step):
    def _clean_text(self, s: Series) -> Series:
        s = s.fillna("")
        s = s.str.replace(url_pattern, "", regex=True)
        s = s.str.replace(www_pattern, "", regex=True)
        s = s.str.replace(special_pattern, "", regex=True)
        s = s.str.replace(r'\s+', " ", regex=True).str.strip()
        return s

    def run(self, data: DataFrame) -> DataFrame:
        """
        Run data cleaning pipeline step

        Args:
            data: Input DataFrame containing the data to clean

        Returns:
            DataFrame: Cleaned data
        """
        data.loc[:, "text"] = self._clean_text(data["text"])
        return data