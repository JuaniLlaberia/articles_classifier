import numpy as np
from tqdm import tqdm
from pandas import DataFrame
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.data_processing.processing_pipeline import Step

class Embedder(Step):
    def __init__(self,
                 step_name: str,
                 model_name: str,
                 chunk_size: int,
                 chunk_overlap: int,
                 threshold: int):
        """
        Initializes a new instance of Embedder class

        Args:
            model_name (str): Ollama embedding model name
            chunk_size (int): Size of chunks in case we need chunking
            chunk_overlap (int): Size of chunks overlap in case we need chunking
            threshold (int): Max. number of charactersto which we don't chunk
        """
        self.step_name = step_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.threshold = threshold

        self.model = OllamaEmbeddings(model=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def _embed_batches(self, texts: list[str], batch_size: int) -> list[list[float]]:
        """
        Embed articles batches and implement chunking when needed

        Args:
            texts (list[str]): List of articles to embed
            batch_size (int): Amount of articles per batch
        Returns:
            list[list[float]]: List of embeddings
        """
        final_embeddings = []
        print(f"Total number of records: {len(texts)}")

        for start_index in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[start_index:start_index + batch_size]

            # 1: Prepare all texts for embedding
            prepared_texts = []
            for text in batch_texts:
                if len(text) > self.threshold:
                    prepared_texts.append(self.text_splitter.split_text(text))
                else:
                    prepared_texts.append([text])

            # 2: Flatten data
            all_texts = [text for text_group in prepared_texts for text in text_group]
            all_embeddings = self.model.embed_documents(all_texts)

            # 3: Reconstruct embeddings for each original text
            embed_idx = 0
            for text_group in prepared_texts:
                if len(text_group) == 1:
                    final_embeddings.append(all_embeddings[embed_idx])
                else:
                    chunk_embeddings = all_embeddings[embed_idx:embed_idx + len(text_group)]
                    final_embeddings.append(np.mean(chunk_embeddings, axis=0))

                embed_idx += len(text_group)

        return final_embeddings

    def run(self, data: DataFrame) -> DataFrame:
        """
        Run embedding pipeline step with parallel processing

        Args:
            data: Input DataFrame containing the data to embed

        Returns:
            DataFrame: Data with embeddings
        """
        texts = data['text'].tolist()
        embeddings = self._embed_batches(texts=texts, batch_size=40)

        data['embedding'] = embeddings
        return data
