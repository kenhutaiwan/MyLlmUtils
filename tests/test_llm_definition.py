from khu_llm_toolkit.model_definition import ModelDefinition
from khu_llm_toolkit.commons import ProviderType
import os
import pathlib
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings

config_file_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'config.ini')


def test_azure():
    llm_def = ModelDefinition(config_file_path)
    assert llm_def is not None
    llm, embeddings = llm_def.get_model('azurecsd-aoai-gpt-35'), llm_def.get_model('azurecsd-aoai-embeddings')
    assert llm is not None
    assert embeddings is not None
    assert isinstance(llm, AzureChatOpenAI)
    assert isinstance(embeddings, AzureOpenAIEmbeddings)
    assert llm.openai_api_version == '2024-02-01'
    assert llm.model_name == 'gpt-3.5-turbo'
    assert llm.azure_endpoint == 'FAKE DATA'
    assert embeddings.model == 'text-embedding-ada-002'


def test_openai():
    llm_def = ModelDefinition(config_file_path)
    assert llm_def is not None
    llm, embeddings = llm_def.get_model('kenhu-openai-gpt-4'), llm_def.get_model('kenhu-openai-embeddings')
    assert llm is not None
    assert embeddings is not None
    assert isinstance(llm, ChatOpenAI)
    assert isinstance(embeddings, OpenAIEmbeddings)
    assert llm.model_name == 'gpt-4'
    assert embeddings.model == 'text-embedding-ada-002'

