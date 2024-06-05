from khu_llm_toolkit.model_definition import ModelDefinition
from khu_llm_toolkit.commons import ProviderType, FrameworkType
import os
import pathlib
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings
from llama_index.llms.azure_openai import AzureOpenAI as LlamaIndexAzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding as LlamaIndexAzureOpenAIEmbeddings
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding as LlamaIndexOpenAIEmbeddings

config_file_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'test-config.ini')


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


def test_azure_llamaindex():
    llm_def = ModelDefinition(config_file_path)
    assert llm_def is not None
    llm, embeddings = llm_def.get_model('azurecsd-aoai-gpt-35', framework=FrameworkType.LLAMA_INDEX), llm_def.get_model('azurecsd-aoai-embeddings', framework=FrameworkType.LLAMA_INDEX)
    assert llm is not None
    assert embeddings is not None
    assert isinstance(llm, LlamaIndexAzureOpenAI)
    assert isinstance(embeddings, LlamaIndexAzureOpenAIEmbeddings)
    assert llm.api_version == '2024-02-01'
    assert llm.model == 'gpt-35-turbo'
    assert llm.azure_endpoint == 'FAKE DATA'
    assert embeddings.azure_endpoint == 'FAKE DATA'
    assert embeddings.api_version == '2024-02-01'


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


def test_openai_llamaindex():
    llm_def = ModelDefinition(config_file_path)
    assert llm_def is not None
    llm, embeddings = llm_def.get_model('kenhu-openai-gpt-4', framework=FrameworkType.LLAMA_INDEX), llm_def.get_model('kenhu-openai-embeddings', framework=FrameworkType.LLAMA_INDEX)
    assert llm is not None
    assert embeddings is not None
    assert isinstance(llm, LlamaIndexOpenAI)
    assert isinstance(embeddings, LlamaIndexOpenAIEmbeddings)

