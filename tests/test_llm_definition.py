from khu_llm_toolkit.model_definition import ModelDefinition
from khu_llm_toolkit.commons import ProviderType
import os
import pathlib

config_file_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'config.ini')


def test_azure():
    llm_def = ModelDefinition(ProviderType.AZURE, config_file_path)
    assert llm_def is not None
    llm, embeddings = llm_def.get_models()
    assert llm is not None
    assert embeddings is not None
    assert os.environ['OPENAI_API_KEY'] == 'AZURE_OPENAI_API_KEY'
    assert os.environ['OPENAI_API_TYPE'] == 'azure'
    assert os.environ['OPENAI_API_VERSION'] == '2023-05-15'
    assert os.environ['AZURE_OPENAI_ENDPOINT'] == 'https://openai4azurecsd.openai.azure.com/'
    assert os.environ['COMPLETIONS_MODEL'] == 'gpt-35-turbo'


def test_openai():
    llm_def = ModelDefinition(ProviderType.OPENAI, config_file_path)
    assert llm_def is not None
    llm, embeddings = llm_def.get_models()
    assert llm is not None
    assert embeddings is not None
    assert os.environ['OPENAI_API_KEY'] == 'OPENAI_API_KEY'
