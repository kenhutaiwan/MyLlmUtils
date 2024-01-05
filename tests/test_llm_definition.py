from MyLlmUtils.llm_definition import LlmDefinition
from MyLlmUtils.commons import ProviderType
import os
import pathlib

config_file_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'config.ini')


def test_azure():
    llm_def = LlmDefinition(ProviderType.AZURE, config_file_path)
    assert llm_def is not None
    assert llm_def.get_llm() is not None
    assert os.environ['OPENAI_API_KEY'] == 'AZURE_OPENAI_API_KEY'
    assert os.environ['OPENAI_API_TYPE'] == 'azure'
    assert os.environ['OPENAI_API_VERSION'] == '2023-05-15'
    assert os.environ['OPENAI_API_BASE'] == 'https://openai4azurecsd.openai.azure.com/'
    assert os.environ['COMPLETIONS_MODEL'] == 'gpt-35-turbo'


def test_openai():
    llm_def = LlmDefinition(ProviderType.OPENAI, config_file_path)
    assert llm_def is not None
    assert llm_def.get_llm() is not None
    assert os.environ['OPENAI_API_KEY'] == 'OPENAI_API_KEY'
