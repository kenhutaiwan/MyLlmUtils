import configparser
import os
import pathlib


config = configparser.ConfigParser()
config.read(os.path.join(pathlib.Path(__file__).parent.resolve(), 'config.ini'))


def test_sections():
    """
    除DEFAULT之外,有openai, azure兩個section
    Returns: True or False
    """
    assert len(config.sections()) == 2
    assert 'openai' in config.sections()
    assert 'azure' in config.sections()


def test_openai():
    assert config['openai']['API_KEY'] == 'OPENAI_API_KEY'
    assert config['openai']['COMPLETIONS_MODEL'] == 'gpt-4-0613'
    assert config['openai']['EMBEDDING_MODEL'] == 'text-embedding-ada-002'
    assert 'API_BASE' not in config['openai']
    assert 'API_VERSION' not in config['openai']


def test_azure():
    assert config['azure']['API_KEY'] == 'AZURE_OPENAI_API_KEY'
    assert config['azure']['API_BASE'] == 'https://openai4azurecsd.openai.azure.com/'
    assert config['azure']['API_VERSION'] == '2023-05-15'
    assert config['azure']['COMPLETIONS_MODEL'] == 'gpt-35-turbo'
    assert config['azure']['EMBEDDING_MODEL'] == 'text-embedding-ada-002'
