import configparser
import os
import pathlib


config = configparser.ConfigParser()
config.read(os.path.join(pathlib.Path(__file__).parent.resolve(), 'test-config.ini'))


def test_sections():
    """
    除DEFAULT之外,有openai, azure兩個section
    Returns: True or False
    """
    assert len(config.sections()) == 6
    assert 'azurecsd-aoai-gpt-35' in config.sections()
    assert 'azurecsd-aoai-embeddings' in config.sections()


def test_openai():
    assert config['kenhu-openai-gpt-4']['API_KEY'] == 'FAKE DATA'
    assert config['kenhu-openai-gpt-4']['COMPLETIONS_MODEL'] == 'gpt-4'
    assert 'API_BASE' not in config['kenhu-openai-gpt-4']
    assert 'API_VERSION' not in config['kenhu-openai-gpt-4']


def test_azure():
    assert config['azurecsd-aoai-gpt-35']['API_KEY'] == 'FAKE DATA'
    assert config['azurecsd-aoai-gpt-35']['API_BASE'] == 'FAKE DATA'
    assert config['azurecsd-aoai-gpt-35']['API_VERSION'] == '2024-02-01'
    assert config['azurecsd-aoai-gpt-35']['COMPLETIONS_MODEL'] == 'gpt-35-turbo'
