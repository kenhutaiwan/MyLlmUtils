import configparser
import os
import pathlib


config = configparser.ConfigParser()
config.read(os.path.join(pathlib.Path(__file__).parent.resolve(), 'config.ini'))


def test_sections():
    """
    除DEFAULT之外,有openai, azure和pinecone三個section
    Returns: True or False
    """
    assert len(config.sections()) == 3
    assert 'openai' in config.sections()
    assert 'azure' in config.sections()
    assert 'pinecone' in config.sections()


def test_openai():
    assert config['openai']['USE_AZURE'] == 'False'
    assert config['openai']['OPENAI_API_KEY'] == 'OPENAI_API_KEY'
    assert config['openai']['COMPLETIONS_MODEL_NAME'] == 'gpt-4-0613'
    assert config['openai']['EMBEDDING_MODEL_NAME'] == 'text-embedding-ada-002'


def test_azure():
    assert config['azure']['USE_AZURE'] == 'True'
    assert config['azure']['OPENAI_API_KEY'] == 'AZURE_OPENAI_API_KEY'
    assert config['azure']['OPENAI_API_BASE'] == 'https://openai4azurecsd.openai.azure.com/'
    assert config['azure']['OPENAI_API_VERSION'] == '2023-05-15'
    assert config['azure']['COMPLETIONS_TYPE'] == 'chat'
    assert config['azure']['COMPLETIONS_MODEL_NAME'] == 'gpt-35-turbo'
    assert config['azure']['EMBEDDING_MODEL_NAME'] == 'text-embedding-ada-002'


def test_pinecone():
    assert config['pinecone']['PINECONE_API_KEY'] == 'PINECONE_API_KEY'
    assert config['pinecone']['PINECONE_ENV'] == 'us-central1-gcp'
    assert config['pinecone']['PINECONE_INDEX'] == 'rdtwo23h1-24'