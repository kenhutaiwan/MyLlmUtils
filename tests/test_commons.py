from MyLlmUtils.commons import ProviderType


def test_provider_type():
    config_dict = {'azure': 1, 'openai': 2}

    assert ProviderType.AZURE.value == 'azure'
    assert ProviderType.OPENAI.value == 'openai'
    assert f'/folder/{ProviderType.AZURE.value}' == '/folder/azure'
    assert config_dict[ProviderType.AZURE.value] == 1
    assert config_dict[ProviderType.OPENAI.value] == 2
