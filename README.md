<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<h3 align="center">My LLM Utilities</h3>

<!-- ABOUT THE PROJECT -->
## About The Project

<p>
個人工具箱，便於生成式AI的開發探索。私人研究，與工作無關，來窺伺動靜的人請自行退去。
</p>

### 2024-02-19

修改專案及套件名稱

### Built With

* [![Python][Python.org]][Python-url]
* [![OpenAI][OpenAI.com]][OpenAI-url]


<!-- GETTING STARTED -->
## Getting Started

本節說明如何建置本專案, 以及如何將建置成果供用戶使用.

### Prerequisites

開發環境需要先安裝好Python和poetry

* pyenv
  
  ```shell
  curl https://pyenv.run | bash
  ```
* Python

  ```sh
  pyenv install 3.11
  ```
  
* poetry
  
  ```shell
  curl -sSL https://install.python-poetry.org | python3 -
  ```

* Virtual environment

  ```
  poetry install
  poetry shell
  ```

### Development

see [How to upload your python package to PyPi](https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56)

#### 使用 Poetry

* Virtual environment

  ```
  poetry shell
  ```

* Test

    ```
    pytest 
    ```

* Build

  ```
  poetry build
  ```

* Upload

  ```
  poetry publish --username=__token__ --password=pypi token值
  ```

#### 手動執行 setuptools

* Virtual environment

  ```
  source .venv/bin/activate
  ```

* Test

    ```
    pytest 
    ```

* Github Release

  建立並發布git tag

  ```
  git tag -a 0.1.1 -m "adjust config file content layout"
  git push origin 0.1.1
  ```

  然後到Github倉庫頁面建立一個新的release

  複製Assets中Source code(.tar.gz)的URL, 把它貼到setup.py裡的download_url中

* Build

  ```
  python3 setup.py sdist
  python3 setup.py clean --all
  ```

* Upload

  PyPi不再允許在上傳過程中用個人帳號密碼做為身份驗證方式, 參考[Hackmd記錄](https://hackmd.io/4zug-RFaS362quf2Qfj2CA#2023-09-21)。
  在上傳過程中要求認證的時候, 以"__token__"做為username, 以該檔案中的token值做為密碼 

  ```
  twine upload dist/*
  ```

<!-- USAGE EXAMPLES -->
## Usage

完成建置並推送到PyPi server後, 就可以在其它專案中將它設為相依套件, 並在程式碼中使用.

* 相依套件

Poetry: 在 pyproject.toml 填入  

```
khu_llm_toolkit = "^0.1.6"
```

Pip: 在 requirements.txt 填入  

```
khu_llm_toolkit >= 0.1.6
```

* 匯入套件

```
from khu_llm_toolkit import ModelDefinition
from khu_llm_toolkit.commons import ProviderType

config_file_path = os.path.join(os.getcwd(), f"instance/model_definition.ini")
model_def = ModelDefinition(ProviderType.AZURE, config_file_path)
llm, embeddings = model_def.get_models(temperature=temperature)
```

* 設定檔範例

```
[DEFAULT]

[openai]
USE_AZURE = False
API_KEY = OPENAI_API_KEY
CHAT_COMPLETIONS_MODEL = gpt-4-0613
EMBEDDINGS_MODEL = text-embedding-ada-002

[azure]
USE_AZURE = True
API_KEY = AZURE_OPENAI_API_KEY
API_BASE = AZURE_OPENAI_API_ENDPOINT_URL
API_VERSION = 2023-05-15
COMPLETIONS_MODEL = gpt-35-turbo
EMBEDDINGS_MODEL = text-embedding-ada-002
```

<!-- ROADMAP -->
## Roadmap

- [x] 支援 Gemini (0.1.7 2024-04-25)
- [x] 一設定檔,多模型 (0.1.7 2024-04-25)
  - [x] 排除Gemini Embeddings模型使用錯誤問題 (0.1.8 2024-06-15)    
- [ ] 支援 Huggingface上的開源模型

See the [open issues](https://github.com/kenhutaiwan/MyLlmUtils/issues) for a full list of proposed features (and known issues).

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Ken Hu - kenhu@duck.com

Project Link: [https://github.com/kenhutaiwan/MyLlmUtils](https://github.com/kenhutaiwan/MyLlmUtils)



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/kenhutaiwan/MyLlmUtils.svg?style=for-the-badge
[contributors-url]: https://github.com/kenhutaiwan/MyLlmUtils/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/kenhutaiwan/MyLlmUtils.svg?style=for-the-badge
[forks-url]: https://github.com/kenhutaiwan/MyLlmUtils/network/members
[stars-shield]: https://img.shields.io/github/stars/kenhutaiwan/MyLlmUtils.svg?style=for-the-badge
[stars-url]: https://github.com/kenhutaiwan/MyLlmUtils/stargazers
[issues-shield]: https://img.shields.io/github/issues/kenhutaiwan/MyLlmUtils.svg?style=for-the-badge
[issues-url]: https://github.com/kenhutaiwan/MyLlmUtils/issues
[license-shield]: https://img.shields.io/github/license/kenhutaiwan/MyLlmUtils.svg?style=for-the-badge
[license-url]: https://github.com/kenhutaiwan/MyLlmUtils/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
[Python.org]: https://img.shields.io/badge/Python-00FFEE?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[OpenAI.com]: https://img.shields.io/badge/OpenAI-666666?style=for-the-badge&logo=openai&logoColor=white
[OpenAI-url]: https://openai.com/