# 대회 서버에 PORORO 설치하기

저희는 오피스아워에서 소개된 [PORORO](https://github.com/kakaobrain/pororo)를 사용해 역번역으로 리트리버와 리더 질문 증강을 진행했습니다. 하지만 공식 설치 방법이 잘 동작하지 않아서 설치 과정에서 발생한 여러 문제를 공유하고 좋은 방법을 찾기 위해 저희가 경험한 대회 서버에 PORORO 설치하는 방법을 공유합니다. 다만 패키지 버전 문제가 많이 발생해 좋은 방법이라기보다는, 이렇게도 설치해서 사용할 수 있구나 하고 봐주시면 감사하겠습니다.

우선, 아무것도 설치되지 않은 상태의 서버에서 새로운 conda 환경을 만들고 pororo를 공식 설치 방법에 따라 설치합니다.

```bash
conda create -y --prefix ~/shark python=3.8
conda init
conda activate ~/shark
pip install pororo
```

이렇게 하면 설치가 잘 된 듯 하지만, pororo를 import하면 아래와 같은 오류가 발생합니다.

## 실험 파일 (test.py)

```
from pororo import Pororo
mt = Pororo(task="translation", lang="multi")
print(mt("死神は りんごしか食べない。", src="ja", tgt="ko"))
```

## 첫 번째 에러: ModuleNotFoundError: No module named torch.utils._pytree

```
(/data/ephemeral/home/shark) root@instance-12649:~# python ~/test.py
Traceback (most recent call last):
  File "/data/ephemeral/home/test.py", line 1, in <module>
    import pororo
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/pororo/__init__.py", line 2, in <module>
    from pororo.pororo import Pororo  # noqa
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/pororo/pororo.py", line 10, in <module>
    from pororo.tasks.utils.base import PororoTaskBase
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/pororo/tasks/__init__.py", line 20, in <module>
    from pororo.tasks.age_suitability import PororoAgeSuitabilityFactory
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/pororo/tasks/age_suitability.py", line 8, in <module>
    from transformers import RobertaModel
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/transformers/__init__.py", line 26, in <module>
    from . import dependency_versions_check
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/transformers/dependency_versions_check.py", line 16, in <module>
    from .utils.versions import require_version, require_version_core
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/transformers/utils/__init__.py", line 37, in <module>
    from .generic import (
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/transformers/utils/generic.py", line 462, in <module>
    import torch.utils._pytree as _torch_pytree
ModuleNotFoundError: No module named 'torch.utils._pytree'
```
### 해결 방법

- 현재 버전의 torch의 utils에 _pytree를 직접 추가합니다

```
wget -P ~/shark/lib/python3.8/site-packages/torch/utils https://raw.githubusercontent.com/pytorch/pytorch/v1.11.0/torch/utils/_pytree.py
```


## 두 번째 에러: AttributeError: module 'torch.nn' has no attribute 'SiLU'

```
(/data/ephemeral/home/shark) root@instance-12649:~# python test.py
Traceback (most recent call last):
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/transformers/utils/import_utils.py", line 1764, in _get_module
    return importlib.import_module("." + module_name, self.__name__)
  File "/data/ephemeral/home/shark/lib/python3.8/importlib/__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1014, in _gcd_import
  File "<frozen importlib._bootstrap>", line 991, in _find_and_load
  File "<frozen importlib._bootstrap>", line 975, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 671, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 843, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/transformers/models/roberta/modeling_roberta.py", line 27, in <module>
    from ...activations import ACT2FN, gelu
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/transformers/activations.py", line 217, in <module>
    "silu": nn.SiLU,
AttributeError: module 'torch.nn' has no attribute 'SiLU'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "test.py", line 1, in <module>
    import pororo
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/pororo/__init__.py", line 2, in <module>
    from pororo.pororo import Pororo  # noqa
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/pororo/pororo.py", line 10, in <module>
    from pororo.tasks.utils.base import PororoTaskBase
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/pororo/tasks/__init__.py", line 20, in <module>
    from pororo.tasks.age_suitability import PororoAgeSuitabilityFactory
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/pororo/tasks/age_suitability.py", line 8, in <module>
    from transformers import RobertaModel
  File "<frozen importlib._bootstrap>", line 1039, in _handle_fromlist
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/transformers/utils/import_utils.py", line 1755, in __getattr__
    value = getattr(module, name)
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/transformers/utils/import_utils.py", line 1754, in __getattr__
    module = self._get_module(self._class_to_module[name])
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/transformers/utils/import_utils.py", line 1766, in _get_module
    raise RuntimeError(
RuntimeError: Failed to import transformers.models.roberta.modeling_roberta because of the following error (look up to see its traceback):
module 'torch.nn' has no attribute 'SiLU'
```

### 해결 방법

- 현재 버전의 torch의 SiLU를 직접 추가합니다
- `code ~/shark/lib/python3.8/site-packages/torch`를 통해 VS Code로 편집하는 것을 권장합니다


```py
# torch/nn/modules/__init__.py

from .activation import SiLU, ...

...

__all__ = [ 'SiLU', ... ]
```

```py
# torch/nn/functional.py

...

def silu(input, inplace: bool = False):
    r"""Applies the Sigmoid Linear Unit (SiLU) function, element-wise.

    See :class:`~torch.nn.SiLU` for more details.
    """
    if inplace:
        return input.mul_(torch.sigmoid(input))
    else:
        return input * torch.sigmoid(input)

...
```

```py
# torch/nn/modules/activation.py

class SiLU(Module):
    r"""Applies the Sigmoid Linear Unit (SiLU) function element-wise:
    :math:`\text{SiLU}(x) = x * \sigma(x)`, where :math:`\sigma(x)` is the logistic sigmoid.

    .. math::
        \text{SiLU}(x) = x * \sigma(x) = \frac{x}{1 + \exp(-x)}
    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        super(SiLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.silu(input, inplace=self.inplace)

    def extra_repr(self):
        return 'inplace={}'.format(self.inplace)
```

## 세 번째 에러: ImportError: cannot import name 'cached_download' from 'huggingface_hub'

```
(/data/ephemeral/home/shark) root@instance-12649:~# python test.py
Traceback (most recent call last):
  File "test.py", line 1, in <module>
    import pororo
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/pororo/__init__.py", line 2, in <module>
    from pororo.pororo import Pororo  # noqa
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/pororo/pororo.py", line 10, in <module>
    from pororo.tasks.utils.base import PororoTaskBase
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/pororo/tasks/__init__.py", line 43, in <module>
    from pororo.tasks.sentence_embedding import PororoSentenceFactory
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/pororo/tasks/sentence_embedding.py", line 6, in <module>
    from sentence_transformers import util
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/sentence_transformers/__init__.py", line 3, in <module>
    from .datasets import SentencesDataset, ParallelSentencesDataset
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/sentence_transformers/datasets/__init__.py", line 3, in <module>
    from .ParallelSentencesDataset import ParallelSentencesDataset
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/sentence_transformers/datasets/ParallelSentencesDataset.py", line 4, in <module>
    from .. import SentenceTransformer
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/sentence_transformers/SentenceTransformer.py", line 12, in <module>
    from huggingface_hub import HfApi, HfFolder, Repository, hf_hub_url, cached_download
ImportError: cannot import name 'cached_download' from 'huggingface_hub' (/data/ephemeral/home/shark/lib/python3.8/site-packages/huggingface_hub/__init__.py)
```

### 해결 방법

- sentence_transformers에서 cached_download 대신 hf_hub_download를 사용하도록 바꿉니다
- cached_download_args 부분부터 아래의 download_args로 수정해주세요
- `code ~/shark/lib/python3.8/site-packages/sentence_transformers`를 통해 VS Code로 편집하는 것을 권장합니다

```py
# sentence_transformers/SentenceTransformers.py

from huggingface_hub import HfApi, hf_hub_url, HfFolder
```

```py
# sentence_transformers/utils.py

from huggingface_hub import HfApi, hf_hub_url, hf_hub_download, HfFolder

...

def snapshot_download(
   ...
   download_args = {'url': url,
     'cache_dir': storage_folder,
     'force_filename': relative_filepath,
     'library_name': library_name,
     'library_version': library_version,
     'user_agent': user_agent,
     'use_auth_token': use_auth_token}
   path = hf_hub_download(**download_args)
...
```

## 네 번째 에러: AttributeError: module 'numpy' has no attribute 'float'

```
(/data/ephemeral/home/shark) root@instance-12649:~# python test.py
Traceback (most recent call last):
  File "test.py", line 1, in <module>
    import pororo
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/pororo/__init__.py", line 2, in <module>
    from pororo.pororo import Pororo  # noqa
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/pororo/pororo.py", line 10, in <module>
    from pororo.tasks.utils.base import PororoTaskBase
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/pororo/tasks/__init__.py", line 46, in <module>
    from pororo.tasks.text_summarization import PororoSummarizationFactory
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/pororo/tasks/text_summarization.py", line 6, in <module>
    from fairseq import hub_utils
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/fairseq/__init__.py", line 19, in <module>
    import fairseq.criterions  # noqa
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/fairseq/criterions/__init__.py", line 13, in <module>
    from fairseq.criterions.fairseq_criterion import (  # noqa
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/fairseq/criterions/fairseq_criterion.py", line 9, in <module>
    from fairseq import metrics, utils
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/fairseq/utils.py", line 20, in <module>
    from fairseq.data import iterators
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/fairseq/data/__init__.py", line 23, in <module>
    from .indexed_dataset import (
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/fairseq/data/indexed_dataset.py", line 101, in <module>
    6: np.float,
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/numpy/__init__.py", line 305, in __getattr__
    raise AttributeError(__former_attrs__[attr])
AttributeError: module 'numpy' has no attribute 'float'.
`np.float` was a deprecated alias for the builtin `float`. To avoid this error in existing code, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
```
### 해결 방법

- numpy의 float 대신 python의 내장형 float를 사용합니다
- `code ~/shark/lib/python3.8/site-packages/fairseq`를 통해 VS Code로 편집하는 것을 권장합니다
- `Ctrl + Shift + F`로 `np.float`를 검색 후 전부 `float`로 변경합니다

## 다섯 번째 에러: TypeError: Metaspace.__new__() got an unexpected keyword argument 'add_prefix_space'

```
(/data/ephemeral/home/shark) root@instance-12649:~# python test.py
100% [......................................................................] 447581863 / 447581863
100% [............................................................................] 551947 / 551947
100% [............................................................................] 459685 / 459685
Traceback (most recent call last):
  File "test.py", line 2, in <module>
    mt = Pororo(task="translation", lang="multi")
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/pororo/pororo.py", line 203, in __new__
    task_module = SUPPORTED_TASKS[task](
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/pororo/tasks/machine_translation.py", line 173, in load
    tokenizer = CustomTokenizer.from_file(
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/pororo/tasks/utils/tokenizer.py", line 75, in from_file
    return CustomTokenizer(vocab, merges, **kwargs)
  File "/data/ephemeral/home/shark/lib/python3.8/site-packages/pororo/tasks/utils/tokenizer.py", line 37, in __init__
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
TypeError: Metaspace.__new__() got an unexpected keyword argument 'add_prefix_space'
```
### 해결 방법

- pororo에서 사용하는 pre_tokenizer의 Metaspace 생성 인자를 변경합니다
- `code ~/shark/lib/python3.8/site-packages/pororo`를 통해 VS Code로 편집하는 것을 권장합니다

```py
# pororo/tasks/utils/tokenizer.py

class CustomTokenizer(BaseTokenizer):
    def __init__(
      ...
      tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
          replacement=replacement,
          prepend_scheme="first",
          split=True,
      )

      tokenizer.decoder = decoders.Metaspace(
          replacement=replacement,
          prepend_scheme="first",
          split=True,
      )
```
여기까지 진행하셨다면 test.py를 실행해 Machine Translation 동작을 확인할 수 있습니다.

```bash
(/data/ephemeral/home/shark) root@instance-12649:~# python test.py
사신은 사과밖에 먹지 않는다.
```

끝까지 읽어주셔서 감사합니다.
