# coding: utf-8
import io
import os

from setuptools import find_packages, setup

NAME = "neurst"
DESCRIPTION = "Neural Speech Translation Toolkit"
URL = "https://github.com/bytedance/neurst"
EMAIL = "zhaochengqi.d@bytedance.com"
AUTHOR = "ZhaoChengqi"

# TODO: one must manually install following packages if needed
ALTERNATIVE_REQUIRES = [
    "jieba>=0.42.1",  # unnecessary for all
    "subword-nmt>=0.3.7",  # unnecessary for all
    "thai-segmenter>=0.4.1",  # unnecessary for all
    "soundfile>=0.10",  # for speech processing
    "python_speech_features>=0.6",  # for speech processing
    "transformers>=3.4.0",  # Not necessary for all
    "sentencepiece>=0.1.7",
    "mecab-python3>=1.0.3",  # for sacrebleu[ja]
    "ipadic>=1.0.0",  # for sacrebleu[ja]
    "torch>=1.7.0",  # for converting models from fairseq
    "fairseq>=0.10.1",  # for converting models from fairseq
    "tensorflow_addons>=0.11.2",  # for group normalization
    "pydub>=0.24.1",  # for audio processing
    "sox>=1.4.1",  # for audio processing
]

REQUIRES = ["six>=1.11.0,<2.0.0",
            "pyyaml>=3.13",
            "sacrebleu>=1.4.0",
            "regex>=2019.1.24",
            "sacremoses>=0.0.38",
            # "tensorflow>=2.4.0", # ONE must manually install tensorflow
            "tqdm>=0.46",
            ]

DEV_REQUIRES = ["flake8>=3.5.0,<4.0.0",
                "mypy>=0.620; python_version>='3.6'",
                "tox>=3.0.0,<4.0.0",
                "isort>=4.0.0,<5.0.0",
                "pytest>=4.0.0,<5.0.0"] + REQUIRES + ALTERNATIVE_REQUIRES

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except IOError:
    long_description = DESCRIPTION

about = {}
with io.open(os.path.join(here, NAME, "__version__.py")) as f:
    exec(f.read(), about)

setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8"
    ],
    keywords="neurst",
    packages=find_packages(exclude=["docs", "tests"]),
    install_requires=REQUIRES,
    tests_require=[
        "pytest>=4.0.0,<5.0.0"
    ],
    python_requires=">=3.6",
    extras_require={
        "dev": DEV_REQUIRES,
    },
    package_data={
        # for PEP484 & PEP561
        NAME: ["py.typed", "*.pyi"],
    },
    entry_points={
        "console_scripts": [
            "neurst-run = neurst.cli.run_exp:cli_main",
            "neurst-view = neurst.cli.view_registry:cli_main",
            "neurst-vocab = neurst.cli.generate_vocab:cli_main"
        ],
    },
)
