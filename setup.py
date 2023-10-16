import setuptools
import io

project_name = "jllm"  
version = "1.1.5" 
 
setuptools.setup(
    name=project_name,
    version=version,
    author="Jian Lu",
    license="Apache 2.0",
    description=("Running Large Language Model easily, faster and low-cost."),

    url="https://github.com/janelu9/EasyLLM",
    project_urls={
        "Homepage": "https://github.com/janelu9/EasyLLM",
    },
    long_description=io.open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires='>=3.9', 
    install_requires=[
    "deepspeed",
    "protobuf",
    "sentencepiece",
    "transformers",
    "pyarrow",
    "tiktoken"
    ],
)
