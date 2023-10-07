import setuptools

project_name = "jllm"  
version = "0.0.1" 
 
setuptools.setup(
    name=project_name,
    version=version,
    author="Jian Lu",
    license="Apache 2.0",
    description=("Running Large Language Model easily, faster and low-cost"),

    url="https://github.com/janelu9/EasyLLM",
    project_urls={
        "Homepage": "https://github.com/janelu9/EasyLLM",
    },
    long_description="README.md",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires='>=3.11', 
    install_requires=[
    "deepspeed",
    "protobuf==3.20.3",
    "sentencepiece",
    "transformers",
    "pyarrow",
    "tiktoken"
    ],
)
