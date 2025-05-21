from setuptools import setup, find_packages

setup(
    name="moe_lora",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "loguru>=0.7.0",
        "tqdm>=4.65.0",
        "accelerate>=0.20.0",
        "peft>=0.4.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="扩展式适配器架构框架",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/moe_lora",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 