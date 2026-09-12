"""Amharic ASR SDK - Agent-ready speech recognition."""

from setuptools import setup, find_packages

setup(
    name="amharic-asr-agent",
    version="0.1.0",
    description="State-of-the-art Amharic ASR with agent tool calling",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Haile",
    author_email="haile@example.com",
    url="https://github.com/haile/amharic-asr",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.12.0",
        "onnxruntime>=1.12.0",
        "soundfile>=0.10.0",
        "librosa>=0.9.0",
        "pydantic>=1.10.0",
        "fastapi>=0.78.0",
        "uvicorn>=0.18.0",
        "python-multipart>=0.0.5",
        "requests>=2.28.0",
        "aiohttp>=3.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.960",
        ],
        "tensorrt": [
            "tensorrt>=8.0.0",
            "pycuda>=2021.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "amharic-asr=amharic_asr.cli:main",
            "amharic-asr-server=api.server:app",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
)
