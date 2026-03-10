"""
Setup configuration for Federated Learning Healthcare AI
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
req_file = this_directory / "requirements.txt"
if req_file.exists():
    with open(req_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)

setup(
    name="federated-learning-healthcare-ai",
    version="1.0.0",
    author="Pranay M Mahendrakar",
    author_email="pranaymahendrakar2001@gmail.com",
    description=(
        "Privacy-Preserving Healthcare AI using Federated Learning "
        "on distributed hospital ECG and patient monitoring data"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PranayMahendrakar/federated-learning-healthcare-ai",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "plotly>=5.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fl-server=server.fl_server:main",
            "fl-client=client.fl_client:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "federated-learning",
        "healthcare-ai",
        "differential-privacy",
        "ECG",
        "arrhythmia",
        "privacy-preserving",
        "deep-learning",
        "PyTorch",
        "Flower",
    ],
    project_urls={
        "Bug Reports": "https://github.com/PranayMahendrakar/federated-learning-healthcare-ai/issues",
        "Source": "https://github.com/PranayMahendrakar/federated-learning-healthcare-ai",
        "Documentation": "https://github.com/PranayMahendrakar/federated-learning-healthcare-ai#readme",
    },
)
