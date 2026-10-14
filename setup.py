from setuptools import setup, find_packages
from pathlib import Path

setup(
    name="codeclaw",
    version="0.1.0",
    description="CodeClaw — Desktop and CLI AI coding workbench",
    long_description=Path(__file__).with_name("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/Dheudb/codeclaw",
    license="MIT",
    packages=find_packages(include=["codeclaw", "codeclaw.*"]),
    package_data={"codeclaw.skills": ["builtin/*/SKILL.md"]},
    python_requires=">=3.9",
    install_requires=[
        "anthropic>=0.40.0",
        "openai>=1.0.0",
        "rich>=13.0.0",
        "pydantic>=2.0.0",
        "httpx",
        "beautifulsoup4",
        "lxml",
        "ddgs>=9.16,<10; python_version >= '3.10'",
        "duckduckgo-search; python_version < '3.10'",
        "gitpython",
        "wcwidth",
        "pymupdf",
        "PyYAML>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "codeclaw=codeclaw.main:main",
        ],
    },
    extras_require={
        "mcp": ["mcp>=1.30,<2; python_version >= '3.10'"],
        "browser": ["playwright>=1.40"],
        "tokenizers": ["tiktoken", "transformers>=4.0"],
    },
)
