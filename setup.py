from setuptools import setup, find_packages

setup(
    name="codeclaw",
    version="0.1.0",
    description="CodeClaw — Open-source agentic coding CLI engine",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "anthropic>=0.40.0",
        "openai>=1.0.0",
        "rich>=13.0.0",
        "pydantic>=2.0.0",
        "httpx",
        "beautifulsoup4",
        "duckduckgo-search",
        "gitpython",
        "wcwidth",
        "pymupdf",
    ],
    entry_points={
        "console_scripts": [
            "codeclaw=codeclaw.main:main",
        ],
    },
)
