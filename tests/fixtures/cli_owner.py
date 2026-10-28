"""Run the production CLI; only optional local token-encoder discovery is disabled."""
from unittest.mock import patch
from xuloom.memory.memory import MemoryCompactor
from xuloom.main import main

with patch.object(MemoryCompactor, "_resolve_token_encoder", return_value=None):
    main()
