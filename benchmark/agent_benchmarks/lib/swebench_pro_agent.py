"""Keep the official locked protocol while pinning its LiteLLM runtime extras."""

from locked_mini_swe import LockedMiniSwe


class V2MiniSwe(LockedMiniSwe):
    async def install(self, environment):
        await super().install(environment)
        # The official installer requests bare litellm. Its current completion
        # path imports MCP lazily, so --help succeeds but the first call fails.
        await self.exec_as_agent(
            environment,
            command=(
                'export PATH="$HOME/.local/bin:$PATH"; '
                'uv pip install --python "$(uv tool dir)/mini-swe-agent/bin/python" '
                "'litellm[mcp]==1.106.2' && "
                '"$(uv tool dir)/mini-swe-agent/bin/python" '
                "-c 'import litellm.responses.mcp.chat_completions_handler'"
            ),
        )
