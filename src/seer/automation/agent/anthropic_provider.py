import logging
from typing import Any, Dict, Iterator, List, Optional, cast

from langfuse.decorators import langfuse_context

from seer.automation.agent.models import Message, Usage
from seer.automation.agent.utils import with_exponential_backoff

logger = logging.getLogger(__name__)

class AnthropicProvider:
    """
    Provider for Anthropic's Claude models.
    """

    def __init__(self, model: str = "claude-3-5-sonnet-v2@20241022"):
        self.model = model
        # Import anthropic only when needed to avoid unnecessary dependencies
        try:
            import anthropic
            self.client = anthropic.Client()
        except ImportError:
            logger.error("Anthropic package is not installed")
            raise

    @with_exponential_backoff(
        max_retries=5,
        initial_delay=1.0,
        max_delay=20.0,
        error_types=("overloaded_error",)
    )
    def generate_text_stream(
        self,
        max_tokens: int,
        messages: List[Message] = None,
        prompt: str = None,
        system_prompt: str = None,
        temperature: float = 0.0,
        **kwargs
    ) -> Iterator[Any]:
        """
        Stream text generation from the Anthropic API.
        
        Args:
            max_tokens: Maximum number of tokens to generate
            messages: List of messages for chat-based models
            prompt: Text prompt for completion-based models
            system_prompt: System instructions for the model
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional keyword arguments for the API
            
        Yields:
            Chunks of the generated response
        """
        try:
            import anthropic
            
            # Convert messages to Anthropic format if provided
            anthropic_messages = []
            if messages:
                for msg in messages:
                    if msg.role == "user":
                        anthropic_messages.append({"role": "user", "content": msg.content})
                    elif msg.role == "assistant":
                        anthropic_messages.append({"role": "assistant", "content": msg.content})
                    elif msg.role == "system":
                        # System messages are handled differently in Anthropic
                        system_prompt = msg.content
            
            # Create the stream
            stream = self.client.messages.create(
                model=self.model,
                messages=anthropic_messages,
                system=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs
            )
            
            # Yield from the stream
            current_tool_call = None
            current_input_json = []
            total_input_tokens = 0
            total_output_tokens = 0
            
            for chunk in stream:
                if chunk.type == "message_start" and chunk.message and hasattr(chunk.message, 'usage'):
                    if chunk.message.usage:
                        total_input_tokens += chunk.message.usage.input_tokens
                        total_output_tokens += chunk.message.usage.output_tokens
                elif chunk.type == "message_delta" and hasattr(chunk, 'usage') and chunk.usage:
                    total_output_tokens += chunk.usage.output_tokens
                
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in Anthropic text generation: {str(e)}")
            raise