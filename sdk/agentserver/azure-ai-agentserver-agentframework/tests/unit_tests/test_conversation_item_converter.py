# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import pytest

from agent_framework import Content

from openai.types.conversations.message import Message
from openai.types.responses.response_input_text import ResponseInputText
from openai.types.responses.response_function_tool_call_item import ResponseFunctionToolCallItem
from openai.types.responses.response_function_tool_call_output_item import ResponseFunctionToolCallOutputItem
from openai.types.responses import response_reasoning_item

from azure.ai.agentserver.agentframework.models.conversation_converters import ConversationItemConverter


@pytest.fixture()
def converter() -> ConversationItemConverter:
    return ConversationItemConverter()


@pytest.mark.unit
def test_to_chat_message_converts_basic_message(converter: ConversationItemConverter) -> None:
    item = Message(
        id="msg_1",
        role="user",
        status="completed",
        type="message",
        content=[ResponseInputText(text="Hello world", type="input_text")],
    )

    result = converter.to_chat_message(item)

    assert result is not None
    assert result.role == "user"
    assert result.text is not None and "Hello world" in result.text
    assert result.contents is not None
    assert any(getattr(content, 'type', None) == 'text' for content in result.contents)


@pytest.mark.unit
def test_to_chat_message_converts_function_call_item(converter: ConversationItemConverter) -> None:
    item = ResponseFunctionToolCallItem(
        id="call_item_1",
        type="function_call",
        status="completed",
        call_id="call_123",
        name="do_something",
        arguments='{"foo": "bar"}',
    )

    result = converter.to_chat_message(item)

    assert result is not None
    assert result.role == "assistant"
    assert result.contents is not None
    assert len(result.contents) == 1
    content = result.contents[0]
    assert isinstance(content, Content) and content.type == "function_call"
    assert content.call_id == "call_123"
    assert content.name == "do_something"
    assert isinstance(content.arguments, dict)
    assert content.arguments.get("foo") == "bar"


@pytest.mark.unit
def test_to_chat_message_converts_function_result_item(converter: ConversationItemConverter) -> None:
    item = ResponseFunctionToolCallOutputItem(
        id="call_output_1",
        type="function_call_output",
        call_id="call_456",
        status="completed",
        output='{"answer": 42}',
    )

    result = converter.to_chat_message(item)

    assert result is not None
    assert result.role == "tool"
    assert result.contents is not None
    assert len(result.contents) == 1
    content = result.contents[0]
    assert isinstance(content, Content) and content.type == "function_result"
    assert content.call_id == "call_456"
    assert content.result == {"answer": 42}


@pytest.mark.unit
def test_to_chat_message_converts_reasoning_item(converter: ConversationItemConverter) -> None:
    reasoning_item = response_reasoning_item.ResponseReasoningItem(
        id="reasoning_1",
        type="reasoning",
        status="completed",
        summary=[response_reasoning_item.Summary(text="High-level summary", type="summary_text")],
        content=[response_reasoning_item.Content(text="Chain-of-thought", type="reasoning_text")],
    )

    result = converter.to_chat_message(reasoning_item)

    assert result is not None
    assert result.role == "assistant"
    assert result.text == "High-level summary"
    assert result.contents is not None
    assert any(getattr(content, 'type', None) == 'text_reasoning' for content in result.contents)
    text_reasoning = next(content for content in result.contents if getattr(content, 'type', None) == 'text_reasoning')
    assert text_reasoning.text == "Chain-of-thought"


@pytest.mark.unit
def test_to_chat_message_returns_none_for_unsupported_items(converter: ConversationItemConverter) -> None:
    class UnsupportedItem:
        type = "unsupported"

    assert converter.to_chat_message(UnsupportedItem()) is None
