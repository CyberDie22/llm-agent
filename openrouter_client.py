import time
from typing import Generator, Any

import openai
from openai.types.chat.chat_completion_chunk import Choice

from messagetypes import Message, UserMessage, AssistantMessage, ToolCall

openrouter_client = openai.Client(
    # base_url="https://openrouter.helicone.ai/api/v1",
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
    # default_headers={
    # }
)

def stream_openrouter_response(model: str, messages: list[dict], **kwargs) -> Generator[Choice, Any, None]:
    tries = 0
    delay = 0.5
    while True:
        response = openrouter_client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
            stream=True,
        )

        try:
            for chunk in response:
                yield chunk.choices[0]
            else:
                break
        except openai.APIError as e:
            print(f"Error: {e} - {e.body}")
            print(messages)
            print("Retrying...")
            tries += 1
            if tries > 5:
                raise e
            time.sleep(delay)
            delay *= 2

def stream_openrouter_response_and_combine(model: str, messages: list[dict], **kwargs) -> Generator[dict, Any, None]:
    response = stream_openrouter_response(model, messages, **kwargs)

    full_message = {
        'role': None,
        'content': '',
        'tool_calls': None,
        # 'finish_reason': None
    }

    for choice in response:
        yield choice
        if choice.delta:
            if choice.delta.role and not full_message['role']:
                full_message['role'] = choice.delta.role

            if choice.delta.content:
                full_message['content'] += choice.delta.content

            if choice.delta.tool_calls:
                if not full_message['tool_calls']:
                    full_message['tool_calls'] = {}
                for tool_call in choice.delta.tool_calls:
                    if full_message['tool_calls'].get(tool_call.index) is None:
                        full_message['tool_calls'][tool_call.index] = {}
                    current_tool_call = full_message['tool_calls'][tool_call.index]

                    if tool_call.type and not current_tool_call.get('type'):
                        current_tool_call['type'] = tool_call.type

                    if tool_call.id and not current_tool_call.get('id'):
                        current_tool_call['id'] = tool_call.id

                    if current_tool_call.get('type') == 'function':
                        if tool_call.function:
                            if current_tool_call.get('function') is None:
                                current_tool_call['function'] = {}
                            current_function = current_tool_call['function']

                            if tool_call.function.name and not current_function.get('name'):
                                current_function['name'] = tool_call.function.name

                            if tool_call.function.arguments:
                                if current_function.get('arguments') is None:
                                    current_function['arguments'] = ''
                                current_function['arguments'] += tool_call.function.arguments

                            current_tool_call['function'] = current_function

                    # if tool_call.finish_reason and not full_message['finish_reason']:
                    #     full_message['finish_reason'] = tool_call.finish_reason
                    #
                    # full_message['tool_calls'][tool_call.index] = current_tool_call

    if full_message['tool_calls'] is not None:
        full_message['tool_calls'] = list(full_message['tool_calls'].values())
    else:
        del full_message['tool_calls']

    yield {
        'complete': True,
        'message': full_message
    }

def stream_openrouter_response_with_chunk_callback(model: str, messages: list[dict], callback, **kwargs) -> AssistantMessage:
    response = stream_openrouter_response_and_combine(model, messages, **kwargs)
    for chunk in response:
        if chunk.get('complete'):
            return chunk['message']
        callback(chunk)

def stream_openrouter_response_with_message_callback(model: str, messages: list[dict], callback, **kwargs) -> Generator[Choice, Any, None]:
    response = stream_openrouter_response_and_combine(model, messages, **kwargs)
    for chunk in response:
        if isinstance(chunk, dict) and chunk.get('complete'):
            callback(chunk['message'])
            continue
        yield chunk

def openrouter_response(model: str, messages: list[dict], **kwargs) -> AssistantMessage:
    def noop(*args): pass
    response = stream_openrouter_response_with_chunk_callback(model, messages, noop, **kwargs)
    return response

def print_openrouter_response(model: str, messages: list[dict], print_role: str | None = None, **kwargs) -> AssistantMessage:
    print_info = {
        'has_printed_role': False,
        'print_role': print_role
    }

    def print_message_choice(choice: Choice) -> None:
        if not print_info['has_printed_role'] and print_info['print_role']:
            print(f"{print_info['print_role'].upper()}> ", end='')
            print_info['has_printed_role'] = True

        if choice.delta:
            if not print_info['has_printed_role'] and choice.delta.role:
                print(f"{choice.delta.role.upper()}> ", end='')
                print_info['has_printed_role'] = True
            if choice.delta.content:
                print(choice.delta.content, end='')

    response = stream_openrouter_response_with_chunk_callback(model, messages, print_message_choice, **kwargs)
    print()

    return response
