import json

import openai

openrouter_api_key = "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"

openrouter_client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)

def combine_sequential_messages(messages: list[dict]) -> list[dict]:
    combined_messages = []
    current_message = {'role': '', 'content': ''}
    for message in messages:
        if current_message['role'] != message['role']:
            combined_messages.append(current_message)
            current_message = message
        else:
            current_message['content'] += message['content']
    if current_message['content']:
        combined_messages.append(current_message)
    return combined_messages

def stream_response(model: str, messages: list[dict], **kwargs):
    try:
        response = openrouter_client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
            stream=True,
        )

        full_message = {
            'role': None,
            'content': '',
            'tool_calls': None,
            # 'finish_reason': None
        }

        for choice in response:
            choice = choice.choices[0]
            yield choice.model_dump()
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
    except openai.APIError as e:
        print(f"Error: {e} - {e.body}")
        print(f"Request: {json.dumps(json.loads(e.request.content), indent=4)}")
        yield {'text': f"Error: {e} - {e.body}"}

def get_response(model: str, messages: list[dict], **kwargs):
    stream = stream_response(model, messages, **kwargs)
    for chunk in stream:
        if chunk.get('complete'):
            return chunk['message']
