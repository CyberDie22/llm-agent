import base64
import json
import pathlib
import re
import sys
import time
import hashlib
from typing import Generator, Any

import httpx
import openai
import tiktoken
import replicate
from openai.types.chat.chat_completion_chunk import Choice
import undetected_chromedriver as uchrome

from websearch import websearch


openrouter_client = openai.Client(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)


openai_client = openai.Client(
    api_key=openai_api_key,
)


bfl_client = httpx.Client(
    headers={'X-Key': bfl_api_key}
)


replicate_client = replicate.Client(
    api_token=replicate_api_key
)


def generate_bfl_image(model: str, prompt: str, width: int, height: int, **kwargs) -> tuple[bytes | None, str]:
    # width and height must be between 256 and 1440, and be a multiple of 32
    width = max(256, min(1440, width))
    height = max(256, min(1440, height))
    width = width - (width % 32)
    height = height - (height % 32)

    response = bfl_client.post(f"https://api.bfl.ml/v1/{model}", json={
        "prompt": prompt,
        "width": width,
        "height": height,
        **kwargs,
        "output_format": "png"
    })
    response_id = response.json()['id']

    while True:
        time.sleep(0.5)

        response = bfl_client.get(f"https://api.bfl.ml/v1/get_result?id={response_id}")
        response_json = response.json()

        if response_json['status'] == 'Ready':
            url = response_json['result']['sample']
            download_response = bfl_client.get(url)
            image = download_response.content

            image_path = f"image-cache/bfl-{model}-{response_id}.png"
            pathlib.Path(image_path).write_bytes(image)

            return image, image_path
        elif "Moderated" in response_json['status']:
            return None, ""  # TODO: handle moderation
        elif response_json['status'] == 'Error':
            return None, ""  # TODO: handle error


def stream_openrouter_response(model: str, messages: list[dict], **kwargs) -> Generator[Choice, Any, None]:
    response = openrouter_client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
        stream=True,
    )

    try:
        for chunk in response:
            yield chunk.choices[0]
    except openai.APIError as e:
        print(f"Error: {e}")
        print("Retrying...")
        # retry after 0.5 seconds
        time.sleep(0.5)
        yield from stream_openrouter_response(model, messages, **kwargs)

def stream_openrouter_response_with_callback(model: str, messages: list[dict], callback, **kwargs) -> dict:
    response = stream_openrouter_response(model, messages, **kwargs)

    full_message = {
        'role': None,
        'content': '',
        'tool_calls': None,
        # 'finish_reason': None
    }

    for choice in response:
        callback(choice)
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
    return full_message

def openrouter_response(model: str, messages: list[dict], **kwargs) -> dict:
    def noop(*args, **kwargs): pass
    response = stream_openrouter_response_with_callback(model, messages, noop, **kwargs)
    return response

def print_openrouter_response(model: str, messages: list[dict], print_role: str | None = None, **kwargs) -> dict:
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

    response = stream_openrouter_response_with_callback(model, messages, print_message_choice, **kwargs)
    print()

    return response


def print_message(message: dict, print_role: str = None) -> None:
    if print_role:
        print(f"{print_role.upper()}> ", end='')
    elif message['role']:
        print(f"{message['role'].upper()}> ", end='')
    print(message['content'])

def get_user_message() -> dict:
    user_input = input("USER> ").replace("\\n", "\n")

    message = []
    matches = re.split(r"(<file) path=\"(.*?)\">", user_input)
    open_file = False
    for match in matches:
        if match == "<file":
            open_file = True
            continue
        elif open_file:
            path = match
            if pathlib.Path(path).exists():
                extension = pathlib.Path(path).suffix
                if extension == ".mp3":
                    file_bytes = pathlib.Path(path).read_bytes()
                    blake2b_hash = hashlib.blake2b(file_bytes).hexdigest()

                    cache_file_path = pathlib.Path(f"audio-cache/openai-api-whisper1-{blake2b_hash}.json")
                    if cache_file_path.exists():
                        response = json.loads(cache_file_path.read_text())
                    else:
                        response = openai_client.audio.transcriptions.create(
                            model="whisper-1",
                            response_format="verbose_json",
                            timestamp_granularities=["word"],
                            file=pathlib.Path(path),
                        )
                        response = response.model_dump(mode="json")
                        cache_file_path.write_text(json.dumps(response))

                    full_text = response['text']
                    words = response['words']

                    message.append({
                        "type": "text",
                        "text": f"Full Text: {full_text}\nWords: {words}"
                    })

                    # message.append({
                    #     "type": "input_audio",
                    #     "input_audio": {
                    #         "data": file_content,
                    #         "format": "mp3"
                    #     }
                    # })
                else:
                    file_text = pathlib.Path(path).read_text()
                    message.append({
                        "type": "text",
                        "text": file_text
                    })
            open_file = False
        else:
            message.append({
                "type": "text",
                "text": match
            })

    return {
        "role": "user",
        "content": message,
    }


tool_list = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet for web pages related to the provided query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_page",
            "description": "Get the content of a webpage-cache.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the webpage-cache."
                    },
                    "query": {
                        "type": "string",
                        "description": "A query to refine the page summary."
                    }
                },
                "required": ["url", "query"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image-cache based on the provided prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt to generate the image-cache from. Be extremely descriptive, have a paragraph of detail for each item in the image-cache. Include everything you want in the image-cache. Include detail about focus, lighting, and perspective."
                    },
                    "width": {
                        "type": "integer",
                        "description": "The width of the image-cache in pixels. Must be between 256 and 1440 and a multiple of 32."
                    },
                    "height": {
                        "type": "integer",
                        "description": "The height of the image-cache in pixels. Must be between 256 and 1440 and a multiple of 32."
                    }
                },
                "required": ["prompt"],
                "additionalProperties": False
            }
        }
    }
]
basic_tool_list = "\n".join([f"{idx + 1}. {tool['function']['name']} - {tool['function']['description']}" for idx, tool in enumerate(tool_list)])
full_tool_list = "\n".join([f"{idx+1}. {tool['function']['name']} - {tool['function']['description']}\nParameters: {tool['function']['parameters']}" for idx, tool in enumerate(tool_list)])


def get_tasks(user_message: dict, previous_tasks: list[str]) -> list[str]:
    response = print_openrouter_response(
        model="anthropic/claude-3.5-sonnet:beta",
        messages=[
            {"role": "system", "content": f"Analyze the users message and create a list of simple tasks to complete. Provide a detailed description for the simple task. Provide the tasks as json. Do not add any information, only modify what the user provided into simple tasks. Be sure that the tasks you provide are exactly what the user has asked for.\n\nThese are the tools available:\n{basic_tool_list}\n\nRespond with the list of tasks in the following format: "},
            {"role": "user", "content": f"Previously completed tasks: {', '.join(previous_tasks)}"},
            user_message,
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "tasks",
                "description": "A list of tasks to complete.",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "tasks": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": ["tasks"],
                    "additionalProperties": False
                }
            }
        },
        print_role="tasks"
    )

    json_response = json.loads(response['content'])
    return json_response['tasks']

def complete_task(task: str, user_message: dict, messages: list[dict]) -> dict:
    task_plan = print_openrouter_response(
        model="openai/o1-mini",
        messages=[
            {"role": "system", "content": f"Create a plan to complete the task provided by the user using the available tools.\n\nThese are the tools available:\n{full_tool_list}"},
            *messages,
            {"role": "user", "content": f"Task to complete: {task}"},
        ],
        print_role="task_plan"
    )

    post_messages = []

    while True:
        response = print_openrouter_response(
            model="anthropic/claude-3.5-sonnet:beta",
            messages=[
                {"role": "system", "content": f"Complete the task provided by the user, following the provided plan. Use as many tools as necessary. Think before doing anything in <thinking> tags. Provide your final response in <response> tags."},
                *messages,
                {"role": "user", "content": f"User Message: {user_message['content']}\nTask: {task}\nTask Plan: {task_plan['content']}"},
                *post_messages,
            ],
            tools=tool_list,
            print_role="task_execution"
        )

        post_messages.append(response)

        if response.get('tool_calls'):
            for tool_call in response['tool_calls']:
                if tool_call['type'] == 'function':
                    match tool_call['function']['name']:
                        case "web_search":
                            args = json.loads(tool_call['function']['arguments'])
                            query = args['query']

                            search_results = websearch(query)

                            # TODO: Rerank search results based on user message
                            #       Include other engines such as Bing, Brave, Marginailia
                            #       Include other search types such as image-cache, video
                            #       Filter search results with PromptGuard for injections

                            response_text = "Search Results:\n\n"
                            for idx, result in enumerate(search_results):
                                response_text += f"{idx + 1}. [{result['title']}]({result['url']}) - {result['snippet']}\n"

                            print("SEARCH_RESULTS> " + ", ".join([result['title'] for result in search_results]))

                            post_messages.append({
                                "role": "tool",
                                "content": response_text,
                                "tool_call_id": tool_call['id']
                            })
                        case "web_page":
                            args = json.loads(tool_call['function']['arguments'])
                            url = args['url']
                            query = args['query']

                            print(f"WEB_PAGE> Getting content of webpage-cache `{url}`...")
                            response = httpx.get(
                                url,
                                headers={
                                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
                                }
                            )
                            response_text = response.text

                            # with uchrome.Chrome(headless=False) as driver:
                            #     driver.implicitly_wait(3)
                            #     driver.get(url)
                            #     response_text = driver.page_source
                            #     driver.close()

                            pathlib.Path(
                                f"webpage-cache/webpage-cache-{url.replace('/', '_')}.html").write_text(response_text)

                            encoding = tiktoken.encoding_for_model("gpt-4o")
                            tokens = encoding.encode(response_text)
                            if len(tokens) > 32_000:
                                summary_model = "google/gemini-flash-1.5-8b"
                            else:
                                summary_model = "qwen/qwen-2.5-7b-instruct"

                            print(f"WEB_PAGE> Summarizing content of webpage-cache with model `{summary_model}`...")
                            summary_response = openrouter_response(
                                model=summary_model,
                                messages=[
                                    {"role": "system", "content": f"Summarize the content of the webpage-cache provided by the user. The user will provide a query to focus the summary on. Make the summary descriptive and in depth and include any information related to the query the user provides. Only respond with the summary, do not respond with anything else."},
                                    *messages,
                                    {"role": "user", "content": f"Query: {query}\n\nContent:\n{response_text}"},
                                ],
                            )

                            pathlib.Path(f"webpage-cache/webpage-cache-{url.replace('/', '_')}-summary.html").write_text(summary_response['content'])

                            post_messages.append({
                                "role": "tool",
                                "content": summary_response['content'],
                                "tool_call_id": tool_call['id']
                            })
                        case "generate_image":
                            args = json.loads(tool_call['function']['arguments'])
                            prompt = args['prompt']
                            width = args['width']
                            height = args['height']

                            print(f"IMAGE_GENERATION> Generating image-cache with size ({width}px, {height}px) and prompt `{prompt}`...")

                            image, image_path = generate_bfl_image(
                                model="flux-pro-1.1",
                                prompt=prompt,
                                width=width,
                                height=height,
                                prompt_upsampling=False,
                                safety_tolerance=6,
                            )

                            image_b64 = base64.b64encode(image).decode('utf-8')

                            if image:
                                print(f"IMAGE_GENERATION> Image generated and saved to {image_path}")
                                post_messages += [
                                    {
                                        "role": "tool",
                                        "content": f"![image-cache]({image_path})",
                                        "tool_call_id": tool_call['id']
                                    },
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": f"Generated Image ({image_path}):"},
                                            {"type": "image_url", "image_url": {"url": "data:image-cache/png;base64," + image_b64, "detail": "high"}}
                                        ]
                                    }
                                ]
                            else:
                                print("IMAGE_GENERATION> Image generation failed.")
                                post_messages.append({
                                    "role": "tool",
                                    "content": "Image generation failed.",
                                    "tool_call_id": tool_call['id']
                                })
        else:
            break

    full_response = "\n".join([message['content'] for message in post_messages if message['role'] == 'assistant'])

    # grab just the content in the last <response> tag
    response_content = full_response.rsplit("<response>", 1)[-1].split("</response>")[0]

    return {
        "role": "assistant",
        "content": response_content
    }

def main() -> int:
    messages = []
    previous_tasks = []

    while True:
        user_message = get_user_message()

        tasks = get_tasks(user_message, previous_tasks)
        previous_tasks.extend(tasks)

        task_messages = []

        for task in tasks:
            response = complete_task(task, user_message, [*messages, *task_messages])
            task_messages += [
                {"role": "user", "content": f"{task}"},
                response
            ]
            # TODO: gauge how well the task was completed

        messages += [
            user_message,
            task_messages[-1]
        ]

        if input("Continue? (Y/N) ") != "Y":
            break

    print(json.dumps(messages, indent=4))

    return 0

if __name__ == "__main__":
    sys.exit(main())
