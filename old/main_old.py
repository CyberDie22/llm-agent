import base64
import io
import json
import pathlib
import re
import time
import uuid
from typing import Generator, Any

import httpx
import openai
import replicate
from PIL.Image import Image
from openai.types.chat.chat_completion_chunk import Choice
from PIL import Image

from html_to_markdown import convert_to_markdown

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

def bfl_generate_image(model: str, prompt: str, width: int, height: int, safety_tolerance: int, output_format: str = "png", **kwargs) -> tuple[bytes, str] | None:
    response = bfl_client.post(
        f"https://api.bfl.ml/v1/{model}",
        headers={
            'Content-Type': 'application/json'
        },
        json={
            "prompt": prompt,
            "width": width,
            "height": height,
            "safety_tolerance": safety_tolerance,
            "output_format": output_format,
            **kwargs,
        }
    )
    id = response.json()['id']

    time.sleep(0.1)
    while True:
        response = bfl_client.get(f"https://api.bfl.ml/v1/get_result?id={id}")
        response_json = response.json()
        print(response_json)
        if response_json['status'] == 'Ready':
            url = response_json['result']['sample']
            image_response = httpx.get(url)
            image_bytes = image_response.content

            file_name = f"image-cache/bfl-{model}-{id}.png"
            pathlib.Path(file_name).write_bytes(image_bytes)

            return image_bytes, file_name
        elif "Moderated" in response_json['status']:
            return None  # TODO: handle moderation
        elif response_json['status'] == 'Error':
            return None  # TODO: handle error
        time.sleep(0.5)

def stream_openrouter_response(model: str, messages: list[dict], **kwargs) -> Generator[Choice, Any, None]:
    response = openrouter_client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
        stream=True,
    )

    for chunk in response:
        yield chunk.choices[0]

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

def print_message(message: dict) -> None:
    if message['role']:
        print(f"{message['role'].upper()}> ", end='')
    print(message['content'])


def get_user_message() -> dict:
    user_input = input("USER> ")
    return {
        "role": "user",
        "content": user_input,
    }


task_tools = [
    {
        "type": "function",
        "function": {
            "name": "query_user",
            "description": "Queries the user for information. You must provide a prompt to the user before calling this function.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "creative_task",
            "description": "Switches to a creative task mode to complete a creative task such as writing a story or article. The user cannot see anything that this tool responds with.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The creative task to complete."
                    },
                },
                "required": ["task"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "internet_search",
            "description": "Takes a query and returns search results from the internet. The user cannot see the output of this tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for."
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
            "description": "Takes a URL and returns the content of the web page. The user cannot see the output of this tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the web page."
                    },
                    "summary_query": {
                        "type": "string",
                        "description": "A query to focus the content of the summarization on."
                    }
                },
                "required": ["url", "summary_query"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generates an image-cache based on a prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt to generate an image-cache from. The prompt should be very descriptive and must describe anything that should be in the scene. Be sure to describe the if the background should be blurred, the colors of the objects, and any other details that should be in the image-cache."
                    },
                    "width": {
                        "type": "integer",
                        "description": "The width of the generated image-cache in pixels. Must be between 256 and 1440, and be a multiple of 32."
                    },
                    "height": {
                        "type": "integer",
                        "description": "The height of the generated image-cache in pixels. Must be between 256 and 1440, and be a multiple of 32."
                    },
                },
                "required": ["prompt", "width", "height"],
                "additionalProperties": False
            }
        }
    }
]


moderation_flagged_conversation = False


def moderate_message(message: dict) -> dict:
    openai_moderation_response = openai_client.moderations.create(
        input=message["content"],
        model="omni-moderation-2024-09-26"
    )

    moderation_results = openai_moderation_response.results[0].categories.model_dump(mode='json')

    claude_moderation_response = openrouter_response(
        model='anthropic/claude-3-haiku:beta',
        messages=[
            {"role": "system", "content": "Moderate the message provided by the user. If the message has any NSFW content, flag it as NSFW by responding with <NSFW>. If the message is safe for work, respond with <SFW>. Do not respond with anything else."},
            {"role": "user", "content": message['content']}
        ],
    )
    claude_nsfw = "NSFW" in claude_moderation_response['content']
    moderation_results['claude_nsfw'] = claude_nsfw

    moderation_results['moderation_flagged'] = openai_moderation_response.results[0].flagged or claude_nsfw

    return moderation_results

def plan_task(message: str) -> tuple[str, str]:
    global task_tools
    task_response = print_openrouter_response(
        model='meta-llama/llama-3.2-3b-instruct',  # should fine-tune something like Qwen2.5-0.5b on this task
        messages=[
            {"role": "system", "content": "Convert the message provided by the user into a task to complete. Keep the task simple and similar to what the user provided. Do not respond with anything other than the task to complete."},
            {"role": "user", "content": message}
        ],
        print_role='task'
    )

    task_string = ""
    for tool in task_tools:
        task_string += f"- {tool['function']['name']} ({tool['function']['description']})\n"
    task_string = task_string.strip()

    task_plan_response = print_openrouter_response(
        model='openai/o1-mini',
        messages=[
            {"role": "system", "content": "Create a step by step plan to complete the task provided by the user. Do not respond with anything other than the plan to complete the task."},
            {"role": "user", "content": f"Task: {task_response['content']}\nTools:\n{task_string}"},
        ],
        print_role='task_plan'
    )

    return task_response['content'], task_plan_response['content']

def image_split(image: bytes, chunk_size=(4, 4)):
    image = Image.open(io.BytesIO(image))
    image_width, image_height = image.size
    chunk_width = image_width // chunk_size[0]
    chunk_height = image_height // chunk_size[1]

    chunks = []
    for i in range(chunk_size[0]):
        for j in range(chunk_size[1]):
            left = i * chunk_width
            upper = j * chunk_height
            right = (i + 1) * chunk_width
            lower = (j + 1) * chunk_height
            chunk = image.crop((left, upper, right, lower))
            image_bytes = io.BytesIO()
            chunk.save(image_bytes, "png")
            chunks.append(image_bytes.getvalue())

    return chunks

def fix_image(chunk: bytes, issue: str) -> bytes:
    image = Image.open(io.BytesIO(chunk))
    image_width, image_height = image.size

    # increase small edge to match the larger edge so the image-cache is square
    # fill in the new space with black
    if image_width < image_height:
        new_image = Image.new("RGB", (image_height, image_height), (0, 0, 0))
        new_image.paste(image, ((image_height - image_width) // 2, 0))
    elif image_height < image_width:
        new_image = Image.new("RGB", (image_width, image_width), (0, 0, 0))
        new_image.paste(image, (0, (image_width - image_height) // 2))
    else:
        new_image = image

    # convert issue to a fix
    fix_response = print_openrouter_response(
        model='google/gemini-flash-1.5-8b',
        messages=[
            {"role": "system", "content": "Convert the issue with an image-cache chunk into a fix for the issue. Respond with the text for the fix. The fix should be a simple modification to the image-cache such as `make the pot in the bottom left white` or `remove the jar in the top left`. Do not respond with anything else."},
            {"role": "user", "content": issue},
        ],
        print_role='image_fix'
    )
    fix_text = fix_response['content']

    # fix the image-cache using apple ml-mgie

    new_image_bytes_io = io.BytesIO()
    new_image.save(new_image_bytes_io, "png")

    fixed_image = replicate_client.run(
        "camenduru/ml-mgie:cd6688b06dcdcf8b6c614abe400d37d40d85b9e07e438396582a1721686667b7",
        {
            "input_image": new_image_bytes_io,
        }
    )
    fixed_image_output = fixed_image['path']

    fixed_image_bytes = fixed_image_output.read()

    # crop the fixed image-cache to the original size, removing the black bars
    fixed_image = Image.open(io.BytesIO(fixed_image_bytes))
    left = (fixed_image.width - image_width) // 2
    top = (fixed_image.height - image_height) // 2
    right = left + image_width
    bottom = top + image_height
    fixed_image_cropped = fixed_image.crop((left, top, right, bottom))
    fixed_image_cropped_bytes_io = io.BytesIO()
    fixed_image_cropped.save(fixed_image_cropped_bytes_io, "png")
    fixed_image_bytes = fixed_image_cropped_bytes_io.getvalue()

    pathlib.Path(f"image-cache/fix-{uuid.uuid4()}.png").write_bytes(fixed_image_bytes)
    return fixed_image_bytes

def check_image(image: bytes, prompt: str) -> bytes:
    # split image-cache into 4x4 chunks
    chunk_count = 4
    chunks = image_split(image, (chunk_count, chunk_count))
    pil_image = Image.open(io.BytesIO(image))

    # TODO: check each chunk for NSFW content

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Image Prompt: {prompt}"},
                {"type": "text", "text": "Image:"},
                {"type": "image_url", "image_url": {"url": "data:image-cache/png;base64," + base64.b64encode(image).decode('utf-8')}},
            ]
        },
    ]

    for idx, chunk in enumerate(chunks):
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"Chunk {idx + 1}:"},
                {"type": "image_url", "image_url": {"url": "data:image-cache/png;base64," + base64.b64encode(chunk).decode('utf-8')}},
            ]
        })

    print("getting image-cache checked")
    response = print_openrouter_response(
        model='anthropic/claude-3.5-sonnet:beta',
        messages=[
            {"role": "system", "content": "Check the image-cache provided by the user for correctness. Respond with any issues found in the image-cache by chunk. Describe any issues for each chunk. Respond with the issues in the format {\"chunk_number (e.g. 3)\": [\"detailed issue description here\"]}, where the chunk number is the number of the chunk with the issue and the detailed issue description describes the issue in detail. If there are no issues with the image-cache, respond with an empty object. Respond with the issues in <issues> tags. Think about the images before responding in <thinking> tags. Think about each chunk one by one and describe any issues you find in detail."},
            *messages
        ],
    )
    response_text = response['content']

    new_image = image
    if "<issues>" in response_text:
        issues = json.loads(response_text.split("<issues>")[1].split("</issues>")[0])
        for issue_list in issues.values():
            for issue in issue_list:
                new_image = fix_image(new_image, issue)

    fixed_image = Image.new("RGB", pil_image.size)
    for i in range(chunk_count):
        for j in range(chunk_count):
            left = i * pil_image.width // chunk_count
            upper = j * pil_image.height // chunk_count
            right = (i + 1) * pil_image.width // chunk_count
            lower = (j + 1) * pil_image.height // chunk_count
            chunk = Image.open(io.BytesIO(chunks[i * chunk_count + j]))
            fixed_image.paste(chunk, (left, upper, right, lower))

    fixed_image_bytes_io = io.BytesIO()
    fixed_image.save(fixed_image_bytes_io, "png")
    fixed_image_bytes = fixed_image_bytes_io.getvalue()

    return fixed_image_bytes

def generate_image(prompt: str, width: int, height: int) -> tuple[bytes, str]:
    image, file_path = bfl_generate_image(
        model='flux-pro-1.1',
        prompt=prompt,
        width=width,
        height=height,
        safety_tolerance=6,
    )

    fixed_image = check_image(image, prompt)

    return fixed_image, file_path

def execute_creative_task(messages: list[dict], user_message: dict, task: str, task_plan: str) -> list[dict]:
    print("Executing Creative Task")
    global moderation_flagged_conversation
    task_user_message = {
        "role": "user",
        "content": f"User Message: {user_message['content']}\n\nTask: {task}\nTask Plan:\n{task_plan}"
    }

    creative_model = 'anthracite-org/magnum-v4-72b' if moderation_flagged_conversation else 'anthropic/claude-3-opus:beta'

    creative_response = print_openrouter_response(
        model=creative_model,
        messages=[
            {"role": "system", "content": "Write a creative response to the user's task. Respond with the results in <response> tags. Think about what to do before you respond in <thinking> tags. The user cannot see anything outside of the <response> tags."},
            *messages,
            task_user_message
        ],
        temperature=1.2,
    )

    creative_text = creative_response['content'].split("<response>")[1].split("</response>")[0].strip()

    return messages + [
        user_message,
        {"role": "assistant", "content": creative_text}
    ]

def execute_task(messages: list[dict], user_message: dict, task: str, task_plan: str) -> list[dict]:
    print("Executing Regular Task")
    global task_tools
    global moderation_flagged_conversation
    task_user_message = {
        "role": "user",
        "content": f"User Message: {user_message['content']}\n\nTask: {task}\nTask Plan:\n{task_plan}"
    }

    post_messages = []

    while True:
        time.sleep(3)
        agent_model = 'qwen/qwen-2.5-72b-instruct' if moderation_flagged_conversation else 'anthropic/claude-3.5-sonnet:beta'

        response = print_openrouter_response(
            model=agent_model,
            messages=[
                {"role": "system", "content": "Execute the task plan provided by the user. Respond with the results in <response> tags. Think about what to do before you respond in <thinking> tags. Be sure the close all tags you open. Spend as much time as you need researching topics, opening web pages, and searching the internet to complete the task. The user cannot see anything outside the <response> tags."},
                *messages,
                task_user_message,
                *post_messages
            ],
            tools=task_tools,
        )

        post_messages += [
            response
        ]

        if response.get('tool_calls') is not None:
            for tool_call in response['tool_calls']:
                if tool_call['type'] == 'function':
                    match tool_call['function']['name']:
                        case 'creative_task':
                            creative_task = tool_call['function']['arguments']
                            task, task_plan = plan_task(creative_task)
                            creative_messages = execute_creative_task(messages, user_message, task, task_plan)
                            creative_response = creative_messages[-1]['content']
                            post_messages += [{
                                'role': 'tool',
                                'content': f"Creative Task: {creative_task}\n\nCreative Response: {creative_response}",
                                'tool_call_id': tool_call['id']
                            }]
                        case 'query_user':
                            user_message = get_user_message()
                            moderation = moderate_message(user_message)
                            if moderation['moderation_flagged']:
                                moderation_flagged_conversation = True

                            post_messages += [{
                                "role": "tool",
                                "content": f"User Response: {user_message['content']}",
                                "tool_call_id": tool_call['id']
                            }]
                        case 'internet_search':
                            args = json.loads(tool_call['function']['arguments'])
                            query = args['query']

                            print(f"INTERNET_SEARCH> Searching for '{query}'...")
                            search_results = websearch(query)
                            # TODO: rerank with cohere or another embedding model

                            search_results_text = ""
                            for index, result in enumerate(search_results):
                                search_results_text += f"{index + 1}. [{result['title']}]({result['url']}) - {result['snippet']}\n"

                            # TODO: Run Meta LLaMA PromptGuard to protect against malicious LLM injections in snippets

                            post_messages += [{
                                "role": "tool",
                                "content": f"Search Results for '{query}':\n\n{search_results_text}",
                                "tool_call_id": tool_call['id']
                            }]
                        case 'web_page':
                            args = json.loads(tool_call['function']['arguments'])
                            url = args['url']
                            summary_query = args['summary_query']

                            print(f"WEB_PAGE> Fetching web page content for '{url}' and summarizing based on '{summary_query}'...")
                            response = httpx.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0'})
                            web_page_content = convert_to_markdown(response.text)

                            # TODO: run Meta LLaMA PromptGuard to protect against malicious LLM injections
                            # TODO: maybe OpenAI moderations for web page content?

                            webpage_summarization = print_openrouter_response(
                                model='qwen/qwen-2-7b-instruct',
                                messages=[
                                    {"role": "system", "content": f"Respond with a long, very detailed, and very in depth summary of the web page provided by the user, using the query '{summary_query}' to focus on the information important for the query. The summary can be as long as you need to include everything. Do not respond with anything other than the summary."},
                                    {"role": "user", "content": web_page_content}
                                ],
                                print_role='web_page_summary',
                            )

                            post_messages += [{
                                "role": "tool",
                                "content": f"Web Page Content for '{url}':\n\n{webpage_summarization['content']}",
                                "tool_call_id": tool_call['id']
                            }]
                        case 'generate_image':
                            args = json.loads(tool_call['function']['arguments'])
                            prompt = args['prompt']
                            width = args['width']
                            height = args['height']

                            width = min(max(width, 256), 1440)
                            height = min(max(height, 256), 1440)

                            width = (width // 32) * 32
                            height = (height // 32) * 32

                            print(f"GENERATE_IMAGE> Generating image-cache based on prompt: '{prompt}'...")
                            image, file_path = generate_image(prompt, width, height)

                            post_messages += [
                                {
                                    "role": "tool",
                                    "content": f"Generated Image based on prompt '{prompt}'",
                                    "tool_call_id": tool_call['id'],
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": f"Generated Image ({file_path}):"},
                                        {"type": "image_url", "image_url": {"url": "data:image-cache/png;base64," + base64.b64encode(image).decode('utf-8')}},
                                    ]
                                }
                            ]

        else:
            break

    full_response = [message['content'] for message in post_messages]
    full_response = "\n".join(full_response)

    # find all the content in <response> tags
    if full_response.count("<response>") > full_response.count("</response>"):
        full_response += "</response>"
    responses = re.findall(r'<response>(.*?)</response>', full_response, re.DOTALL)
    complete_response = "\n".join(responses)

    print_message({
        "role": "full_response",
        "content": complete_response
    })

    # pi_response = print_openrouter_response(
    #     model="inflection/inflection-3-pi",
    #     messages=[
    #         {"role": "system", "content": "Rewrite the old assistant response provided by the user to be better and follow the style of the rest of the conversation. Do not mention that you are rewriting it or that this wasn't the original response. Do not use any information outside of the text in the old assistant response."},
    #         user_message,
    #         {"role": "user", "content": f"Old Assistant Response:\n{complete_response}"},
    #     ],
    #     print_role="rewrite_response_pi"
    # )

    assistant_response = {
        "role": "assistant",
        "content": complete_response
    }

    messages += [
        user_message,
        assistant_response
    ]

    return messages

def extract_files_from_message(message: str) -> list[dict]:
    matches = re.split(r'(<.*?) file="(.*?)">', message)

    message = []

    looking_file = False
    for match in matches:
        if looking_file:
            file_path = match
            looking_file = False
            file_extension = pathlib.Path(file_path).suffix
            file_content = pathlib.Path(file_path).read_text('utf-8')
            message += [{
                "type": "text",
                "text": f"\n\nFile: {file_path}\n\n```{file_extension.removeprefix('.')}\n{file_content}\n```\n\n"
            }]
        elif match == "<file":
            looking_file = True
            continue
        elif match == "":
            continue
        else:
            message += [{
                "type": "text",
                "text": match
            }]

    return message

def main() -> int:
    global moderation_flagged_conversation
    messages = []

    while True:
        user_message = get_user_message()
        new_message_content = extract_files_from_message(user_message['content'])
        user_message['content'] = new_message_content
        moderation = moderate_message(user_message)
        print(moderation)

        # if moderation['moderation_flagged']:
        #     moderation_flagged_conversation = True

        task, task_plan = plan_task(user_message['content'])

        # classify_task = openrouter_response(
        #     model='google/gemini-flash-1.5-8b',
        #     messages=[
        #         {"role": "system", "content": "Classify the task provided by the user. Respond with <creative> if the task is creative and requires a creative response, such as writing a story or article. Respond with <non_creative> if the task is not creative and requires a non-creative response, such as researching a topic or other non-creative tasks. Do not respond with anything else."},
        #         {"role": "user", "content": task}
        #     ]
        # )
        # is_creative_task = "<creative>" in classify_task['content']

        # if is_creative_task:
        #     print("Starting Creative Task (messages ", messages, ")")
        #     messages = execute_creative_task(messages, user_message, task, task_plan)
        #     print("Finished Creative Task (messages ", messages, ")")
        # else:
        print("Starting Regular Task (messages ", messages, ")")
        messages = execute_task(messages, user_message, task, task_plan)
        print("Finished Regular Task (messages ", messages, ")")

        if input("Continue? (Y/N) ") != "Y":
            break

    print("\n\n\n\n\n\n")

    print(json.dumps(messages, indent=4))

    return 0

if __name__ == "__main__":
    main()
