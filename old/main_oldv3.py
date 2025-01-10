import base64
import io
import json
import pathlib
import re
import sys
import datetime
import time

import httpx

from openrouter_client import openrouter_response, print_openrouter_response
from websearch import websearch

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from markdownify import markdownify
import tiktoken
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

def date_string() -> str:
    # return 'Tuesday, August 31, 2021'
    return datetime.date.today().strftime('%A, %B %d, %Y')

def extract_xml_tag(text: str, tag: str) -> str:
    start_tag = f'<{tag}>'
    end_tag = f'</{tag}>'

    start_index = text.index(start_tag) + len(start_tag)
    end_index = text.index(end_tag)

    return text[start_index:end_index].strip()


def get_page_source(url, timeout=30):
    """
    Fetch the HTML source of a webpage-cache using undetected-chromedriver

    Args:
        url (str): The URL to scrape
        timeout (int): Maximum time to wait for page load in seconds

    Returns:
        str: HTML source of the page
    """
    try:
        # Initialize the driver
        options = uc.ChromeOptions()
        options.headless = True
        options.page_load_strategy = 'eager'

        driver = uc.Chrome(options=options, use_subprocess=False, no_sandbox=True)

        # Set page load timeout
        driver.set_page_load_timeout(timeout)

        # Navigate to the URL
        driver.get(url)

        # Wait for the page to load (wait for body tag)
        try:
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
        except TimeoutException:
            print(f"Timeout waiting for page load after {timeout} seconds")

        # Add a small delay to ensure dynamic content loads
        time.sleep(2)

        # Get the page source
        page_source = driver.page_source

        return page_source

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

    finally:
        try:
            driver.quit()
        except Exception as e:
            pass

def get_user_input() -> dict:
    user_input = input('USER> ')

    file_input_regex = re.compile(r'(<(\w+?) path="(.*?)">)')
    input_split = file_input_regex.split(user_input)

    message_parts = []

    file_tag = ""
    file_type = ""
    skip_tag = False
    for part in input_split:
        if skip_tag:
            message_parts += [{
                'type': 'text',
                'text': file_tag
            }]
            file_tag = ""
            file_type = ""
            skip_tag = False
            continue

        if file_type:
            match file_type:
                case 'img':
                    image_bytes = pathlib.Path(part).read_bytes()
                    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                    message_parts += [{
                        'type': 'image_url',
                        'image_url': {
                            'url': 'data:image-cache/png;base64,' + image_b64,
                            'detail': 'high'
                        }
                    }]
                case 'text':
                    file_text = pathlib.Path(part).read_text()
                    message_parts += [{
                        'type': 'text',
                        'text': file_text
                    }]

            file_tag = ""
            file_type = ""
            continue

        if file_tag:
            if part in ['img', 'text']:
                file_type = part
                continue
            else:
                skip_tag = True
                continue

        if file_input_regex.fullmatch(part):
            file_tag = part
            continue
        else:
            message_parts += [{
                'type': 'text',
                'text': part
            }]

    return {
        'role': 'user',
        'content': message_parts
    }

def detect_nsfw(user_message: dict) -> bool:
    response = openrouter_response(
        model='anthropic/claude-3-haiku:beta',
        messages=[
            {'role': 'system', 'content': 'Determine whether the user message is NSFW. Respond with just <nsfw> or <sfw>. Do not respond with anything else.'},
            user_message
        ]
    )

    response_text = response['content']
    return '<nsfw>' in response_text


def web_search(args: dict, tool_call_id: str, messages: list[dict]) -> list[dict]:
    query = args['query']

    print(f"WEB_SEARCH> Searching for: {query}")

    results = websearch(query)

    if not results:
        return [{
            'role': 'tool',
            'tool_call_id': tool_call_id,
            'content': 'No results found.'
        }]

    # TODO: Generate a new snippet with an LLM
    #       Use PromptGuard to guard against injections

    result_texts = []
    for i, result in enumerate(results[:3]):
        result_texts.append(f'{i + 1}. [{result["title"]}]({result["url"]})\n{result["snippet"]}')

    result_text = '\n\n'.join(result_texts)

    result_text += "\n\nRemember to open the web pages to find information instead of relying on the snippet provided. Only use the snippet to filter for relevant web pages."

    return [{
        'role': 'tool',
        'tool_call_id': tool_call_id,
        'content': result_text
    }]

def web_page_content(args: dict, tool_call_id: str, messages: list[dict]) -> list[dict]:
    url = args['url']
    query = args.get('query', '')

    print(f"WEB_PAGE> Loading page: {url}")

    page_source = get_page_source(url)

    # TODO: Use PromptGuard to guard against injections

    if not page_source:
        return [{
            'role': 'tool',
            'tool_call_id': tool_call_id,
            'content': 'Failed to load the page.'
        }]

    page_markdown = markdownify(page_source)

    image_regex = re.compile(r'!\[(.*?)\]\((.+?)(?: ".*?")\)')
    page_markdown_split = image_regex.split(page_markdown)

    message_parts = []
    alt_text = ""
    image = False
    skip = 0
    for part in page_markdown_split:
        if skip > 0:
            skip -= 1
            continue

        if alt_text:
            if part.endswith('.svg-b64') or 'image-cache/svg-b64+xml' in part:
                svg_path = part

                if svg_path.startswith('data:'):
                    svg_path = svg_path.split(',')[1]
                    svg_bytes = base64.b64decode(svg_path)
                    svg_file = io.BytesIO(svg_bytes)
                else:
                    res = httpx.get(svg_path)
                    svg_file = io.BytesIO(res.content)

                drawing = svg2rlg(svg_file)
                png_file = io.BytesIO()
                renderPM.drawToFile(drawing, png_file, fmt="PNG")

                png_b64 = base64.b64encode(png_file.getvalue()).decode('utf-8')

                url = 'data:image-cache/png;base64,' + png_b64
            else:
                url = part

            message_parts += [
                {
                    'type': 'text',
                    'text': f'![{alt_text}]('
                },
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': url,
                        'detail': 'high'
                    }
                },
                {
                    'type': 'text',
                    'text': ')'
                }
            ]
            alt_text = ""
            image = False
            continue

        if image:
            alt_text = part

        if image_regex.fullmatch(part):
            image = True
            continue
        else:
            alt_text = ""
            message_parts.append({
                'type': 'text',
                'text': part
            })
            skip = 2

    encoding = tiktoken.encoding_for_model('gpt-4o')
    tokens = encoding.encode(page_markdown)

    if len(tokens) > 950_000:
        summary_model = "google/gemini-pro-1.5"
    elif len(tokens) > 190_000:
        summary_model = "google/gemini-flash-1.5-8b"
    # elif len(tokens) > 128_000:
    else:
        summary_model = "anthropic/claude-3-haiku:beta"
    # elif len(tokens) > 32_000:
    #     summary_model = "cohere/command-r-08-2024"
    # else:
    #     summary_model = "qwen/qwen-2.5-7b-instruct"

    print(f"WEB_PAGE> Using summary model (tokens: {len(tokens)}): {summary_model}")

    summary_response = print_openrouter_response(
        model=summary_model,
        messages=[
            {'role': 'system', 'content': f'Summarize the content of the webpage-cache. Only respond with the summary and nothing else. Focus on the information most relevant to this query: {query}'},
            {'role': 'user', 'content': message_parts}
        ],
        print_role='WEB_PAGE'
    )
    summary_text = summary_response['content']

    return [{
        'role': 'tool',
        'tool_call_id': tool_call_id,
        'content': summary_text
    }]

tool_list = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet for information on a given topic. Open web pages to find information instead of relying on the snippet provided. Only use the snippet to filter for relevant web pages.",
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
            "name": "web_page_content",
            "description": "Get the content of a webpage-cache.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The url to open."
                    },
                    "query": {
                        "type": "string",
                        "description": "A query to refine the web page summary."
                    }
                },
                "required": ["url"],
                "additionalProperties": False
            }
        }
    }
]
tool_map = {
    'web_search': web_search,
    'web_page_content': web_page_content
}

basic_tool_string = '\n'.join([f'{i + 1}. {tool["function"]["name"]} - {tool["function"]["description"]}' for i, tool in enumerate(tool_list)])

def complete_task_step(step: str, nsfw: bool, task_messages: list[dict], is_sub_call: bool = False, super_call_messages: list[dict] = None) -> list[dict]:
    step_messages = []

    if not super_call_messages:
        super_call_messages = []

    # Complete task
    while True:
        messages = [
            {'role': 'system', 'content': f'Complete the task provided by the user.'
                                          f'Think in depth about what to do in <thinking> tags before doing anything else.'
                                          f''
                                          f'Be sure to get information from multiple sources to ensure correctness.'
                                          f''
                                          f'Provide your response in <response> tags.'
                                          f''
                                          f'The current date is {date_string()}.'
                                          f'When thinking about dates, first write down the dates, then compare the dates after you have written down the dates.'
                                          f'For example, if you are comparing September 22nd, 2006 and February 4th, 2023, first write both down, then compare them:'
                                          f''
                                          f'<example>'
                                          f'The two dates I need to compare are:'
                                          f'1. September 22nd, 2006'
                                          f'2. February 4th, 2023'
                                          f'The year 2006 occurred before 2023, so September 22nd, 2006 occurred before February 4th, 2023.'
                                          f'</example>'
                                          f''
                                          f'Do not say that an event has not happened yet or that a date is hypothetical until you have compared the date to the current one.'},
            *task_messages,
            {'role': 'user', 'content': f'{step}'},
            *super_call_messages,
            *step_messages
        ]

        if is_sub_call:
            messages.append({'role': 'user', 'content': f'Continue with the task `{step}` taking into account your reflections.'})

        response = print_openrouter_response(
            model='anthropic/claude-3.5-sonnet:beta',
            messages=messages,
            tools=[
                *tool_list,
                {
                    "type": "function",
                    "function": {
                        "name": "finish",
                        "description": "Call when you have finished the task. This should be the last tool called.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False
                        }
                    }
                }
            ]
        )

        tool_call_messages = []

        finished = False
        if response.get('tool_calls'):
            for tool_call in response['tool_calls']:
                tool_name = tool_call['function']['name']
                if tool_call['function'].get('arguments'):
                    tool_args = json.loads(tool_call['function']['arguments'])
                else:
                    tool_args = {}
                tool_id = tool_call['id']

                if tool_name == 'finish':
                    if response.get('content'):
                        step_messages.append({'role': 'assistant', 'content': response['content']})
                        finished = True
                        break

                tool_function = tool_map.get(tool_name)
                if not tool_function:
                    print(f"TOOL_CALL> Unknown tool: {tool_name}")
                    continue

                tool_messages = tool_function(tool_args, tool_id, [
                    *task_messages,
                    {'role': 'user', 'content': f'{step}'},
                    *super_call_messages,
                    *step_messages
                ])
                tool_call_messages.extend(tool_messages)
        else:
            finished = True

        if finished:
            break

        step_messages.append(response)
        step_messages.extend(tool_call_messages)

    # Reflect on task
    reflection_response = print_openrouter_response(
        model='anthropic/claude-3.5-sonnet:beta',
        messages=[
            {'role': 'system', 'content': f'Reflect on the task you just completed.'
                                          f'Think about whether the task has been fully completed and if it was completed correctly.'
                                          f'Respond with your reflection in <reflection> tags.'
                                          f'After your reflection, respond with either <complete> or <incomplete> if it was complete or incomplete.'
                                          f''
                                          f'Do not call any tools.'
                                          f'It is okay to expand the scope of your task if you find it necessary.'
                                          f''
                                          f'The current date is {date_string()}.'
                                          f'When thinking about dates, first write down the dates, then compare the dates after you have written down the dates.'
                                          f'For example, if you are comparing September 22nd, 2006 and February 4th, 2023, first write both down, then compare them:'
                                          f''
                                          f'<example>'
                                          f'The two dates I need to compare are:'
                                          f'1. September 22nd, 2006'
                                          f'2. February 4th, 2023'
                                          f'The year 2006 occurred before 2023, so September 22nd, 2006 occurred before February 4th, 2023.'
                                          f'</example>'
                                          f''
                                          f'Do not say that an event has not happened yet or that a date is hypothetical until you have compared the date to the current one.'},
            *task_messages,
            {'role': 'user', 'content': f'{step}'},
            *super_call_messages,
            *step_messages,
            {'role': 'user', 'content': f'The task you just completed was: {step}'}
        ],
        tools=tool_list,
        print_role='reflection'
    )
    step_messages[-1]['content'] += f"\n\n{reflection_response['content']}"
    reflection = extract_xml_tag(reflection_response['content'], 'reflection')
    is_complete = "<complete>" in reflection_response['content']
    if not is_complete:
        print("TASK> Task is incomplete, retrying.")
        retry = complete_task_step(step, nsfw, task_messages, is_sub_call=True, super_call_messages=[*super_call_messages, *step_messages])
        step_messages.extend(retry)

    if not is_sub_call:
        final_messages = [
            {'role': 'user', 'content': f'{step}'},
            *step_messages
        ]
    else:
        final_messages = step_messages

    return final_messages

def complete_user_task(user_message: dict, nsfw: bool) -> dict:
    task_messages = []
    while True:
        task_response = print_openrouter_response(
            model='qwen/qwen-2.5-7b-instruct',
            messages=[
                {'role': 'system', 'content': f'Take the user message and determine the next step to complete the task.'
                                              f'Think about what the next step should be in <thinking> tags and respond with the nest step in <step> tags.'
                                              f'The step should be concise and provide all needed information.'
                                              f'Do not put anything other than a step to complete inside of <step> tags.'
                                              f'Do not respond with <response> tags.'
                                              f'You must respond with both <thinking> and <step> tags.'
                                              f'Do not complete the task, just provide the task.'
                                              f''
                                              f'The current date is {date_string()}.'
                                              f'Do not make assumptions based on the date.'
                                              f''
                                              f'Available tools:'
                                              f'{basic_tool_string}'},
                *task_messages,
                user_message
            ],
            print_role='task'
        )
        steps_finished = '<step_finish>' in task_response['content']
        if steps_finished:
            print("TASK> Task is finished.")
            break

        task_step = extract_xml_tag(task_response['content'], 'step')
        print(f"STEP> {task_step}")

        step_messages = complete_task_step(task_step, nsfw, task_messages)

        for message in step_messages:
            if message['role'] == 'assistant':
                if message.get('tool_calls'):
                    del message['tool_calls']
            elif message['role'] == 'tool':
                message['role'] = 'user'
                del message['tool_call_id']

            task_messages.append(message)

        reflection_response = print_openrouter_response(
            model='anthropic/claude-3.5-sonnet:beta',
            messages=[
                {'role': 'system', 'content': f'Reflect on the task you just completed.'
                                              f'Think about whether the task has been fully completed and if it was completed correctly.'
                                              f'Respond with your reflection in <reflection> tags.'
                                              f'After your reflection, respond with either <complete> or <incomplete> if it was complete or incomplete.'
                                              f''
                                              f'Do not call any tools.'
                                              f'It is okay to expand the scope of the task if you find it necessary.'
                                              f''
                                              f'The current date is {date_string()}.'
                                              f'When thinking about dates, first write down the dates, then compare the dates after you have written down the dates.'
                                              f'For example, if you are comparing September 22nd, 2006 and February 4th, 2023, first write both down, then compare them:'
                                              f''
                                              f'<example>'
                                              f'The two dates I need to compare are:'
                                              f'1. September 22nd, 2006'
                                              f'2. February 4th, 2023'
                                              f'The year 2006 occurred before 2023, so September 22nd, 2006 occurred before February 4th, 2023.'
                                              f'</example>'
                                              f''
                                              f'Do not say that an event has not happened yet or that a date is hypothetical until you have compared the date to the current one.'},
                {'role': 'user', 'content': f'{user_message}'},
                *task_messages,
                {'role': 'user', 'content': f'The task you just completed was: {user_message}'},
            ],
            tools=tool_list,
            print_role='reflection'
        )
        step_messages[-1]['content'] += f"\n\n{reflection_response['content']}"
        reflection = extract_xml_tag(reflection_response['content'], 'reflection')
        is_complete = "<complete>" in reflection_response['content']
        if is_complete:
            print("TASK> Task is complete.")
            break

    # Final response
    final_response = print_openrouter_response(
        model='anthropic/claude-3.5-sonnet:beta',
        messages=[
            {'role': 'system', 'content': f'Complete the users task. Think about how to respond in <thinking> tags before responding. Respond in <response> tags. Today\'s date is {date_string()}.'},
            *task_messages,
            user_message
        ],
        print_role='final_response'
    )
    final_response_text = final_response['content']
    response = extract_xml_tag(final_response_text, 'response')

    return {'role': 'assistant', 'content': response}

def main() -> int:
    while True:
        user_message = get_user_input()
        message_is_nsfw = detect_nsfw(user_message)

        response = complete_user_task(user_message, message_is_nsfw)

    return 0

if __name__ == '__main__':
    sys.exit(main())
