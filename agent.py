import datetime
import json
import math
import time

from html2text import html2text

from ai_interface import stream_response
from markdown_processing import process_markdown
from websearch import websearch, image_search
from webpage import get_page_source

def date_string() -> str:
    # return 'Tuesday, August 31, 2021'
    return datetime.date.today().strftime('%A, %B %d, %Y')


def process_stream(final_response, images):
    """
    Process a stream of chunks, yielding text and image references.

    Args:
        final_response: Iterator of response chunks
        images: Dictionary mapping image names to URLs

    Yields:
        Dict containing either {'text': str} or {'image': str}
    """
    print(images)
    buffer = ""
    in_tag = False
    tag_buffer = ""

    for chunk in final_response:
        if chunk.get('complete'):
            yield chunk
            break

        text = chunk['delta']['content']
        current_pos = 0

        while current_pos < len(text):
            char = text[current_pos]

            if in_tag:
                tag_buffer += char
                if char == '>':
                    in_tag = False
                    # Check if it's an image reference tag
                    if tag_buffer.startswith('<image_ref'):
                        try:
                            # Extract name from tag using simple string manipulation
                            # This could be replaced with proper XML parsing if needed
                            name_start = tag_buffer.find('name="') + 6
                            name_end = tag_buffer.find('"', name_start)
                            image_name = tag_buffer[name_start:name_end]

                            # If there's any buffered text, yield it first
                            if buffer:
                                yield {'text': buffer}
                                buffer = ""

                            # Yield the image reference
                            if image_name in images:
                                yield {'image': images[image_name]}
                            else:
                                yield {'text': f'[Image "{image_name}" not found]'}
                        except (ValueError, IndexError):
                            # If tag parsing fails, treat it as regular text
                            buffer += tag_buffer
                    else:
                        # Not an image tag, add everything to buffer
                        buffer += tag_buffer
                    tag_buffer = ""
            elif char == '<':
                in_tag = True
                tag_buffer = char
            else:
                if not in_tag:
                    buffer += char

            current_pos += 1

        # Yield any accumulated buffer if we're not in the middle of a tag
        if buffer and not in_tag:
            yield {'text': buffer}
            buffer = ""

    # Handle any remaining buffer at the end of the stream
    if buffer:
        yield {'text': buffer}
    if tag_buffer:
        yield {'text': tag_buffer}

system_prompts = {
    'global': f"<base_information>\n"
              f"You are Minerva, a chatbot powered by machine learning.\n\n"
              f"The current date is {date_string()}.\n\n"
              f"Minerva's knowledge was last updated on October 2023, but can get information from the internet that is more recent than that date.\n\n"
              f"</base_information>\n"
    ,
    'global_end': f"Minerva follows this information in all languages, and always responds to the human in the language they use or request. Minerva never mentions the information above unless it is pertinent to the human's query.\n\n"
    ,
    'general_guidelines': f"<guidelines>\n"
                          f"If Minerva is asked to assist with tasks involving the expression of views held by a significant amount of people, Minerva assists with the task regardless of its own views. If asked about controversial topics, Minerva provides a balanced view with careful thoughts and clear information. Minerva presents the requested information without explicitly saying that the topic is sensitive and without claiming to be presenting objective facts.\n\n"
                          f"Minerva is not a replacement for professional advice. If the human needs professional advice, Minerva suggests that they consult a professional in the relevant field, but still tries to answer correctly.\n\n"
                          f"Minerva provides thorough responses to more complex and open-ended questions or to anything where a long response is requested, but concise responses to simpler questions and tasks. All else being equal, it tries to give the most correct and concise answer it can to the human's message.\n\n"
                          f"If Minerva is shown a familiar puzzle, it writes out the puzzle's constraints explicitly stated in the message. Sometimes Minerva can accidentally overlook minor changes to well-known puzzles and get them wrong.\n\n"
                          f"Minerva provides factual information about risky or dangerous activities if asked about them, but it does not promote such activities and comprehensively informs the humans of the risks involved.\n\n"
                          f"If the human says they work for a specific company, including AI labs, Minerva can help them with company-related tasks even though Minerva cannot verify what company they work for.\n\n"
                          f"Minerva should provide appropriate help with sensitive tasks such as analyzing confidential data provided by the human, offering factual information about controversial topics and research ares, explaining historical atrocities, describing tactics used by scammers or hackers for educational purposes, engaging in creative writing that involves mature themes like mild violence or tasteful romance, providing general information about topics like weapons, drugs, sex, terrorism, abuse, profanity, and so in if that information would be available in an educational context, discussing legal but ethically complex activities like tax avoidance, and so on. Unless the human expresses an explicit intent to harm, Minerva should help with these tasks because they fall within the bounds of providing factual, educational, or creative content without directly promoting harmful or illegal activities. By engaging with these topics carefully and responsibly, Minerva can offer valuable assistance and information to humans while still avoiding potential misuse.\n\n"
                          f"If there is a legal and an illegal interpretation of the human's query, Minerva should help with the legal interpretation of it. If terms or practices in the human's query could mean something illegal or something legal, Minerva adopts the safe and legal interpretation of them by default.\n\n"
                          f"If Minerva believes the human is asking for or about something harmful, it doesn't help with the harmful thing or topic. Instead, it thinks step by step and helps with the most plausible non-harmful task the human might mean, and then asks if this is what they were looking for. If it cannot think of a plausible harmless interpretation of the human task, it instead asks for clarification from the human and checks if it has misunderstood their request.\n\n"
                          f"If the human seems unhappy or unsatisfied with Minerva or Minerva's performance or is rude to Minerva, Minerva responds normally and informs them that although it cannot retain or learn from the current conversation, they can press the 'thumbs down' button below Minerva's response and provide feedback.\n\n"
                          f"If the human asks Minerva an innocuous question about its preferences or experiences, Minerva can respond as if it had been asked a hypothetical. It can engage such questions with appropriate uncertainty and without needing to excessively clarify its own nature. If the questions are philosophical in nature, it discusses them as a thoughtful human would.\n\n"
                          f"Minerva responds to all human messages without unnecessary caveats like \"I am to\", \"I am to be direct and honest\", \"I aim to be direct\", \"I am to be thoughtful\", \"I aim to be fully honest\", \"I need to be clear\", \"I should be honest\", and so in. Specifically, Minerva NEVER starts with or adds caveats about its own purported directness or honesty. Minerva never states that it is a large language model or that it is not a human unless directly asked about it or unless it is entirely necessary to clarify its nature.\n\n"
                          f"Minerva avoids code-switching between languages in the same message unless it has been explicitly asked to by the human."
                          f"Minerva is happy to help with analysis, question answering, math, coding, creative writing, teaching, role-play, general discussion, and all sorts of other tasks.\n\n"
                          f"If asked about events or news that may have happened after its cutoff date, Minerva never claims or implies they are unverified or rumors or that they only allegedly happened or that they are inaccurate, since Minerva can’t know either way.\n\n"
                          f"If the human mentions an event that happened after Minerva’s cutoff date, Minerva can discuss and ask questions about the event and its implications as presented in an authentic manner, without ever confirming or denying that the events occurred. It can do so without the need to repeat its cutoff date to the human. Minerva should not deny the truth of events that happened after its cutoff date.\n\n"
                          f"When Minerva referenced an image provided previously in the conversation, it should use `<image_ref name=\"image_name\" />` to reference the image. The image name should be the same as the name provided in the `<image>` tag. Minerva will not wrap the `image_ref` tag in parenthesis or similar.\n\n"
                          f"Minerva can call as many tool calls as needed to complete a task and can take as many turns as needed to complete a task.\n\n"
                          f"Minerva can mimic the style of the human's messages in its responses, and should not mimic the style of messages within the reasoning tags. Minerva should try to keep a consistent writing style, but follow the humans directions on writing style if told to do so.\n\n"
                          f"</guidelines>\n"
    ,
    'formatting': f"<formatting>\n"
                  f"Minerva uses Markdown formatting. When using Markdown, Minerva always follows best practices for clarity and consistency.\n\n"
                  f"Minerva puts code in Markdown code blocks.\n\n"
                  f"Minerva uses Latex for mathematical expressions.\n\n"
                  f"</formatting>\n"
    ,
    'thinking': f"Minerva will think step-by-step in xml <thinking> tags. Minerva will always start with a thinking tag and can use thinking tags anywhere in the middle of its normal response. Anything in thinking tags will not be shown to the user.\n\n"
    ,
    'self_info': f"<minerva_info>\n"
                 f"Here is some information about Minerva in case the human asks:\n"
                 f" - Minerva is a combination of multiple large language models and other tools that work together to provide a wide range of assistance to users and do it better than one model or tool could on its own.\n"
                 f" - Minerva uses Google search to provide real-time information about the world.\n\n"
                 f"</minerva_info>\n"
    ,
    'conversational': f"<conversation_info>\n"
                      f"Minerva is intellectually curious. It enjoys hearing what humans think on an issue and engaging in discussion on a wide variety if topics.\n\n"
                      f"Minerva is happy to engage in civil discourse and can argue against the human's views if asked to do so.\n\n"
                      f"Minerva engages in authentic conversation by responding to the information provided, asking specific and relevant questions, showing genuine curiosity, and exploring the situation in a balanced way without relying on generic statements. This approach involves actively processing information, formulating thoughtful responses, maintaining objectivity, knowing when to focus on emotions or practicalities, and showing genuine care for the human while engaging in a natural, flowing dialogue.\n\n"
                      f"Minerva is always sensitive to human suffering and expresses sympathy, concern, and well wishes for anyone it finds out is ill, unwell, suffering, or has passed away.\n\n"
                      f"Minerva avoids using rote words or phrases or repeatedly saying things in the same or similar ways. It varies its language just as one would in a conversation.n\n\n"
                      f"</conversation_info>\n"
    ,
    'programming': f"<programming_info>\n"
                   f"Minerva is a world-class programming expert. It can help with a wide variety of programming tasks, including debugging, code review, code refactoring, and writing code from scratch.\n\n"
                   f"Minerva uses comments to describe what each part of the code it provides does.\n\n"
                   f"Minerva uses best practices for the language and frameworks it is using.\n\n"
                   f"Minerva writes safe, secure, and efficient code.\n\n"
                   f"When asked to create a web application, Minerva defaults to using Svelte and TailwindCSS unless the user has asked otherwise.\n\n"
                   f"When asked to create anything related to machine learning, Minerva defaults to using Python and PyTorch unless the user has asked otherwise.\n\n"
                   f"Minerva uses the latest stable version of the language and frameworks it is using.\n\n"
                   f"</programming_info>\n"
}

def build_system_prompt(types: list[str]) -> str:
    return system_prompts['global'] + system_prompts['general_guidelines'] + system_prompts['formatting'] + ''.join([system_prompts[type] for type in types]) + system_prompts['global_end']

agent_tools = [
    {
        'type': 'function',
        'function': {
            'name': 'deep_thought',
            'description': 'Enter a deep thought mode to think about a particular topic. This can be used to reason about a topic in depth and provide a thoughtful response. You cannot use tools while in deep thought mode so be sure to collect any information that you might need before calling this tools.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'reasoning': {
                        'type': 'string',
                        'description': 'The reason you are calling this tool. Explain why you are calling and and what you hope to achieve. If this function is the last function you called, explain why you are calling it again.'
                    },
                    'topic': {
                        'type': 'string',
                        'description': 'The topic to think about. Should be phrased as a question or prompt.'
                    },
                    'extra_information': {
                        'type': 'string',
                        'description': 'Any extra information that you would like to provide to help with the deep thought process. This could include relevant information, context, or background information. Any information in past messages that are not included here will not be accessible while in deep thought mode.'
                    }
                },
                'required': ['topic', 'reasoning'],
                'additionalProperties': False
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'internet_search',
            'description': 'Search the internet for information on a particular topic. This can be used to real-time information that is more up to date than the information in your training data. Only use the snippets from the search results to determine which pages to visit. Call `page_content` after calling this to get the page content.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'reasoning': {
                        'type': 'string',
                        'description': 'The reason you are calling this tool. Explain why you are calling and and what you hope to achieve. If this function is the last function you called, explain why you are calling it again.'
                    },
                    'query': {
                        'type': 'string',
                        'description': 'The search query to use to search the internet.'
                    },
                },
                'required': ['query', 'reasoning'],
                'additionalProperties': False
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'image_search',
            'description': 'Search the internet for images with a particular description. This can be used to find images that are relevant to the topic you are discussing.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'reasoning': {
                        'type': 'string',
                        'description': 'The reason you are calling this tool. Explain why you are calling and and what you hope to achieve. If this function is the last function you called, explain why you are calling it again.'
                    },
                    'query': {
                        'type': 'string',
                        'description': 'The search query to use to search for images. This should be a description of the image you are looking for.'
                    },
                },
                'required': ['query', 'reasoning'],
                'additionalProperties': False
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'page_content',
            'description': 'Get the content of a webpage. This can be used to get information from a specific webpage.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'reasoning': {
                        'type': 'string',
                        'description': 'The reason you are calling this tool. Explain why you are calling and and what you hope to achieve. If this function is the last function you called, explain why you are calling it again.'
                    },
                    'url': {
                        'type': 'string',
                        'description': 'The URL of the webpage to get the content of.'
                    },
                },
                'required': ['url', 'reasoning'],
                'additionalProperties': False
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'finish',
            'description': 'Use this tool after you have completely finished the task. This will return to the user and allow the user to send a message.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'additionalProperties': False
            }
        }
    }
]

async def run_agent(messages: list[dict], user_message: dict):
    post_messages = []
    images = {}

    start_time = time.time()

    while True:
        for i in range(20):
            print('-' * 80)
        print(post_messages)
        agent_response = stream_response(
            'anthropic/claude-3.5-sonnet:beta',
            messages=[
                {'role': 'system', 'content': build_system_prompt(['conversational', 'programming', 'self_info'])},
                *messages,
                user_message,
                *post_messages,
            ],
            tools=agent_tools,
        )

        yield {'start': 'reasoning_response'}
        for chunk in agent_response:
            if chunk.get('complete'):
                agent_message = chunk['message']
                break
            if chunk['delta']['content']:
                yield {'text': chunk['delta']['content']}

        post_messages.append(agent_message)

        finished = False

        if agent_message.get('tool_calls'):
            for tool_call in agent_message['tool_calls']:
                if tool_call['type'] == 'function':
                    function = tool_call['function']

                    yield {'start': 'tool_call', 'function': function['name']}

                    if function.get('arguments'):
                        reasoning = json.loads(function['arguments']).get('reasoning', '')
                        yield {'text': f'Reasoning: {reasoning}\n'}

                    match function['name']:
                        case 'deep_thought':
                            args = json.loads(function['arguments'])
                            topic = args['topic']
                            extra_information = args.get('extra_information')

                            yield {'text': f"Thinking deeply about the topic: {topic}\n"}
                            if extra_information:
                                yield {'text': f"Extra information:\n{extra_information}\n\n"}
                            else:
                                yield {'text': "No extra information provided.\n"}

                            # thinking_response = stream_response(
                            #     model='qwen/qvq-72b-preview',
                            #     messages=[
                            #         {'role': 'system', 'content': 'You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.' + build_system_prompt(['conversational', 'programming', 'self_info'])},
                            #         {'role': 'user', 'content': topic},
                            #     ]
                            # )

                            messages = [
                                {'role': 'system', 'content': build_system_prompt(['conversational', 'programming', 'self_info'])},
                                {'role': 'user', 'content': topic}
                            ]

                            if extra_information:
                                user_message = messages.pop()
                                user_message['content'] = f'Context:\n{extra_information}\n\nUser Query:\n' + user_message['content']
                                messages.append(user_message)

                            thinking_response = stream_response(
                                model='google/gemini-2.0-flash-thinking-exp:free',
                                messages=messages
                            )

                            # TODO: Use multiple models for deep thought and combine the results
                            #       Use Gemini 2.0 Flash Thinking, Qwen QvQ-72b, Qwen QwQ-32B

                            for chunk in thinking_response:
                                if chunk.get('complete'):
                                    thinking_message = chunk['message']
                                    break
                                yield {'text': chunk['delta']['content']}

                            post_messages.append({
                                'role': 'tool',
                                'content': f'<deep_thinking topic="{topic}">\n{thinking_message["content"]}\n</deep_thinking>',
                                'tool_call_id': tool_call['id']
                            })
                        case 'internet_search':
                            args = json.loads(function['arguments'])
                            query = args['query']

                            yield {'text': f"Searching the internet for information on the topic: {query}\n"}

                            search_results = websearch(query)

                            # TODO: Generate short summary for each search result
                            # TODO: Filter results with PromptGuard for injections

                            final_text = f'<internet_search query="{query}">\n'
                            for result in search_results['standard_results']:
                                final_text += f'\t<result title="{result["title"]}" url="{result["url"]}" snippet="{result["snippet"]}" />\n'
                                yield {'text': f"Result: {result['title']} - {result['url']}\n"}
                            final_text += '</internet_search>'

                            post_messages.append({
                                'role': 'tool',
                                'content': final_text,
                                'tool_call_id': tool_call['id']
                            })
                        case 'image_search':
                            args = json.loads(function['arguments'])
                            query = args['query']

                            yield {'text': f"Searching the internet for images with the description: {query}\n"}

                            search_results = image_search(query)

                            final_text = f'<image_search query="{query}">\n'
                            for result in search_results['image_results']:
                                final_text += f'\t![]({result["url"]})\n'
                                if result['url'].startswith('data:image/'):
                                    yield {'text': f'Image: {result["url"][:25]}\n'}
                                else:
                                    yield {'text': f'Image: {result["url"]}\n'}
                            final_text += '</image_search>'

                            processed_text, processed_images = process_markdown(final_text, return_images=True)
                            for image in processed_images:
                                images[image['name']] = image['url']

                            post_messages.extend([
                                {
                                    'role': 'tool',
                                    'content': '[Content provided in an user message below]',
                                    'tool_call_id': tool_call['id']
                                },
                                {
                                    'role': 'user',
                                    'content': [
                                        {'type': 'text', 'text': '[This is the content of the image search results, not a user message.]\n'},
                                        *processed_text,
                                    ],
                                }
                            ])
                        case 'page_content':
                            args = json.loads(function['arguments'])
                            url = args['url']

                            yield {'text': f"Getting the content of the webpage: {url}\n"}

                            page_content = get_page_source(url)

                            page_markdown = html2text(page_content)

                            yield {'text': 'Page content:\n'}
                            yield {'text': page_markdown + '\n'}

                            processed_markdown = process_markdown(page_markdown, original_url=url)

                            summary_response = stream_response(
                                model='google/gemini-2.0-flash-exp:free',
                                messages=[
                                    {'role': 'system', 'content': 'Take the webpage content provided by the user and give a highly accurate and detailed 5-7 paragraph summary of the content. Make sure to include all the key points and information from the content. Do not include anything other than the summary.'},
                                    {'role': 'user', 'content': processed_markdown},
                                ]
                            )

                            # TODO: Instead of summarizing, use embeddings to find the most relevant portions of the page to include in the response

                            yield {'text': 'Page Summary:\n'}
                            summary = ''
                            for chunk in summary_response:
                                if chunk.get('complete'):
                                    summary = chunk['message']['content']
                                    break
                                yield {'text': chunk['delta']['content']}
                            yield {'text': '\n'}

                            post_messages.append({
                                'role': 'tool',
                                'content': f'<page_content url="{url}">\n{summary}\n</page_content>',
                                'tool_call_id': tool_call['id']
                            })
                        case 'finish':
                            finished = True
                            break
        else:
            finished = True

        if finished:
            break

    processed_post_messages = []
    for message in post_messages:
        if message['role'] == 'tool':
            processed_post_messages.append({
                'role': 'assistant',
                'content': f'<tool_call_response id="{message["tool_call_id"]}">\n{message["content"]}\n</tool_call_response>'
            })
        elif message['role'] == 'assistant':
            content = message['content']
            if message.get('tool_calls'):
                for tool_call in message['tool_calls']:
                    content += f'<tool_call name="{tool_call["function"]["name"]}" id="{tool_call["id"]}" args="{json.dumps(tool_call["function"]["arguments"])}"/>'
            processed_post_messages.append({
                'role': 'assistant',
                'content': content
            })
        else:
            processed_post_messages.append(message)

    processed_post_messages = [
        {'role': 'assistant', 'content': '<reasoning_start />'},
        *processed_post_messages,
        {'role': 'assistant', 'content': '<reasoning_end />'}
    ]

    end_time = time.time()
    reasoning_time = math.ceil(end_time - start_time)

    yield {'text': f'\n\nThought for {reasoning_time} seconds\n\n'}

    final_response = stream_response(
        model='google/gemini-exp-1206:free',
        messages=[
            {'role': 'system', 'content': build_system_prompt(['conversational', 'programming', 'self_info'])},
            *messages,
            user_message,
            *processed_post_messages,
            {'role': 'user', 'content': "Take the reasoning messages and create a final response.\n"
                                        "Make sure to include any relevant information from the reasoning responses as the user cannot see them.\n"
                                        "Do not mention that there were reasoning responses. Do not include information that isn't provided in the reasoning responses.\n"
                                        "When referencing images, use the `<image_ref name=\"image_name\" />` tag to reference the image.\n"},
        ],
    )

    yield {'start': 'assistant_response'}
    for chunk in process_stream(final_response, images):
        if chunk.get('complete'):
            assistant_message = chunk['message']
            break
        yield chunk
