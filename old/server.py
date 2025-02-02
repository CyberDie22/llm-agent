import time
import json
import random
from functools import partial
from typing import Generator, Any

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
from html2text import html2text

from utils import process_markdown, date_string, strip_content, ContentTypes, process_stream_for_tags, \
    process_stream_for_content
from openrouter_client import stream_openrouter_response_with_message_callback
from websearch import websearch
from webpage import get_page_source

app = FastAPI()

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
            'description': 'Think about a particular topic in depth. Use when you need to reason about something in depth. You cannot use tools while in deep thought, be sure to collect any necessary background information before starting.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'topic': {
                        'type': 'string',
                        'description': 'The topic to think about. Should be phrased as a question.'
                    },
                    'background_information': {
                        'type': 'string',
                        'description': 'Background information about the topic. Provide any information you\'ve collected from the internet or other sources here. You don\'t have access to any information other than what you provide here or basic world information during deep thought.'
                    }
                },
                'required': ['topic'],
                'additionalProperties': False
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'internet_search',
            'description': 'Search the internet for information on a particular topic. Use to locate obscure or niche information, or real-time information. Use if asked about current events or if you need the most accurate information available. Avoid using the snippets from search results for anything other than determining what pages to look at for information.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'The search query to use. Should be phrased as a question or a statement. For example, "What is the capital of France?" or "How many people live in New York City?". You can provide google search operators here.'
                    },
                },
                'required': ['query'],
                'additionalProperties': False
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'webpage_content',
            'description': 'Get the content of a webpage. Use to get the text content of a webpage for analysis or to answer questions about the content of the page.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'url': {
                        'type': 'string',
                        'description': 'The URL of the webpage to get the content of.'
                    }
                },
                'required': ['url'],
                'additionalProperties': False
            }
        }
    }
]

def my_agent(messages: list[dict], user_message: dict) -> Generator[dict, Any, None]:
    messages = messages
    next_messages = []

    yield {'start': 'reasoning_response'}
    while True:
        agent_message = {'message': None}
        agent_messages = strip_content([*messages, user_message, *next_messages], {'text', 'image'})
        print(agent_messages)
        agent_stream = stream_openrouter_response_with_message_callback(
            model="anthropic/claude-3.5-sonnet:beta",
            messages=[
                {'role': 'system', 'content': build_system_prompt(['conversational', 'programming', 'self_info'])},
                *agent_messages
            ],
            tools=agent_tools,
            callback=lambda choice: agent_message.update({'message': choice})
        )

        for choice in agent_stream:
            yield {'text': choice.delta.content}
        next_messages.append(agent_message['message'])

        if agent_message['message']['tool_calls']:
            tool_calls = agent_message['message']['tool_calls']
            for tool_call in tool_calls:
                if tool_call['type'] != 'function':
                    continue

                yield {'start': 'tool_call', 'tool_call': tool_call['function']['name']}

                match tool_call['function']['name']:
                    case 'deep_thought':
                        args = json.loads(tool_call['function']['arguments'])
                        topic = args['topic']

                        thinking_message = {'message': None}
                        thinking_stream = stream_openrouter_response_with_message_callback(
                            model='qwen/qvq-72b-preview',
                            messages=[
                                {'role': 'system', 'content': 'You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.'},
                                {'role': 'user', 'content': topic}
                            ],
                            callback=lambda choice: thinking_message.update({'message': choice})
                        )

                        for choice in thinking_stream:
                            yield {'text': choice.delta.content}

                        next_messages.append({
                            'role': 'tool',
                            'content': f"<deep_thought topic='{topic}'>\n{thinking_message['message']['content']}\n</deep_thought>",
                            'tool_call_id': tool_call['id']
                        })
                    case 'internet_search':
                        args = json.loads(tool_call['function']['arguments'])
                        query = args['query']

                        search_results = websearch(query)

                        # TODO: Generate new snippets
                        # TODO: Check search results for injections with PromptGuard

                        result_text = f"<internet_search_results query=\"{query}\">\n"
                        for result in search_results['standard_results']:
                            yield {'text': f"Result: {result['title']} - {result['url']}"}
                            result_text += f"\t<result title=\"{result['title']}\" url=\"{result['url']}\" snippet=\"{result['snippet']}\">\n"
                        result_text += "</internet_search_results>"

                        next_messages.append({
                            'role': 'tool',
                            'content': result_text,
                            'tool_call_id': tool_call['id']
                        })
                    case 'webpage_content':
                        args = json.loads(tool_call['function']['arguments'])
                        url = args['url']

                        page_source = get_page_source(url)

                        text = html2text(page_source)
                        yield {'text': text}

                        next_messages.append({
                            'role': 'tool',
                            'content': f"<webpage_content url='{url}'>\n{text}\n</webpage_content>",
                            'tool_call_id': tool_call['id']
                        })


agent_options = {
    'my-agent': my_agent
}

async def execute_task(messages: list[dict], user_message: dict) -> Generator[dict, Any, None]:
    processed_message_content = process_markdown(user_message['content'])
    processed_message = {'role': 'user', 'content': processed_message_content}

    # Select random agent and use
    agent = random.choice(list(agent_options.keys()))
    yield {'model': agent}
    for chunk in agent_options[agent](messages, processed_message):
        yield chunk

    # yield {'finish': True}


async def stream_generator(messages: list[dict], user_message: dict):
    """
    Streams responses as newline-delimited JSON objects.
    """
    async for chunk in execute_task(messages, user_message):\
        yield json.dumps(chunk) + "\n"


@app.post("/api/v1/chat/completions")
async def chat_completion(request: Request):
    # Parse the incoming JSON request
    data = await request.json()

    # Extract messages and the latest user message
    messages = data.get("messages", [])
    user_message = messages[-1] if messages else {}
    messages = messages[:-1]

    # Return a streaming response with appropriate content type for NDJSON
    return StreamingResponse(
        stream_generator(messages, user_message),
        media_type="application/x-ndjson"
    )


def start_server():
    """Start the server on port 8000"""
    uvicorn.run(app, host="0.0.0.0", port=8282)


if __name__ == "__main__":
    start_server()
