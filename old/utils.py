import base64
import datetime
import hashlib
import pathlib
import re
import subprocess
from typing import Literal, Optional, Union

import httpx

from fileformat import ImageFormat, transcode_image
from openrouter_client import openrouter_response


def cache_hash_blake2b(data: bytes) -> str:
    # Using 32 bytes (256 bits) output, can be adjusted
    return hashlib.blake2b(data, digest_size=32).hexdigest()

def date_string() -> str:
    # return 'Tuesday, August 31, 2021'
    return datetime.date.today().strftime('%A, %B %d, %Y')

def check_cache(cache_type: str, file_hash: str) -> Optional[bytes]:
    file_path = pathlib.Path(f'cache/{cache_type}/{file_hash}')
    if file_path.exists():
        with open(file_path, 'rb') as f:
            return f.read()
    else:
        return None

def cache_file(file: bytes, name: str, cache_type: str, hash_item: Optional[str] = None) -> str:
    if hash_item:
        file_hash = cache_hash_blake2b(b'\x00'.join([name.encode(), cache_type.encode(), hash_item]))
    else:
        file_hash = cache_hash_blake2b(b'\x00'.join([name.encode(), cache_type.encode(), file]))
    file_path = pathlib.Path(f'cache/{cache_type}/{hash}')
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'wb') as f:
        f.write(file)

    return file_hash

def extract_xml_tag(text: str, tag: str) -> str:
    start_tag = f'<{tag}>'
    end_tag = f'</{tag}>'

    start_index = text.index(start_tag) + len(start_tag)
    end_index = text.index(end_tag)

    return text[start_index:end_index].strip()

def process_stream_for_content(input_generator):
    for chunk in input_generator:
        content = chunk.delta.content
        if isinstance(content, str):
            yield {'text': content}
            continue
        for part in content:
            print(part)
            if part['type'] == 'text':
                yield {'text': content}


def process_stream_for_tags(input_generator):
    """
    Process a generator of {'text': str} objects, detecting XML tags and yielding
    special events when tags are found while removing them from the text stream.

    Args:
        input_generator: Generator yielding {'text': str} objects

    Yields:
        Dict objects in one of these formats:
        - {'text': str} for regular text content
        - {'tag_start': str} when an opening tag is detected
        - {'tag_end': str} when a closing tag is detected
    """
    buffer = ""
    tag_buffer = ""
    in_tag = False

    for chunk in input_generator:
        if 'text' not in chunk:
            yield chunk

        text = chunk['text']
        current_pos = 0

        while current_pos < len(text):
            char = text[current_pos]

            if char == '<':
                # Start of a potential tag
                if buffer:
                    yield {'text': buffer}
                    buffer = ""
                in_tag = True
                tag_buffer = char
            elif char == '>' and in_tag:
                # End of a tag
                tag_buffer += char
                tag_content = tag_buffer[1:-1]  # Remove < and >

                if tag_content.startswith('/'):
                    # Closing tag
                    yield {'tag_end': tag_content[1:]}
                else:
                    # Opening tag
                    yield {'tag_start': tag_content}

                tag_buffer = ""
                in_tag = False
            elif in_tag:
                # Building tag content
                tag_buffer += char
            else:
                # Regular text
                buffer += char

            current_pos += 1

        if buffer:
            yield {'text': buffer}
            buffer = ""

    # Handle any remaining content
    if buffer:
        yield {'text': buffer}
    if tag_buffer:
        yield {'text': tag_buffer}


def normalize_message_content(content: list | str) -> list:
    new_content = []

    if isinstance(content, str):
        return [{'type': 'text', 'text': content}]
    for item in content:
        if isinstance(item, str):
            new_content.append({'type': 'text', 'text': item})
        else:
            new_content.append(item)

    return new_content

ContentTypes = Literal['text', 'image', 'audio']
def strip_content(content: list | str, content_types: set[ContentTypes]) -> list:
    if len(content) == 0:
        return []

    if 'role' in content[0]:
        return [{'role': message['role'], 'content': strip_content(message['content'], content_types)} for message in content]

    content = normalize_message_content(content)

    allowed_types = []
    if 'text' in content_types:
        allowed_types.append('text')
    if 'image' in content_types:
        allowed_types.append('image_url')
    if 'audio' in content_types:
        allowed_types.append('input_audio')

    return [item for item in content if item['type'] in allowed_types]

def process_image(image: Union[str, bytes], name: Optional[str] = None) -> list:
    # TODO: deal with SVGs
    if isinstance(image, str):
        if image.startswith('data:image/'):
            image_b64 = image.split(',')[1]
            image_bytes = base64.b64decode(image_b64)
            image_url = None
        else:
            check_hash = cache_hash_blake2b(b'\x00'.join(['image', 'image-from-url', image]))
            image_bytes = check_cache('image-from-url', check_hash)
            if not image_bytes:
                response = httpx.get(image)
                image_bytes = response.content
                cache_file(image_bytes, 'image', 'image-from-url', image)
            image_url = image
    else:
        image_bytes = image
        image_url = None

    check_hash = cache_hash_blake2b(b'\x00'.join(['image', 'image-from-bytes', image_bytes]))
    jxl_bytes = check_cache('image-from-bytes', check_hash)
    if not jxl_bytes:
        jxl_bytes = transcode_image(image_bytes, ImageFormat.JXL)
        cache_file(jxl_bytes, 'image', 'image-from-bytes', image_url)

    jpeg_bytes = transcode_image(image_bytes, ImageFormat.JPEG)
    jpeg_b64 = base64.b64encode(jpeg_bytes).decode()

    # describe image with Gemini 2.0 Flash Experimental
    description_response = openrouter_response(
        model="google/gemini-2.0-flash-exp:free",
        messages=[
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'Create an in depth description of this image. Provide detailed and accurate information about everything in the image including any text, people, objects, as well as where in the image they are.'},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{jpeg_b64}', 'detail': 'high'}},
                ]
            }
        ]
    )
    description = description_response['content']

    # TODO: create regions with Florence 2?

    image_name = name
    if not image_name:
        image_name

    return [
        {'type': 'text', 'text': '![Image]('},
        {
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{jpeg_b64}',
                'detail': 'high'
            }
        }
    ]


def process_markdown(md: str) -> list:
    md_link_regex = re.compile(r'!*\[(.*?)\]\((.*?)(?: "(.*?)")?\)')
    input_split = md_link_regex.split(md)

    message_parts = []

    link_text = ""
    alt_text = ""
    url = ""
    for part in input_split:
        if md_link_regex.fullmatch(part):
            link_text = part
            continue

        if link_text and not alt_text:
            alt_text = part
            continue

        if alt_text and not url:
            url = part

            # check if url is an image
            is_image_url = url.endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg'))
            is_image_b64 = url.startswith('data:image/')
            is_image = is_image_url or is_image_b64
            # check if url is an audio file
            is_audio_url = url.endswith(('.mp3', '.wav'))

            # TODO: support video files
            # TODO: support pdf files https://github.com/microsoft/markitdown
            #       provide image of each page and the page extracted in markdown
            # TODO: cache images and audio files and descriptions
            # TODO: store images on image server and provide that URL

            # TODO: make httpx client global?

            is_markdown_file = url.endswith('.md')

            if is_image:
                message_parts.extend(process_image(url))
            elif is_audio_url:
                # TODO: Describe audio with Gemini
                # TODO: Transcribe audio with Whisper v3 Turbo (research to see if this is still the best available model)

                message_parts.append({
                    'type': 'input_audio',
                    'input_audio': {
                        'data': url,
                        'format': url.split('.')[-1]
                    }
                })
            elif is_markdown_file:
                message_parts.extend(process_markdown(url))
            else:
                message_parts.append({
                    'type': 'text',
                    'text': link_text
                })

            link_text = ""
            alt_text = ""
            url = ""
            continue

        message_parts.append({
            'type': 'text',
            'text': part
        })

    return message_parts
