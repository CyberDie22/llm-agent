import hashlib
import base64
import re
from urllib.parse import urlparse
from typing import Union, Optional

import httpx

from ai_interface import get_response
from fileformat import transcode_image, ImageFormat

def process_image(image: Union[str, bytes], name: Optional[str] = None) -> list | tuple[list, dict]:
    # TODO: deal with SVGs
    if isinstance(image, str):
        image_url = image
        if 'svg' in image_url:
            return [], {}
        if image.startswith('data:image/'):
            image_b64 = image.split(',')[1]
            image_bytes = base64.b64decode(image_b64)
            image_format = image.split(';')[0].split('/')[1]
        else:
            response = httpx.get(image)
            image_bytes = response.content
            image_format = None
            if image_url.endswith(('.jpg', '.jpeg')):
                image_format = 'jpeg'
    else:
        image_bytes = image
        image_format = None
        image_url = None

    if image_format and image_format == 'jpeg':
        jpeg_bytes = image_bytes
    else:
        jpeg_bytes = transcode_image(image_bytes, ImageFormat.JPEG)
    jpeg_b64 = base64.b64encode(jpeg_bytes).decode()

    # describe image with Gemini 2.0 Flash Experimental
    # description_response = get_response(
    #     model="google/gemini-2.0-flash-exp:free",
    #     messages=[
    #         {
    #             'role': 'user',
    #             'content': [
    #                 {'type': 'text', 'text': 'Create an in depth description of this image. Provide detailed and accurate information about everything in the image including any text, people, objects, as well as where in the image they are.'},
    #                 {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{jpeg_b64}', 'detail': 'high'}},
    #             ]
    #         }
    #     ]
    # )
    # description = description_response['content']
    description = "cat"

    # TODO: create regions with Florence 2?

    image_name = name
    if not image_name:
        image_name = f'image-{hashlib.md5(jpeg_bytes).hexdigest()}'

    return ([
        {'type': 'text', 'text': f'<image name="{image_name}">\n'},
        {'type': 'text', 'text': f'\t<image_data>'},
        {
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{jpeg_b64}',
                'detail': 'high'
            }
        },
        {'type': 'text', 'text': '</image_data>\n'},
        {'type': 'text', 'text': f"<image_description>\n{description}\n</description>\n"},
        {'type': 'text', 'text': f'</image>'}
    ], {
        'name': image_name,
        'url': image_url
    })


def process_markdown(md: str, return_images: bool = False, original_url: Optional[str] = None) -> tuple[list, list]:
    md_link_regex = re.compile(r'(!*\[(.*?)\]\((.*?)(?: ".*?")?\))')
    input_split = md_link_regex.split(md)

    images = []

    message_parts = []

    link_text = ""
    got_alt_text = False
    alt_text = ""
    url = ""
    for part in input_split:
        if got_alt_text and url == "":
            url = part
            print("link", link_text)
            print("alt", alt_text)
            print("url", url)

            if url.startswith('/') and original_url:
                url = urlparse(original_url).scheme + '://' + urlparse(original_url).netloc + url

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
                print("Processing image")
                processed_image_message, image_info = process_image(url)
                if processed_image_message:
                    message_parts.extend(processed_image_message)
                    images.append(image_info)
            elif is_audio_url:
                print("Processing audio")
                # TODO: Describe audio with Gemini
                # TODO: Transcribe audio with Whisper v3 Turbo (research to see if this is still the best available model)

                # message_parts.append({
                #     'type': 'input_audio',
                #     'input_audio': {
                #         'data': url,
                #         'format': url.split('.')[-1]
                #     }
                # })
            elif is_markdown_file:
                print("Processing markdown file")
                if return_images:
                    processed_md, processed_images = process_markdown(url, return_images=True, original_url=original_url)
                    message_parts.extend(processed_md)
                    images.extend(processed_images)
                else:
                    processed_md = process_markdown(url, original_url=original_url)
                    message_parts.extend(processed_md)
            else:
                print("Processing link")
                message_parts.append({
                    'type': 'text',
                    'text': link_text
                })

            link_text = ""
            got_alt_text = False
            alt_text = ""
            url = ""
            continue

        if link_text != "" and not got_alt_text:
            alt_text = part
            got_alt_text = True
            continue

        if md_link_regex.fullmatch(part):
            link_text = part
            continue

        print("Part does not match any pattern")
        message_parts.append({
            'type': 'text',
            'text': part
        })

    # condense sequential text parts
    condensed_message_parts = []
    current_text = ""
    for part in message_parts:
        if part['type'] == 'text':
            current_text += part['text']
        else:
            if current_text:
                condensed_message_parts.append({
                    'type': 'text',
                    'text': current_text
                })
                current_text = ""
            condensed_message_parts.append(part)
    if current_text:
        condensed_message_parts.append({
            'type': 'text',
            'text': current_text
        })

    if return_images:
        return condensed_message_parts, images
    else:
        return condensed_message_parts
