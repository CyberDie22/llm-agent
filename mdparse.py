import io
import re
import base64
import httpx
from pathlib import Path

from contextlib import contextmanager
from functools import partial
from typing import Union, List, Optional

import svglib.svglib as svglib
from reportlab.graphics import renderPM

from messagetypes import TextContentPart, ImageContentPart
from media import ffmpeg_convert
from utils import cache_hash_blake2b

ContentPart = Union[TextContentPart, ImageContentPart]

# Core processing functions
@contextmanager
def get_image_content(url: str) -> io.BytesIO:
    """Safely retrieve image content from URL or base64"""
    file = None
    try:
        if url.startswith('data:'):
            svg_data = base64.b64decode(url.split(',')[1])
            file = io.BytesIO(svg_data)
        else:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(url)
                response.raise_for_status()
                file = io.BytesIO(response.content)
        yield file
    finally:
        if file:
            file.close()


def convert_to_png(image_data: io.BytesIO, is_svg: bool) -> io.BytesIO:
    """Convert image data to PNG format"""
    if is_svg:
        drawing = svglib.svg2rlg(image_data)
        png_file = io.BytesIO()
        renderPM.drawToFile(drawing, png_file, fmt="PNG")
        return png_file
    else:
        return ffmpeg_convert(image_data, 'png')


def get_cached_image(cache_path: Path) -> Optional[io.BytesIO]:
    """Retrieve cached image if it exists"""
    if cache_path.exists():
        jxl_bytes = cache_path.read_bytes()
        return ffmpeg_convert(io.BytesIO(jxl_bytes), 'png')
    return None


def cache_image(image: io.BytesIO, cache_path: Path) -> None:
    """Cache image in JXL format"""
    jxl_bytes = ffmpeg_convert(image, 'jxl').getvalue()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(jxl_bytes)


def get_cache_path(base_path: Path, cache_type: str, file_hash: str) -> Path:
    """Get the path for a cached file"""
    return base_path / cache_type / f"{file_hash}.jxl"


def process_image(url: str, cache_path: Path, is_svg: bool) -> io.BytesIO:
    """Process and cache an image, returning PNG bytes"""
    cached = get_cached_image(cache_path)
    if cached:
        return cached

    with get_image_content(url) as image_data:
        png_data = convert_to_png(io.BytesIO(image_data), is_svg)
        cache_image(png_data, cache_path)
        return png_data


def create_image_content(png_bytes: bytes, alt_text: str) -> List[ContentPart]:
    """Create content parts for an image"""
    png_b64 = base64.b64encode(png_bytes).decode('utf-8')
    url = 'data:image/png;base64,' + png_b64

    # TODO: describe image using Gemini Experimental 1121

    return [
        {'type': 'text', 'text': f'![{alt_text}]('} if alt_text else {'type': 'text', 'text': '!['},
        {'type': 'image', 'image_url': {'url': url, 'detail': alt_text}},
        {'type': 'text', 'text': ')'}
    ]


def process_file_link(path: str, process_fn: callable) -> List[ContentPart]:
    """Process a local file link"""
    content = Path(path).read_text()
    if path.endswith('.md'):
        return process_fn(content)
    return [{'type': 'text', 'text': content}]


file_markdown_regex = re.compile(r'!?\[(.*?)\]\((.+?)(?: ".*?")?\)')

def extract_markdown(md: str, cache_base: Path = Path('image-cache')) -> List[ContentPart]:
    """
    Process markdown content, handling images and links

    Args:
        md: Markdown content to process
        cache_base: Base directory for image cache

    Returns:
        List of processed content parts
    """

    def process_part(part: str, alt: str, url: str) -> List[ContentPart]:
        """Process a single markdown part"""
        if not part:
            return []

        if part.startswith('!') and url:  # Image
            is_svg_url = url.endswith('.svg')
            is_svg_b64 = url.startswith('data:image/svg+xml;base64,')
            is_svg = is_svg_url or is_svg_b64
            cache_type = 'svg-b64' if is_svg_b64 else 'svg-url' if is_svg_url else 'url'
            cache_hash = cache_hash_blake2b(
                url.encode('utf-8') if url.startswith('http')
                else base64.b64decode(url.split(',')[1])
            )

            cache_path = get_cache_path(cache_base, cache_type, cache_hash)
            png_bytes = process_image(url, cache_path, is_svg)
            return create_image_content(png_bytes, alt)

        elif url:  # Link
            if not url.startswith('http'):  # Local file
                return process_file_link(url, partial(extract_markdown, cache_base=cache_base))
            return [{'type': 'text', 'text': part}]

        return [{'type': 'text', 'text': part}]

    parts: List[ContentPart] = []
    curr_part = ""
    curr_alt = ""

    for part in file_markdown_regex.split(md):
        if curr_alt:
            parts.extend(process_part(curr_part, curr_alt, part))
            curr_part = ""
            curr_alt = ""
            continue

        if curr_part:
            curr_alt = part
            continue

        if file_markdown_regex.fullmatch(part):
            curr_part = part
            continue

        parts.extend(process_part(part, "", ""))

    return parts
