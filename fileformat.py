import subprocess
import tempfile
import os
from typing import Union, Optional, Dict
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class ImageFormat(Enum):
    JPEG = "jpg"
    JXL = "jxl"
    PNG = "png"
    WEBP = "webp"
    AVIF = "avif"


@dataclass
class FormatSettings:
    codec: str
    extension: str


FORMAT_CONFIGS = {
    ImageFormat.JPEG: FormatSettings(
        codec="mjpeg",
        extension=".jpg",
    ),
    ImageFormat.JXL: FormatSettings(
        codec="libjxl",
        extension=".jxl",
    ),
    ImageFormat.PNG: FormatSettings(
        codec="png",
        extension=".png",
    ),
    ImageFormat.WEBP: FormatSettings(
        codec="libwebp",
        extension=".webp",
    ),
    ImageFormat.AVIF: FormatSettings(
        codec="libaom-av1",
        extension=".avif",
    )
}


def transcode_image(
        input_source: Union[bytes, str, Path],
        output_format: ImageFormat,
        keep_temp: bool = False,
        custom_params: Optional[Dict[str, Union[str, int, float]]] = None
) -> Union[bytes, str]:
    """
    Transcode image to specified format using ffmpeg.

    Args:
        input_source: Either image bytes or path to input image file
        output_format: Target format from ImageFormat enum
        keep_temp: If True and input is bytes, returns path to temp output file
                  instead of bytes. If input is a file path, this is ignored.
        custom_params: Optional dict of ffmpeg parameters to override defaults

    Returns:
        Union[bytes, str]: Either the transcoded bytes or path to the output file,
        depending on input type and keep_temp flag

    Raises:
        subprocess.CalledProcessError: If ffmpeg conversion fails
        FileNotFoundError: If input file path doesn't exist
        ValueError: If input_source type is invalid
    """
    input_is_bytes = isinstance(input_source, bytes)

    # Get format configuration
    format_config = FORMAT_CONFIGS[output_format]

    # Merge custom parameters with defaults
    params = {}
    if custom_params:
        params.update(custom_params)

    # Setup input source
    if input_is_bytes:
        input_file = tempfile.NamedTemporaryFile(suffix='.bin', delete=False)
        input_file.write(input_source)
        input_file.flush()
        input_path = input_file.name
    else:
        input_path = str(Path(input_source).resolve())
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

    # Create output file
    output_file = tempfile.NamedTemporaryFile(
        suffix=format_config.extension,
        delete=False
    )
    output_path = output_file.name
    output_file.close()

    try:
        # Build ffmpeg command
        cmd = ['ffmpeg', '-y', '-i', input_path]

        # Add codec
        # cmd.extend(['-c:v', format_config.codec])

        # Add format-specific parameters
        # for key, value in params.items():
        #     cmd.extend([f'-{key}', str(value)])

        # Add output path
        cmd.append(output_path)
        print(' '.join(cmd))

        # Run ffmpeg
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        # Handle output based on input type and keep_temp flag
        if input_is_bytes and not keep_temp:
            # Read output file and return bytes
            with open(output_path, 'rb') as f:
                output_bytes = f.read()
            os.unlink(output_path)
            return output_bytes
        else:
            # Return path to output file
            return output_path

    finally:
        # Clean up input file if it was temporary
        if input_is_bytes:
            os.unlink(input_path)

        # Clean up output file if something went wrong
        if input_is_bytes and not keep_temp and os.path.exists(output_path):
            os.unlink(output_path)


# Example usage:
if __name__ == '__main__':
    # Example 1: Convert to JPEG with custom quality
    with open('input.png', 'rb') as f:
        input_bytes = f.read()

    # Convert to high quality JPEG
    jpeg_bytes = transcode_image(
        input_bytes,
        ImageFormat.JPEG,
        custom_params={"q:v": "95"}
    )

    # Example 2: Convert file to lossless JXL
    jxl_path = transcode_image('input.png', ImageFormat.JXL)

    # Example 3: Convert to WEBP with custom parameters
    webp_bytes = transcode_image(
        input_bytes,
        ImageFormat.WEBP,
        custom_params={
            "lossless": "0",
            "q:v": "90"
        }
    )