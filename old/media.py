import ffmpeg
import io
import os
import tempfile
from typing import Optional, Union
from pathlib import Path


def ffmpeg_convert(
        file_obj: io.IOBase,
        output_format: str,
        output_path: Optional[Union[str, Path]] = None
) -> Optional[io.BytesIO]:
    """
    Convert a file-like object to the specified format using FFMPEG.

    Args:
        file_obj: File-like object containing the input media
        output_format: Desired output format (e.g., 'mp4', 'webm', 'mp3', 'jpg', 'png')
        output_path: Optional path to write output directly to file

    Returns:
        io.BytesIO object containing the converted media if output_path is None,
        None if output_path is provided

    Raises:
        ValueError: If file_obj is closed or output_format is empty
        ffmpeg.Error: If conversion fails
        OSError: If file operations fail
    """
    if file_obj.closed:
        raise ValueError("File object is closed")
    if not output_format:
        raise ValueError("Output format must be specified")

    output_format = output_format.strip().lower()
    temp_input = None
    temp_output = None

    try:
        # Check if input is a real file
        try:
            input_path = os.path.realpath(file_obj.name)
            if not os.path.exists(input_path):
                raise AttributeError
        except (AttributeError, TypeError):
            # For memory-based files, create a temp file
            temp_input = tempfile.NamedTemporaryFile(delete=False)
            file_obj.seek(0)
            temp_input.write(file_obj.read())
            temp_input.flush()
            input_path = temp_input.name

        # Setup output path
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            final_output = str(output_path)
        else:
            temp_output = tempfile.NamedTemporaryFile(suffix=f'.{output_format}', delete=False)
            temp_output.close()
            final_output = temp_output.name

        # Run the conversion
        stream = ffmpeg.input(input_path)
        stream = ffmpeg.output(stream, final_output)
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, overwrite_output=True)

        if output_path:
            return None
        else:
            output_buffer = io.BytesIO()
            with open(final_output, 'rb') as f:
                output_buffer.write(f.read())
            output_buffer.seek(0)
            return output_buffer

    finally:
        # Clean up temporary files
        if temp_input:
            try:
                os.unlink(temp_input.name)
            except OSError:
                pass

        if temp_output:
            try:
                os.unlink(temp_output.name)
            except OSError:
                pass
