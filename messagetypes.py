from typing import TypedDict, Union, Literal, List, NotRequired

class ImageUrl(TypedDict):
    url: str
    detail: NotRequired[str]

class ImageContentPart(TypedDict):
    type: Literal['image']
    image_url: ImageUrl

class TextContentPart(TypedDict):
    type: Literal['text']
    text: str

class AudioInputData(TypedDict):
    data: str  # Base64 encoded audio data
    format: Literal['wav', 'mp3']

class AudioContentPart(TypedDict):
    type: Literal['input_audio']
    input_audio: AudioInputData

class RefusalContentPart(TypedDict):
    type: Literal['refusal']
    refusal: str

class AudioResponse(TypedDict):
    id: str

class FunctionCall(TypedDict):
    name: str
    arguments: str

class ToolCall(TypedDict):
    id: str
    type: Literal['function']
    function: FunctionCall

AnyContentPart = Union[TextContentPart, ImageContentPart, AudioContentPart]

# Message type definitions
class SystemMessage(TypedDict):
    role: Literal['system']
    content: Union[str, List[TextContentPart]]
    name: NotRequired[str]

class UserMessage(TypedDict):
    role: Literal['user']
    content: Union[str, List[Union[TextContentPart, ImageContentPart, AudioContentPart]]]
    name: NotRequired[str]

class AssistantMessage(TypedDict):
    role: Literal['assistant']
    content: NotRequired[Union[str, List[Union[TextContentPart, RefusalContentPart]]]]
    name: NotRequired[str]
    refusal: NotRequired[str]
    audio: NotRequired[AudioResponse]
    tool_calls: NotRequired[List[ToolCall]]

class ToolMessage(TypedDict):
    role: Literal['tool']
    content: Union[str, List[Union[TextContentPart]]]
    tool_call_id: str

# Combined message type for use in lists/arrays
Message = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]