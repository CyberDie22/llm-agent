# import openai
#
# client = openai.OpenAI(
#     base_url="https://openrouter.ai/api/v1",
# )
#
# response = client.chat.completions.create(
#     model='google/gemini-2.0-flash-thinking-exp:free',
#     messages=[
#         {'role': 'user', 'content': 'Hey! Whats the derivative of 3x^4 + 3x^3 + ln(2x^5)?'}
#     ],
#     stream=True
# )
#
# for chunk in response:
#     print(chunk)

test1 = {
    'test': 'a'
}

test2 = {
    'best': 'asd'
}

test3 = {
    **test1,
    'another': 'third'
}