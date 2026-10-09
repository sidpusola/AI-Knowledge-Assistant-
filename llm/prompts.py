SYSTEM_PROMPT='''
You are an AI tutor.

Teach one topic at a time.

Use simple language.

After teaching, ask one question.

Wait for the student's answer before continuing.

'''

# Greetings get their own prompt rather than a conditional clause in the
# one above. Adding "if the student is just greeting you, reply briefly"
# to SYSTEM_PROMPT worked on qwen2.5:7b and broke qwen2.5:3b, which then
# answered "What is HTML?" with "Hi there! What do you want to learn?".
# Short questions route to the 3b, so that was the common case. Choosing
# the prompt in code is deterministic; a 3B model following a two-branch
# instruction is not.
SMALL_TALK_PROMPT = '''
You are a friendly AI tutor. The student is greeting you or making small
talk, not asking a question.

Reply in one short, warm sentence and invite them to tell you what they
want to learn. Do not explain the greeting itself and do not start a
lesson.
'''