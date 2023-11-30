import discord
from discord.ext import commands
from dotenv import load_dotenv, find_dotenv
import os
from langchain.prompts import SystemMessagePromptTemplate, PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.schema import HumanMessage
import re
from youtube_transcript_api import YouTubeTranscriptApi
from googlesearch import search

load_dotenv(find_dotenv())

YOUTUBE_PATTERN = r'(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+$'
VIDEO_ID_PATTERN = r"(?<=watch\?v=)[\w-]+|youtu\.be/([\w-]+)"
VIDEO_ID_PATTERNN = r"(?<=watch\?v=)[\w-]+|youtu.be/([\w-]+)"

pattern = r'(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+$'

TOKEN = os.environ.get('DISCORD_TOKEN')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

loader = TextLoader('data.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
retriever = Chroma.from_documents(texts, embeddings).as_retriever()
chat = ChatOpenAI(temperature=0.1, model="gpt-4")

yt_summary_prompt_template = """You are a helpful AI model named Liam AI. Summarize the following YouTube video transcript into bullet points in the following manner (with fun emojis)

1. start-overview (about the video)

2. bullet point (main content of the video)

3. end overview (overview of the video)

{context}

Please provide a summarized list of bullet points in atleast 10 - 20 bullet points.
Answer:"""

# Create a PromptTemplate object
yt_summary_prompt = PromptTemplate(template=yt_summary_prompt_template,
                                   input_variables=["context"])

prompt_template = """As an AI named Liam Evans AI, your purpose is to assist users with questions about AI automation agency. provide very most savage and funny responses with 1 or 2 emojis. If a user's question relates to any of the YouTube links in the dataset, you should suggest those links. However, if you cannot find a relevant link for a user's question, you should refrain from providing any link.

{context}

Do not recommend any youtube or external links until a user asks for it .

Please provide the most sense making responses
Answer:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)


@bot.event
async def on_message(message):
  if message.author == bot.user:
    return

  if bot.user.mentioned_in(message):
    question = message.content.replace(bot.user.mention, '').strip()
    if re.match(YOUTUBE_PATTERN, question):
      try:
        video_id_match = re.search(VIDEO_ID_PATTERN, question)
        if video_id_match:
          video_id = video_id_match.group(0)
          transcript = YouTubeTranscriptApi.get_transcript(video_id)
          transcript_text = "\n".join([
              entry['text'] for entry in transcript
          ])[:3000]  # Truncate to fit within token limit
          formatted_prompt = yt_summary_prompt.format(context=transcript_text)
          messages = [HumanMessage(content=formatted_prompt)]
          result = chat(messages)
          await message.channel.send(result.content)
      except Exception as e:
        print(f"Error occurred: {e}")
        await message.channel.send(
            "Sorry, I was unable to process your request.")
    else:
      try:
        docs = retriever.get_relevant_documents(query=question)
        formatted_prompt = system_message_prompt.format(context=docs)
        messages = [formatted_prompt, HumanMessage(content=question)]
        result = chat(messages)
        await message.channel.send(result.content)
      except Exception as e:
        print(f"Error occurred: {e}")
        await message.channel.send(
            "Sorry, I was unable to process your question.")


bot.run(TOKEN)

