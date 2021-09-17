# bot.py
import os
import sys
import random
import numpy as np
import torch
import discord
import IPython.display as ipd
from simpletransformers.language_generation import LanguageGenerationModel
from discord.ext import commands,tasks
from dotenv import load_dotenv
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from audio_processing import griffin_lim
from text import text_to_sequence
from denoiser import Denoiser
from pydub import AudioSegment

load_dotenv()


DISCORD_TOKEN = '' #Put your token Here!
PATH_TO_MODEL = "models/best_modelbuddy" #Put the path to your model here!
DEDICATED_CHANNEL_NAME = 'okbuddybot' #Put the name of the channel in your server where you want the bot to chat in here!

#Make false if you dont want to use ur gpu.
USE_CUDA = True
#CUDA cuts generation time in half. Make sure you follow github page if you want to set this to True.

'''Experimental Memory Feature! Tacks the previous responses into one big string to give the bot more context.
Might cause processing times to go up'''
EXPERIMENTAL_MEMORY = False 
EXPERIMENTAL_MEMORY_LENGTH = 500 #Max char length before memory resets. Higher numbers can heavily affect model inference times. Default 500

'''Ignore all of this stuff below VVVV'''

tacotron2_pretrained_model = "MLPTTS"
waveglow_pretrained_model = "waveglow.pt"
thisdict = {}
for line in reversed((open('merged.dict.txt', "r").read()).splitlines()):
    thisdict[(line.split(" ",1))[0]] = (line.split(" ",1))[1].strip()
def ARPA(text):
    out = ''
    for word_ in text.split(" "):
        word=word_; end_chars = ''
        while any(elem in word for elem in r"!?,.;") and len(word) > 1:
            if word[-1] == '!': end_chars = '!' + end_chars; word = word[:-1]
            if word[-1] == '?': end_chars = '?' + end_chars; word = word[:-1]
            if word[-1] == ',': end_chars = ',' + end_chars; word = word[:-1]
            if word[-1] == '.': end_chars = '.' + end_chars; word = word[:-1]
            if word[-1] == ';': end_chars = ';' + end_chars; word = word[:-1]
            else: break
        try: word_arpa = thisdict[word.upper()]
        except: word_arpa = ''
        if len(word_arpa)!=0: word = "{" + str(word_arpa) + "}"
        out = (out + " " + word + end_chars).strip()
    if out[-1] != ";": out = out + ";"
    return out

# initialize Tacotron2 with the pretrained model
hparams = create_hparams()
hparams.sampling_rate = 22050 # Don't change this
hparams.max_decoder_steps = 1000 # How long the audio will be before it cuts off (1000 is about 11 seconds)
hparams.gate_threshold = 0.1 # Model must be 90% sure the clip is over before ending generation (the higher this number is, the more likely that the AI will keep generating until it reaches the Max Decoder Steps)
model = Tacotron2(hparams)
model.load_state_dict(torch.load(tacotron2_pretrained_model)['state_dict'])
_ = model.cuda().eval().half()
# Load WaveGlow
waveglow = torch.load(waveglow_pretrained_model)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)

def tts(text, location):
        print("Starting gen!")
        sigma = 0.8
        denoise_strength = 0.324
        raw_input = False #MAKE TRUE disables automatic ARPAbet conversion, useful for inputting your own ARPAbet pronounciations or just for testing

        for i in text.split("\n"):
            if len(i) < 1: continue;
            print(i)
            if raw_input:
                if i[-1] != ";": i=i+";" 
            else: i = ARPA(i)
            print(i)
            with torch.no_grad(): # save VRAM by not including gradients
                sequence = np.array(text_to_sequence(i, ['english_cleaners']))[None, :]
                sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
                mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
                audio_denoised = waveglow.infer(mel_outputs_postnet, sigma=sigma)#; print(""); ipd.display(ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate))
                audio = ipd.Audio(audio_denoised.cpu().numpy(), rate=hparams.sampling_rate)
                audio = AudioSegment(audio.data, frame_rate=22050, sample_width=2, channels=1)
                audio.export(location, format="mp3", bitrate="64k")

memory = ''
print('IMPORTS FINISHED')

def genCleanMessage(optPrompt):
        global memory
        memory = ''
        formattedPrompt = '<|sor|>' + optPrompt + '<|eor|><|sor|>'
        formattedPrompt = '<|sor|>'
        #if (len(memory) == 0):
        #    formattedPrompt = '<|soss|><|sot|>' + optPrompt + '<|eot|><|sor|>'
        memory += formattedPrompt
        print('\nPROMPT:' + formattedPrompt + '\n')
        print('\nMEMORY:' + memory + '\n')
        #if formattedPrompt == '<|sor|>!t<|eor|><|sor|>':
        #    memory = '<|sols|><|sot|>I'
        model = LanguageGenerationModel("gpt2", PATH_TO_MODEL, use_cuda=USE_CUDA)
        text_generation_parameters = {
            #CHANGE THIS BACK TO 50
			'max_length': 20,
			'num_return_sequences': 1,
			'prompt': memory,
			'temperature': 0.8, #0.8
			'top_k': 40,
			#'truncate': '<|eo',
	}
        output_list = model.generate(prompt=memory, args=text_generation_parameters)
        response = output_list[0]
        response = response.replace(memory, '')
        #memory += ' '
        #for element in response:
        #    if element != '!':
        #        memory += element
        #memory += ' '
        i = 0
        cleanStr = ''
        print(response)
        for element in response:
            if element == '<':
                i = 1
            if i == 0 and element != '!':
                cleanStr += element
            if element == '>':
                i = 0
        if not cleanStr:
            cleanStr = 'Idk how to respond to that lol. I broke.'
        memory += cleanStr + "<|eor|>"
        cleanStr = cleanStr.replace("Draven", "Cameron")
        cleanStr = cleanStr.replace("draven", "Cameron")
        return cleanStr




intents = discord.Intents().all()
client = discord.Client(intents=intents)
bot = commands.Bot(command_prefix='!',intents=intents)
@bot.event
async def on_ready():
    print('Bot has connected to Discord!')

@bot.command(name='join', help='Tells the bot to join the voice channel')
async def join(ctx):
    print('join command')
    if not ctx.message.author.voice:
        await ctx.send("{} is not connected to a voice channel".format(ctx.message.author.name))
        return
    else:
        channel = ctx.message.author.voice.channel
    await channel.connect()
    server = ctx.message.guild
    voice_channel = server.voice_client
    while True:
            async with ctx.typing():
        #filename = await YTDLSource.from_url(url, loop=bot.loop)
        #voice_channel.play(discord.FFmpegPCMAudio(executable="ffmpeg.exe", source='download_17.wav'))
                responseTest = genCleanMessage('Mashalla.')
            await ctx.send(responseTest)
            async with ctx.typing():
                    tts(responseTest, 'temp.wav')
                    #tts(" ", 'temp.wav')
            await ctx.send('thinking...')
            async with ctx.typing():
                    voice_channel.play(discord.FFmpegPCMAudio(executable="ffmpeg.exe", source='temp.wav'))
            await ctx.send('done!')

@bot.command(name='leave', help='To make the bot leave the voice channel')
async def leave(ctx):
    print('leave command')
    voice_client = ctx.message.guild.voice_client
    if voice_client.is_connected():
        await voice_client.disconnect()
    else:
        await ctx.send("The bot is not connected to a voice channel.")
'''
@bot.event
async def on_message(message):
    print('GEN MESSAGE FOUND')
    if message.author == bot.user:
        return
    if str(message.channel) == DEDICATED_CHANNEL_NAME:
        pass
    if message.content == '!r' and str(message.channel) == DEDICATED_CHANNEL_NAME:
        global memory
        memory = ''
        await message.channel.send('```convo reset```')
        print(memory)
        return
    if str(message.channel) == DEDICATED_CHANNEL_NAME:
        if (len(memory) > EXPERIMENTAL_MEMORY_LENGTH) or (not EXPERIMENTAL_MEMORY):
            memory = ''
        prompt = message.content
        genMessage = genCleanMessage(prompt)
        await message.channel.send(genMessage)
    elif message.content == 'raise-exception':
        raise discord.DiscordException
'''


if __name__ == "__main__" :
    print('_main_ started')
    bot.run(DISCORD_TOKEN)
