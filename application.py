from __future__ import absolute_import, division, unicode_literals
import AI
import numpy as np
import os
import time
import discord


path_to_file = "data.txt"

data = AI.data(path_to_file)

#model = AI.create_model(data)
#model = AI.load_model(data)

#AI.train_model(model, data, 1)

#model = AI.output_mode(model, data)
#print(AI.generate_text(model, data, "Input text here"))

'''
token = "Discord token goes here"

client = discord.Client()

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.content.startswith('!Response '):
        msg = str(message) + AI.generate_text(model, message.content[9:]+"\n")
        await client.send_message(message.channel, msg)

@client.event
async def on_ready():
    print('Logged in as')
    print(client.user.name)
    print(client.user.id)
    print('------')

client.run(token)
'''