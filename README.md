## Using tensorflow to make speech-capable discord bots
This repository has what you need to create a model, train it, and generate text using it.
This repository also allows you to connect the tensorflow model directly to discord using a token that you must generate for your bot.
### Setting up the project
This project requires `python 3.6` for compatibility with tensorflow.  
  
Install dependencies with:  
`pip install -r requirements.txt`
### Setting up a discord bot
You can use this link to get started with making a discord bot:  
https://discordapp.com/developers/applications  
  
Other tutorials are available to demonstrate how a bot is created for discord. Follow those tutorials until you are at the point where you have a token for your discord bot. This token is all you need to connect application.py to discord.
### Setting up tensorflow data
All that is required to load data is a file that has the data in it, and the lines that follow:
```
path_to_file = "data.txt"
data = AI.data(path_to_file)
```
You must use a relative or absolute path to your file. There is no strict rule to how you format your data, but you should know that the bot will attempt to replicate the text in the file. Leaving chat metadata in files may not have desired results.
### Setting up a tensorflow model
One of two methods can be used to create a model. Starting from the beginning, you can create a new model if you have not already trained one:  
`model = AI.create_model(data)`  
If you have already trained a model, you can load it like so:  
`model = AI.load_model(data)`  
### Training your model
Training a model is as simple as these two lines (which could be condensed to one):
```
number_of_epochs = 1
AI.train_model(model, data, number_of_epochs)
```
The number of epochs is the number of times the model will train on your data. For chat logs consisting of ~1000 or less messages, the number of epochs may end up numbering in the dozens. For chat logs nearing tens or hundreds of thousands of messages, the needed number of epochs will be closer to 5-10.
### Generating text from a model
A model must be converted to output mode first, and then text can be generated as shown:
```
model = AI.output_mode(model, data)
print(AI.generate_text(model, data, "Input text here"))
```
### GPU Support
This project has GPU support, but you must also go through these extra steps in order for GPU support to work:
https://www.tensorflow.org/install/gpu
