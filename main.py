import discord
from discord.ext import commands
import mtcnn
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
import PIL
import os
from urllib import request
import numpy as np
import cv2

TOKEN = 'token goes here'
bot = commands.Bot(command_prefix='$')

@bot.event
async def on_ready():
    
    # Confirm bot is online in console
    print('We have logged in as {0.user}'.format(bot))


@bot.command()
async def match(ctx):

    # -- INITIALIZATION COMMUNICATION -- #

    # Send message to user
    await ctx.send(f'Finding your celebrity match... {ctx.author.mention}')

    # -- IMAGE LOADING -- #
    
    # Create image file from user avatar
    filename = 'avatar.jpg'
    await ctx.author.avatar_url.save(filename)

    # Read the file in color and convert it to an image
    img = cv2.imread('./avatar.jpg', cv2.IMREAD_COLOR)
    
    # -- FACE DETECTION -- #

    # Initialize the mtcnn detector
    detector = MTCNN()

    # Extraction parameters
    target_size = (224, 224)
    border_rel = 0

    # Detect faces in the image
    detections = detector.detect_faces(img)

    # Create box around the face
    x1, y1, width, height = detections[0]['box']
    dw = round(width * border_rel)
    dh = round(height * border_rel)
    x2, y2 = x1 + width + dw, y1 + height + dh
    face = img[y1:y2, x1:x2]

    # Resize the pixels to the model size
    face = PIL.Image.fromarray(face)
    face = face.resize((224, 224))
    face = np.asarray(face)

    # -- PREPROCESSING -- #

    # Convert to float32
    face_pp = face.astype('float32')
    face_pp = np.expand_dims(face_pp, axis = 0)

    # Normalize pixel values
    face_pp = preprocess_input(face_pp, version = 2)

    # -- PREDICTION -- #

    # Create the resnet50 model
    model = VGGFace(model='resnet50')

    # Predict the face with the input
    prediction = model.predict(face_pp)

    # -- EXTRACT AND DISPLAY RESULTS -- #

    # Convert prediction to names with probability
    results = decode_predictions(prediction)

    # Send avatar image
    await ctx.send('Here are your results:', file=discord.File(fp=filename))
    
    # Display prediction results
    for result in results[0]:
        
        # String formatting
        f_result = result[0][2:]
        f_result = f_result[:-1]
        
        # Send message
        msg = ('%s: %.3f%%' % (f_result, result[1]*100))
        await ctx.send(msg) 

    # -- CLEAN UP -- #

    # Remove avatar file
    os.remove('./avatar.jpg')
    

bot.run(TOKEN)
