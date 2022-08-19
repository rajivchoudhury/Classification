from cortex import Cortex
import asyncio
import pygame
import pandas as pd
from eeglib.helpers import CSVHelper
from sklearn.neural_network import MLPClassifier
from time import time

pygame.init()
res = (1260,720)
screen = pygame.display.set_mode(res)

white_color = (255,255,255)
black_color = (0, 0, 0)
red_color = (255, 0, 0)
color_light = (170,170,170)
smallfont = pygame.font.SysFont('Corbel',35)

# First parameter is Client ID from cortex
# Second parameter is Secret ID
cortex = Cortex("xxx", "xxx", "wss://localhost:6868")

async def connect():
    response = await cortex.queryHeadsets()
    if response == []:
        print("Turn on the headset to proceed")
    elif len(response) == 1:
        print("Headset's " + response[0]["id"] + " status is: " + response[0]["status"])
    else:
        s = "The devices available are\n"
        for device in response:
            s += device["id"] + "\n"
        print(s)
        s = input("Enter the device you want to connect: ")
        print(await cortex.actionOnHeadset("connect", s))

async def main():

    trainText = smallfont.render('Training Model...', True, black_color)

    screen.fill(white_color)
    screen.blit(trainText , (550, 310))
    pygame.display.update()
    df = pd.read_csv('dataset.csv', header=None)

    print("Training Data...")
    x_train = df.iloc[:, 0:5]
    y_train = df.iloc[:, 5]
    y_train_o = pd.get_dummies(y_train)

    x_mean = x_train.mean() # find the mean values of each feature of the train data
    x_std = x_train.std() # find the std. dev. of each feature of the train data
    x_train = (x_train - x_mean) / x_std # z-score normalize the train data


    # Classification
    clf = MLPClassifier(
        hidden_layer_sizes=(1000,100,100), alpha=1, activation='relu', solver='adam', 
        batch_size=60, max_iter=10000, shuffle=True, learning_rate='adaptive',
        learning_rate_init=0.001, power_t=0.5, 
    )
    # clf = SVC(kernel="linear", C=0.025)
    clf.fit(x_train, y_train)

    print('Training DONE...')
    trainText = smallfont.render('Training Done...', True, black_color)

    screen.fill(white_color)
    screen.blit(trainText , (550, 310))
    pygame.display.update()

    await cortex.requestAccess()
    await cortex.getAuthorizationToken()

    await connect()

    await asyncio.sleep(3)

    await cortex.createSession("open")
    await cortex.updateSession()

    text1 = smallfont.render('Neutral' , True , black_color)
    text2 = smallfont.render('RotateClockwise' , True , black_color)
    text3 = smallfont.render('Disappear' , True , black_color)
    text4 = smallfont.render('Pull' , True , black_color)
    image = pygame.image.load('block.png')

    result = await cortex.subscribe(['eeg'])
    print(result)
    while True:
        f = []
        s_time = time()

        while len(f) <= 128:
            data = await cortex.receiveData()

            if data != []:
                eeg = data['eeg'][2:7] 
                # print(eeg)
                f.append(eeg)

        df = pd.DataFrame(f)
        df.to_csv('helper.csv', encoding='utf-8-sig', index=False)

        helper = CSVHelper('helper.csv')

        for eeg in helper:
            temp = eeg.PFD()

        x_test = []
        x_test.append(temp)
        x_test = pd.DataFrame(x_test)
        x_test = (x_test - x_mean) / x_std # z-score normalize the test data
        pred = clf.predict(x_test)
        print(pred)
        print("Time = ", (time() - s_time))

        if 'Idle' in pred:
            text1 = smallfont.render('Neutral' , True , red_color)
            text2 = smallfont.render('RotateClockwise' , True , black_color)
            text3 = smallfont.render('Disappear' , True , black_color)
            text4 = smallfont.render('Pull' , True , black_color)

        if 'RotateLeft' in pred:
            text1 = smallfont.render('Neutral' , True , black_color)
            text2 = smallfont.render('RotateClockwise' , True , red_color)
            text3 = smallfont.render('Disappear' , True , black_color)
            text4 = smallfont.render('Pull' , True , black_color)

        if 'Pull' in pred:
            text1 = smallfont.render('Neutral' , True , black_color)
            text2 = smallfont.render('RotateClockwise' , True , black_color)
            text3 = smallfont.render('Disappear' , True , black_color)
            text4 = smallfont.render('Pull' , True , red_color)

        if 'Disappear' in pred:
            text1 = smallfont.render('Neutral' , True , black_color)
            text2 = smallfont.render('RotateClockwise' , True , black_color)
            text3 = smallfont.render('Disappear' , True , red_color)
            text4 = smallfont.render('Pull' , True , black_color)
        
        for ev in pygame.event.get():

            if ev.type == pygame.QUIT:
                pygame.quit()
                exit(0)

        screen.fill(white_color)
        screen.blit(text1 , (300, 250))
        screen.blit(text2 , (10,300))
        screen.blit(text3 , (450,300))
        screen.blit(text4 , (300, 350))

        screen.blit(image, (690, 150))
        pygame.display.update()

asyncio.get_event_loop().run_until_complete(main())