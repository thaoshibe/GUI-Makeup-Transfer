
from magican import Magican
import PySimpleGUI as sg
import os.path
import os, cv2, glob, time
from utils import testcam
from imageio import imread, imsave
import tensorflow as tf
import numpy as np
from PIL import Image
import argparse

version = '23 July 2020'
sg.theme('Material1')
# sg.theme('SystemDefault')

img_size = 256
refer_path = './makeup'
list_refer = glob.glob(os.path.join(refer_path, '*.*'))
refer_button_size = 100
border_size =1
border_size_2=0
fixed_style=20
cap = None
parser = argparse.ArgumentParser()
parser.add_argument('--start_index', default=0)
args = parser.parse_args()
start_index = int(args.start_index)

magican = Magican()

# --------------------------------- The GUI ---------------------------------

fist_panel = [[sg.Image(filename='./static/vinai_vision.png')],
            [sg.Image(filename='', key='-FRAME-')],
            [sg.Image(filename='./static/shibe_thank.png')]
            ]

images_col = [
            [sg.Image(filename='./static/makeup_with_ai2.png', key='-MAKEUP_TITLE-')],
            # [sg.Image(filename='./static/1.png')],
            [sg.ReadFormButton('0', button_color=sg.TRANSPARENT_BUTTON, image_data= '', image_subsample=2, border_width=border_size),
                sg.ReadFormButton('1', button_color=sg.TRANSPARENT_BUTTON, image_data= '', image_subsample=2, border_width=border_size),
                sg.ReadFormButton('2', button_color=sg.TRANSPARENT_BUTTON, image_data= '', image_subsample=2, border_width=border_size),
                sg.ReadFormButton('3', button_color=sg.TRANSPARENT_BUTTON, image_data= '', image_subsample=2, border_width=border_size),
                sg.ReadFormButton('4', button_color=sg.TRANSPARENT_BUTTON, image_data= '', image_subsample=2, border_width=border_size),
                sg.ReadFormButton('5', button_color=sg.TRANSPARENT_BUTTON, image_data= '', image_subsample=2, border_width=border_size),
                sg.ReadFormButton('6', button_color=sg.TRANSPARENT_BUTTON, image_data= '', image_subsample=2, border_width=border_size),
                sg.ReadFormButton('7', button_color=sg.TRANSPARENT_BUTTON, image_data= '', image_subsample=2, border_width=border_size),
                sg.ReadFormButton('8', button_color=sg.TRANSPARENT_BUTTON, image_data= '', image_subsample=2, border_width=border_size),
                sg.ReadFormButton('9', button_color=sg.TRANSPARENT_BUTTON, image_data= '', image_subsample=2, border_width=border_size)],
            # [sg.Image(filename='./static/2.png')],
            [sg.ReadFormButton('10', button_color=sg.TRANSPARENT_BUTTON, image_data= '', image_subsample=2, border_width=border_size),
                sg.ReadFormButton('11', button_color=sg.TRANSPARENT_BUTTON, image_data= '', image_subsample=2, border_width=border_size),
                sg.ReadFormButton('12', button_color=sg.TRANSPARENT_BUTTON, image_data= '', image_subsample=2, border_width=border_size),
                sg.ReadFormButton('13', button_color=sg.TRANSPARENT_BUTTON, image_data= '', image_subsample=2, border_width=border_size),
                sg.ReadFormButton('14', button_color=sg.TRANSPARENT_BUTTON, image_data= '', image_subsample=2, border_width=border_size),
                sg.ReadFormButton('15', button_color=sg.TRANSPARENT_BUTTON, image_data= '', image_subsample=2, border_width=border_size),
                sg.ReadFormButton('16', button_color=sg.TRANSPARENT_BUTTON, image_data= '', image_subsample=2, border_width=border_size),
                sg.ReadFormButton('17', button_color=sg.TRANSPARENT_BUTTON, image_data= '', image_subsample=2, border_width=border_size),
                sg.ReadFormButton('18', button_color=sg.TRANSPARENT_BUTTON, image_data= '', image_subsample=2, border_width=border_size),
                sg.ReadFormButton('19', button_color=sg.TRANSPARENT_BUTTON, image_data= '', image_subsample=2, border_width=border_size)],
            [sg.Image(filename='./static/2.png')],
            [sg.Image(filename='', key='-REFER-'), sg.Image(filename='', key='-IN-'), sg.Image(filename='', key='-OUT-')],
            [sg.Button('Capture', button_color=(sg.theme_background_color(), sg.theme_background_color()), key='-Capture-', image_data='', border_width=5)],
            [sg.ReadFormButton('20', button_color=(sg.theme_background_color(), sg.theme_background_color()), image_data= '', image_subsample=2, border_width=border_size_2),
                sg.ReadFormButton('21', button_color=(sg.theme_background_color(), sg.theme_background_color()), image_data= '', image_subsample=2, border_width=border_size_2),
                sg.ReadFormButton('22', button_color=(sg.theme_background_color(), sg.theme_background_color()), image_data= '', image_subsample=2, border_width=border_size_2),
                sg.ReadFormButton('23', button_color=(sg.theme_background_color(), sg.theme_background_color()), image_data= '', image_subsample=2, border_width=border_size_2),
                sg.ReadFormButton('24', button_color=(sg.theme_background_color(), sg.theme_background_color()), image_data= '', image_subsample=2, border_width=border_size_2),
                sg.ReadFormButton('25', button_color=(sg.theme_background_color(), sg.theme_background_color()), image_data= '', image_subsample=2, border_width=border_size_2),
                sg.ReadFormButton('26', button_color=(sg.theme_background_color(), sg.theme_background_color()), image_data= '', image_subsample=2, border_width=border_size_2),
                sg.ReadFormButton('27', button_color=(sg.theme_background_color(), sg.theme_background_color()), image_data= '', image_subsample=2, border_width=border_size_2),
                sg.ReadFormButton('28', button_color=(sg.theme_background_color(), sg.theme_background_color()), image_data= '', image_subsample=2, border_width=border_size_2),
                sg.ReadFormButton('29', button_color=(sg.theme_background_color(), sg.theme_background_color()), image_data= '', image_subsample=2, border_width=border_size_2)]
            ]
# ----- Full layout -----
layout = [[sg.Column(fist_panel), sg.VSeperator(), sg.Column(images_col)]]
# ----- Make the window -----
window = sg.Window('─── ･ ｡ﾟ☆: *.☽ .* :☆ﾟ. ───', layout, resizable=True, grab_anywhere=True, finalize=True, location=(0, 0))
# window.Maximize()
sg.PopupQuickMessage('ĐANG TẢI CHƯƠNG TRÌNH... ೕ(˃ᴗ˂๑)', location=(1000, 500), auto_close_duration=5)
# graph.DrawImage(filename=list_refer[1], location=(0, 0))
# ----- Run the Event Loop -----
button_capture = cv2.imread('./static/button_capture.png')
# button_capture = cv2.resize(button_capture, (1152, 20))
window['-Capture-'].update(image_data=cv2.imencode('.png', button_capture)[1].tobytes())

for i in range(0, fixed_style):
    style_img = cv2.resize(cv2.imread(os.path.join(refer_path, '{}.png'.format(i))), (refer_button_size, refer_button_size))
    window.FindElement(str(i)).update(image_data=cv2.imencode('.png', style_img)[1].tobytes())

cap = cv2.VideoCapture(testcam(start_index)) if not cap else cap
# cap = cv2.VideoCapture(1)
# sg.PopupQuickMessage('Setting up camera', auto_close_duration=2, location=(500, 500))

add_custom_refer = 20

while True:
    ret, frame = cap.read()
    # window['-FRAME-'].update(data=cv2.imencode('.png', frame)[1].tobytes())
    event, values = window.read(timeout=0.01)
    list_button = (str(i) for i in range(0, len(list_refer)))
    if event in list_button:
        magican.new_refer(os.path.join(refer_path, '{}.png'.format(event)))
        window['-REFER-'].update(data=cv2.imencode('.png', cv2.cvtColor(magican.refer_img, cv2.COLOR_RGB2BGR))[1].tobytes())
    if event in ('-EXIT-', None):
        window.close()
        break
    try:
        cropped = magican.algin_face(frame)
        window['-FRAME-'].update(data=cv2.imencode('.png', magican.frame)[1].tobytes())
        window['-REFER-'].update(data=cv2.imencode('.png', cv2.cvtColor(magican.refer_img, cv2.COLOR_RGB2BGR))[1].tobytes())
        window['-IN-'].update(data=cv2.imencode('.png', cropped)[1].tobytes())
        window['-OUT-'].update(data=cv2.imencode('.png', magican.makeup())[1].tobytes())
    except:
        window['-FRAME-'].update(data=cv2.imencode('.png', frame)[1].tobytes())

    if event == '-Capture-':
        face = magican.algin_face(frame)
        cv2.imwrite(os.path.join(refer_path, '{}.png'.format(add_custom_refer)), face)
        list_refer = glob.glob(os.path.join(refer_path, '*.*'))
        face = cv2.resize(cv2.imread('./makeup/{}.png'.format(add_custom_refer)), (refer_button_size, refer_button_size))
        window.FindElement('{}'.format(add_custom_refer)).update(image_data=cv2.imencode('.png', face)[1].tobytes())
        if add_custom_refer < 29:
            add_custom_refer+=1
        else:
            add_custom_refer=20
# ----- Exit program -----
window.close()