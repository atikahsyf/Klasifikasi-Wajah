import tkinter
import tkinter.ttk as ttk
from tkinter import messagebox
#from detect_gender_webcam import start
from train import training, start
from PIL import ImageTk, Image


def call():
    res = messagebox.askquestion('Keluar',
                                 'Do you really want to exit?')

    if res == 'yes':
        main_window.destroy()


def processing():
    training()
    canvas = tkinter.Canvas(main_window, width=525, height=340)
    canvas.place(x=402, y=70)
    cvstext = canvas.create_text(200, 165, text='', font=(
        'Helvetica 12 bold'), anchor=tkinter.NW)
    our_text = "Training Selesai"
    delta = 100
    delay = 0
    canvas.itemconfigure(cvstext, text=our_text)
    for x in range(len(our_text)+1):
        s = our_text[:x]
        def new_text(s=s): return canvas.itemconfigure(cvstext, text=s)
        canvas.after(delay, new_text)
        delay += delta


main_window = tkinter.Tk()
main_window.title('Gender Detection')
main_window.iconbitmap('assets/windowIcon.ico')
main_window.geometry("961x541")


backg = ImageTk.PhotoImage(file='assets/Frame1.png')
trainBttn = ImageTk.PhotoImage(file='assets/trainBttn.png')
startBttn = ImageTk.PhotoImage(file='assets/startBttn.png')
stopBttn = ImageTk.PhotoImage(file='assets/stopBttn.png')


label = tkinter.Label(
    main_window,
    image=backg
)
label.place(x=0, y=0)


border = tkinter.LabelFrame(main_window, bd=6, bg="white")
border.pack()

tombolMulai = ttk.Button(
    main_window, image=startBttn, command=start)
tombolKeluar = ttk.Button(
    main_window, image=stopBttn, command=call)
tombolTrain = ttk.Button(
    main_window, image=trainBttn, command=processing)


tombolTrain.place(x=420, y=450)
tombolMulai.place(x=600, y=450)
tombolKeluar.place(x=780, y=450)
main_window.mainloop()
