import cv2
import tkinter as tk
from PIL import Image, ImageTk

root = tk.Tk()
root.title("Webcam App")

label = tk.Label(root)
label.pack()

cap = cv2.VideoCapture("http://192.168.1.6:8080/video")

def update():
    ret, frame = cap.read()
    if ret:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        label.imgtk = img
        label.config(image=img)
    label.after(10, update)

update()

root.mainloop()

cap.release()