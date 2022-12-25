
#from get_text import get_text

from tkinter import *
class MyWindow:
    def __init__(self, win):
        self.lbl1=Label(win, text='User Name')
        self.lbl2=Label(win, text='Password')
        #self.lbl3=Label(win, text='Result')
        self.t1=Entry(bd=3)
        self.t2=Entry(show= '*')
        self.t3=Entry()
        self.btn1 = Button(win, text='Login')
        #self.btn2=Button(win, text='Subtract')
        self.lbl1.place(x=100, y=50)
        self.t1.place(x=200, y=50)
        self.lbl2.place(x=100, y=100)
        self.t2.place(x=200, y=100)
        self.b1=Button(win, text='Login', command=self.add)
        #self.b2=Button(win, text='Subtract')
        #self.b2.bind('<Button-1>', self.sub)
        self.b1.place(x=200, y=150)
        #self.b2.place(x=200, y=150)
        #self.lbl3.place(x=100, y=200)
        #self.t3.place(x=200, y=200)
    def add(self):
        self.t3.delete(0, 'end')
        num1=self.t1.get()
        num2=self.t2.get()
        result=num1+num2
        if num1=='admin' and num2=='123456':
            print('Login sucsess')
            window.destroy()
            import get_text
        
window=Tk()
mywin=MyWindow(window)
window.title('Cyberbulling Detection')
window.geometry("400x300+10+10")
window.mainloop()
