import tkinter as tk
from tkinter.constants import END
import csv
from code_complete import *
#from tkinter import font

class CodeCompleteGUI():
    def __init__(self, root):
        self.root = root        
        self.root.title('Code Completion Project')
        self.root.geometry('{}x{}'.format(890, 450))
        
        # Create all of the main containers
        top_frame = tk.Frame(self.root, bg='cyan', width=50, height=50, pady=3)
        center_frame = tk.Frame(self.root, width=50, height=40, padx=3, pady=3)
        
        # layout all of the main containers
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        
        top_frame.grid(row=0, sticky="ew")
        center_frame.grid(row=1, sticky="nsew")
        
        # create the widgets for the top frame
        heading = tk.Label(top_frame, text='Enter Code Snippet to Autocomplete in the textbox below')
        heading.grid(row=0, columnspan=15)
        heading.config(font=("Helvetica", 12, "bold italic"))
        
        label_input_box = tk.Label(top_frame, text='Enter code snippet:')
        label_input_box.grid(row=1, column=0)
        label_input_box.config(font=("Courier", 15))
        
        # input box widget
        input_tbox = tk.Text(top_frame, height=2, width=30)
        input_tbox.grid(row=1, column=3)
        input_tbox.config(font=("Courier", 20))
        
        # create the center widgets
        center_frame.grid_rowconfigure(0, weight=1)
        center_frame.grid_columnconfigure(0, weight=0)
        ctr_mid = tk.Frame(center_frame, width=250, height=190)
        ctr_mid.grid(row=0, column=1, sticky="nsew")
        
        output_textbox = tk.Text(center_frame, height=20, width=200, state='disabled')
        output_textbox.grid(row=0, column=1)
        output_textbox.config(font=("Courier", 20))
        
        self.button = tk.Button(top_frame, text='Auto complete', fg="red", font="Courier: 13")
        self.button.grid(row=1, column=5)
        self.button.bind('<Button-1>', lambda event, input_box=input_tbox, output_box=output_textbox: self._complete_label(event, input_box, output_box))

    def _complete_label(self, event, input_box, output_box):
        output_box.config(state="normal")
        output_box.delete("1.0", END)
                
        self.input_value = input_box.get("1.0", "end-1c")
        
        for value in self.get_predictions(self.input_value):
            output_box.insert(END, value+'\n')
        output_box.config(state="disabled")
        
    def setOutput(self, results):
        self.predicted_results = results
        
    def get_predictions(self, input_data):
        return predict(input_data)
    
    def get_input(self):
        return self.input_value
    
    def mainloop(self):
        self.root.mainloop()
    
if __name__=='__main__':
    root = tk.Tk()
    cc = CodeCompleteGUI(root)
    
    cc.mainloop()