import tkinter as tk
from tkinter import messagebox

def censor_offensive_words():
    # Get input sentence from entry widget
    input_sentence = entry.get()

    # Offensive words dataset
    offensive_words_dataset = ['bevarsi', 'bewarsi', 'gandu', 'chutiya', 'lowda', 'mindri', 'magane', 'loosu', 'shata', 'tunne unnu', 'tika keisko', 'tika Densko', 'tika', 'dengbeda', 'nin amman', 'nin akkan', 'keytini', 'dengtini', 'jhatt', 'soole munde', 'dagar', 'choolu', 'huchnanmagne', 'soole magne', 'nin ajji', 'nin ajja', 'bosadimude', 'bolimaga', 'suley']

    # Split the input sentence into words
    words = input_sentence.split()

    # Initialize a variable to track if any offensive words are found
    offensive_found = False

    # Check each word in the input sentence
    for i, word in enumerate(words):
        # Check if the word is in the offensive words dataset
        if word.lower() in offensive_words_dataset:
            # Replace the offensive word with asterisks of the same length
            words[i] = '*' * len(word)
            # Set the flag to indicate that an offensive word is found
            offensive_found = True

    # Join the censored words back into a sentence
    censored_sentence = ' '.join(words)

    # If any offensive word is found, show censored sentence and a message
    if offensive_found:
        messagebox.showinfo("Censored Sentence", f"{censored_sentence}\n\nThe statement is offensive")
    else:
        # If no offensive word is found, show the original sentence and a neutral message
        messagebox.showinfo("Censored Sentence", f"{input_sentence}\n\nThe statement is neutral")

# Create the main window
root = tk.Tk()
root.title("Offensive Word Censor")

# Create and place the input field
entry = tk.Entry(root, width=50)
entry.pack(pady=10)

# Create and place the Censor button
censor_button = tk.Button(root, text="Censor", command=censor_offensive_words)
censor_button.pack(pady=5)

# Run the application
root.mainloop()