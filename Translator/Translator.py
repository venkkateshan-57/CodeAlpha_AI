import tkinter as tk
from tkinter import ttk, messagebox
from deep_translator import GoogleTranslator
import pyperclip


class TranslationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Language Translation Tool")
        self.root.geometry("800x600")
        self.root.config(bg="#f0f0f0")

        # Manual language mapping with full names
        self.languages = {
            'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de',
            'Italian': 'it', 'Portuguese': 'pt', 'Russian': 'ru', 'Japanese': 'ja',
            'Korean': 'ko', 'Chinese (Simplified)': 'zh-CN', 'Chinese (Traditional)': 'zh-TW',
            'Arabic': 'ar', 'Hindi': 'hi', 'Bengali': 'bn', 'Turkish': 'tr',
            'Dutch': 'nl', 'Greek': 'el', 'Hebrew': 'he', 'Thai': 'th',
            'Vietnamese': 'vi', 'Indonesian': 'id', 'Malay': 'ms', 'Filipino': 'tl',
            'Polish': 'pl', 'Ukrainian': 'uk', 'Czech': 'cs', 'Swedish': 'sv',
            'Danish': 'da', 'Finnish': 'fi', 'Norwegian': 'no', 'Romanian': 'ro',
            'Hungarian': 'hu', 'Slovak': 'sk', 'Bulgarian': 'bg', 'Croatian': 'hr',
            'Serbian': 'sr', 'Slovenian': 'sl', 'Lithuanian': 'lt', 'Latvian': 'lv',
            'Estonian': 'et', 'Icelandic': 'is', 'Irish': 'ga', 'Welsh': 'cy',
            'Catalan': 'ca', 'Basque': 'eu', 'Galician': 'gl', 'Persian': 'fa',
            'Urdu': 'ur', 'Punjabi': 'pa', 'Tamil': 'ta', 'Telugu': 'te',
            'Malayalam': 'ml', 'Kannada': 'kn', 'Gujarati': 'gu', 'Marathi': 'mr',
            'Nepali': 'ne', 'Sinhala': 'si', 'Khmer': 'km', 'Lao': 'lo',
            'Burmese': 'my', 'Georgian': 'ka', 'Armenian': 'hy', 'Yiddish': 'yi',
            'Swahili': 'sw', 'Afrikaans': 'af', 'Zulu': 'zu', 'Xhosa': 'xh',
            'Amharic': 'am', 'Somali': 'so', 'Hausa': 'ha', 'Yoruba': 'yo',
            'Igbo': 'ig', 'Albanian': 'sq', 'Macedonian': 'mk', 'Bosnian': 'bs',
            'Azerbaijani': 'az', 'Kazakh': 'kk', 'Uzbek': 'uz', 'Mongolian': 'mn'
        }

        self.setup_ui()

    def setup_ui(self):
        # Title
        title = tk.Label(self.root, text="Language Translation Tool",
                         font=("Arial", 20, "bold"), bg="#f0f0f0", fg="#333")
        title.pack(pady=20)

        # Language selection frame
        lang_frame = tk.Frame(self.root, bg="#f0f0f0")
        lang_frame.pack(pady=10)

        # Source language
        tk.Label(lang_frame, text="From:", font=("Arial", 12),
                 bg="#f0f0f0").grid(row=0, column=0, padx=5)
        self.source_lang = ttk.Combobox(lang_frame, values=sorted(self.languages.keys()),
                                        width=20, state="readonly")
        self.source_lang.set("English")
        self.source_lang.grid(row=0, column=1, padx=5)

        # Swap button
        swap_btn = tk.Button(lang_frame, text="⇄", font=("Arial", 14),
                             command=self.swap_languages, bg="#4CAF50", fg="white",
                             cursor="hand2", width=3)
        swap_btn.grid(row=0, column=2, padx=10)

        # Target language
        tk.Label(lang_frame, text="To:", font=("Arial", 12),
                 bg="#f0f0f0").grid(row=0, column=3, padx=5)
        self.target_lang = ttk.Combobox(lang_frame, values=sorted(self.languages.keys()),
                                        width=20, state="readonly")
        self.target_lang.set("Spanish")
        self.target_lang.grid(row=0, column=4, padx=5)

        # Input text area
        input_frame = tk.Frame(self.root, bg="#f0f0f0")
        input_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        tk.Label(input_frame, text="Enter text:", font=("Arial", 12),
                 bg="#f0f0f0").pack(anchor="w")

        self.input_text = tk.Text(input_frame, height=8, font=("Arial", 11),
                                  wrap=tk.WORD, relief=tk.SOLID, borderwidth=1)
        self.input_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Translate button
        translate_btn = tk.Button(self.root, text="Translate", font=("Arial", 14, "bold"),
                                  command=self.translate_text, bg="#2196F3", fg="white",
                                  cursor="hand2", padx=30, pady=10)
        translate_btn.pack(pady=10)

        # Output text area
        output_frame = tk.Frame(self.root, bg="#f0f0f0")
        output_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        tk.Label(output_frame, text="Translation:", font=("Arial", 12),
                 bg="#f0f0f0").pack(anchor="w")

        self.output_text = tk.Text(output_frame, height=8, font=("Arial", 11),
                                   wrap=tk.WORD, relief=tk.SOLID, borderwidth=1,
                                   state=tk.DISABLED)
        self.output_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Action buttons
        action_frame = tk.Frame(self.root, bg="#f0f0f0")
        action_frame.pack(pady=10)

        copy_btn = tk.Button(action_frame, text="📋 Copy", font=("Arial", 11),
                             command=self.copy_translation, bg="#FF9800", fg="white",
                             cursor="hand2", padx=20, pady=5)
        copy_btn.pack(side=tk.LEFT, padx=5)

        clear_btn = tk.Button(action_frame, text="🗑️ Clear", font=("Arial", 11),
                              command=self.clear_all, bg="#f44336", fg="white",
                              cursor="hand2", padx=20, pady=5)
        clear_btn.pack(side=tk.LEFT, padx=5)

    def translate_text(self):
        text = self.input_text.get("1.0", tk.END).strip()

        if not text:
            messagebox.showwarning("Warning", "Please enter text to translate!")
            return

        source = self.languages.get(self.source_lang.get())
        target = self.languages.get(self.target_lang.get())

        try:
            translator = GoogleTranslator(source=source, target=target)
            translation = translator.translate(text)

            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert("1.0", translation)
            self.output_text.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Error", f"Translation failed: {str(e)}")

    def swap_languages(self):
        source = self.source_lang.get()
        target = self.target_lang.get()
        self.source_lang.set(target)
        self.target_lang.set(source)

    def copy_translation(self):
        text = self.output_text.get("1.0", tk.END).strip()
        if text:
            pyperclip.copy(text)
            messagebox.showinfo("Success", "Translation copied to clipboard!")
        else:
            messagebox.showwarning("Warning", "No translation to copy!")

    def clear_all(self):
        self.input_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = TranslationApp(root)
    root.mainloop()