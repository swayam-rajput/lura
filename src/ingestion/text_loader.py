from pathlib import Path
import os

def load_text(file_path:str):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f'{file_path} not found')
    
    if path.suffix.lower() !='.txt':
        raise ValueError(f'{file_path} is not a .txt file')

    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        text = f.read()
    text = " ".join(text.split())
    return text

