import unicodedata
import asyncpg


async def connect_to_db():
    return await asyncpg.connect(
        user="postgres",
        password="postgres",
        database="postgres",
        host="localhost",
        port="5432"
    )
        
def normalize_folder_name(folder_name):
    return ''.join(c for c in unicodedata.normalize('NFD', folder_name) if unicodedata.category(c) != 'Mn')