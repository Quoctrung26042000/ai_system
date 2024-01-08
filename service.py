import asyncpg


async def connect_to_db():
    return await asyncpg.connect(
        user="postgres",
        password="postgres",
        database="postgres",
        host="localhost",
        port="5432"
    )



