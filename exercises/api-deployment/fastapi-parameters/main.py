from fastapi import FastAPI
from pydantic import BaseModel

class RequestBody(BaseModel):

    field_id: int


app = FastAPI()


@app.post('/add_data/{path_param}')
async def exercise_function(path_param: str,
                            body_param: RequestBody,
                            query_param: int = 1):

    return {'path': path_param, 'query': query_param, 'body': body_param}
