from fastapi import FastAPI
from api.routes import router  # make sure router is imported from routes.py

app = FastAPI()

# include the router
app.include_router(router)
