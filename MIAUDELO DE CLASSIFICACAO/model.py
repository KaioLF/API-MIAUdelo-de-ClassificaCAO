from sqlite3 import dbapi2
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
from PIL import Image
from pydantic import BaseModel
from typing import List
import io
import numpy as np
import sys
import tensorflow as tf
import base64
import sqlite3

#Carregando o modelo
model = tf.keras.models.load_model("5x84x3-80.model")
print("Modelo Carregado")

# Get the input shape for the model layer
input_shape = model.layers[0].input_shape

#Definindo nossas classes, as quais a API vai retornar
categories = ["Dog", "Cat"]

#Definindo o nome do banco de dados
DB_NAME = "database.db"

# Define a aplicativo da FastAPI
app = FastAPI()

# Definindo a resposta da predicao
class Prediction(BaseModel):
  filename: str
  contenttype: str
  prediction: List[float] = []
  likely_class: int
  predicted_class : str

# Define a resposta do outpout do banco de dados
class DBoutput(BaseModel):
  id: int
  filenameDB : str
  predicted_classDB : str

#Cria o banco de dados
def create_database():
  try:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE database(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          filename TEXT,
          predicted_class TEXT
        );
     """)
    conn.close()
  except Exception as e:
       print(e)

#Guarda as predições no banco de dados
def insert_prediction(filename_db : str, predicted_class_db : str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(f"""
        INSERT INTO database (filename, predicted_class)
        VALUES (?, ?);
    """, (filename_db, predicted_class_db))
    conn.commit()
    id = cursor.lastrowid
    conn.close()
    return id

#Acessa as predições no banco de dados
def get_prediction_db():
  conn = sqlite3.connect(DB_NAME)
  cursor = conn.cursor()
  cursor.execute("""
      SELECT 
          id, filename, predicted_class 
      FROM 
          database
    """)
  lines = cursor.fetchall()
  conn.close()

  dboutputs = []
  for line in lines:
    id = line[0]
    filename = line[1]
    predicted_class = line[2]
    dboutput = DBoutput(id=id,filenameDB=filename, predicted_classDB=predicted_class)
    dboutputs.append(dboutput)

  return dboutputs


# Definindo a rota principal
@app.get('/')
def root_route():
  return { "Erro": 'Use /docs ao invés da rota raiz!' }

# Define a rota do /prediction
@app.post('/prediction/', response_model = Prediction)

async def prediction_route(file: UploadFile = File(...)):
  
  try:
    
    #Le a imagem do usuário
    user_image = await file.read()
    
    #Decoda a imagem encodada
    base64bytes = base64.b64decode(user_image)
    bytesObj = io.BytesIO(base64bytes)

    #Abre a imagem
    pil_image = Image.open(bytesObj)
    
    # Redimensionando a imagem
    pil_image = pil_image.resize((input_shape[1], input_shape[2]))

    # Convertendo a imamgem para RGB e evitando canais alpha
    if pil_image.mode == 'RGBA':
      pil_image = pil_image.convert('RGB')

    # Converte a imagem em escala de cinza
    if input_shape[3] and input_shape[3] == 1:
      pil_image = pil_image.convert('L')

    # Converte a imagem no formato do numpy
    numpy_image = np.array(pil_image).reshape((input_shape[1], input_shape[2], input_shape[3]))

    # Dados de escala
    numpy_image = numpy_image / 255

    # Gerando a predição
    prediction_array = np.array([numpy_image])
    predictions = model.predict(prediction_array)
    prediction = predictions[0]
    likely_class = np.argmax(prediction)
    predicted_class = categories[np.argmax(prediction)]
    
    id = insert_prediction(filename_db=file.filename, predicted_class_db=predicted_class)

    #Retorna nosso modelo base de respostas
    return {
      "filename" : file.filename,
      "contenttype": file.content_type,
      "prediction": prediction.tolist(),
      "likely_class": likely_class,
      "predicted_class" : predicted_class
    }

  except:
    e = sys.exc_info()[1]
    raise HTTPException(status_code=500, detail=str(e))

#Definindo a rota GET que busca os dados no banco de dados
@app.get("/predictionsDB", response_model=List[DBoutput])
def get_predictions():
  outputs = get_prediction_db()
  return outputs

if __name__ == "__main__":
  create_database()
  uvicorn.run(app, host = "localhost", port = 8000)