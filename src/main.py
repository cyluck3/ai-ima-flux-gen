from together import Together
import aiohttp
import asyncio
import uuid
import os
import json
from utils.agent import flowtask

# API
# tgp_v1_lkTrxs-5aS21tgMJnoTZt0EZ3YZ3RGNVfhb_8OTWkP0

def generate_unique_foldername(idmanual, new): # generador de nombres de carpeta
    """Generates a unique foldername using UUID."""
    unique_id = uuid.uuid4()  # Generar id único

    filename = ""
    if (new == False):
        filename = f"{idmanual}_/"  # Crea carpeta para muchos usos (guardando en la misma del prompt en uso)
    else:
        filename = f"{idmanual}_{unique_id}/"  # Crea una nueva carpeta siempre (incluso con el mismo prompt)
    return filename


def generate_unique_filename(): # generador de nombres de archivo
    """Generates a unique filename using UUID."""
    unique_id = uuid.uuid4()  # Generate a unique ID
    filename = f"imagen_{unique_id}.png"  # Create the filename
    return filename


async def imaGen(prompt, session, unique_foldername):
    """
    Genera una imagen a partir de un prompt utilizando la API de Together.
    """
    print(f"Generando imagen para: {prompt}")  # Imprime el prompt actual
    client = Together(api_key="tgp_v1_lkTrxs-5aS21tgMJnoTZt0EZ3YZ3RGNVfhb_8OTWkP0")
    response = client.images.generate(
        prompt=prompt,
        model="black-forest-labs/FLUX.1-schnell",
        steps=4,
        disable_safety_checker=True,
    )
    # Obtener la URL de la imagen generada
    image_url = response.data[0].url

    # Descargar la imagen y guardarla localmente
    async with session.get(image_url) as resp:
        image_data = await resp.read()

    # Generate a unique filename
    unique_filename = generate_unique_filename()

    # Definir el nombre del archivo de salida
    with open(f"./modelos/modelgen/picsgenerated/{unique_foldername}" + unique_filename, "wb") as file:
        file.write(image_data)
    print(f"Imagen generada y guardada para: {prompt}")  # Imprime cuando la imagen se guarda


async def maxgen():
    """
    Genera imágenes para cada prompt de forma secuencial (una tras otra).
    """
    # TEMA SOBRE EL QUE SE GENERARAN PROMPTS PARA IMÁGENES
    about = "Gatos en otros planetas surrealistas" #Especificar de qué serán las imágenes

    # SE CREAN LAS CARPETAS DE ALMACENAMIENTO DE IMÁGENES SI NO EXISTEN
    # Create the folder if it doesn't exist NOMBRE DE CARPETA GENERAL CONTENEDORA DE CARPETAS DE IMÁGENES
    os.makedirs("./picsgenerated/", exist_ok=True)

    # Generate a unique foldername NOMBRE UNICO DE CARPETA PARA UNA PILA DE IMÁGENES
    unique_foldername = generate_unique_foldername(about, False)
    os.makedirs(f"/modelos/modelgen/picsgenerated/{unique_foldername}", exist_ok=True)

    # GENERADOR DE PROMPTS AUTOMATIZADO
    agente = flowtask("generador-prompts", "gemma-3-27b-it") # Recordar: Deepseek para temas con menos restricciones (no ilegales), Google no permite prompts delicados, sugestivos NSFW, etc. Las API's de OpenRouter tienen límite diario y generan error "choices" tras alcanzar el límite, cuando eso pase reemplacer por alguna API de una fuente diferena a OpenRouter

    res = await agente.add_instruction(f"""eres un experto generador de diccionarios JSON, vas a crear una lista de 20 prompts en inglés para generar imagenes sobre: '{about}', tu me la entregarás en el siguiente formato, sin usar markdown: 
    {{ 
       "1":"prompt1",
       "2":"prompt2",
       "3":"prompt3",
       "4":"prompt4",
       "5":"prompt5",
        etc.
    }}
    Solo entregarás dicho diccionario con prompts, sin decir ni agregar nada más antes o después""")

    listprompts = []
    try:
        tojson = json.loads(res)
        print(tojson, type(tojson))
        for key, value in tojson.items():
            print(value)
            listprompts.append(value)

    except Exception as e:
        print(e)


    # Escribir en esta lista todos los prompts a procesar, 1 prompt = 1 imagen, MODO MANUAL
    lista_personalizada = [
        "prompt1",
        "prompt2",
        "prompt3",
        "prompt4"
    ]
    async with aiohttp.ClientSession() as session:
        for element in listprompts:
            await imaGen(f"{element}", session, unique_foldername)  # Wait for each image to be generated and saved


try:
    asyncio.run(maxgen())
except Exception as e:
    print(f"An error occurred: {e}")  # Added: Print the error message




# Documentation
# https://www.together.ai/models/flux-1-schnell
# https://docs.together.ai/docs/images-overview#supported-image-models

# https://huggingface.co/black-forest-labs/FLUX.1-schnell