import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. Cargar variables de entorno y cliente apuntando a Groq
load_dotenv()
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

MODELO = "llama-3.3-70b-versatile"

# 2. DEFINIR LA HERRAMIENTA (Nuestra función en Python puro)
def calculadora(operacion: str, a: float, b: float) -> float:
    """Calculadora básica para sumar, restar, multiplicar y dividir."""
    if operacion == "sumar": return a + b
    elif operacion == "restar": return a - b
    elif operacion == "multiplicar": return a * b
    elif operacion == "dividir": return a / b if b != 0 else "Error: División por cero"
    return "Operación no válida"

# 3. DESCRIBIR LA HERRAMIENTA AL MODELO
tools = [{
    "type": "function",
    "function": {
        "name": "calculadora",
        "description": "Realiza operaciones matemáticas básicas entre dos números.",
        "parameters": {
            "type": "object",
            "properties": {
                "operacion": {"type": "string", "enum": ["sumar", "restar", "multiplicar", "dividir"]},
                "a": {"type": "number", "description": "Primer número"},
                "b": {"type": "number", "description": "Segundo número"}
            },
            "required": ["operacion", "a", "b"]
        }
    }
}]

# 4. CICLO DE RAZONAMIENTO Y EJECUCIÓN (ReAct Loop)
def ejecutar_agente(pregunta_usuario: str):
    print(f"\nUsuario: {pregunta_usuario}")
    mensajes = [{"role": "user", "content": pregunta_usuario}]

    # Paso A: El Modelo Razona (Reasoning)
    respuesta = client.chat.completions.create(
        model=MODELO,
        messages=mensajes,
        tools=tools
    )
    mensaje_modelo = respuesta.choices[0].message

    # Paso B: ¿El modelo decidió usar una herramienta? (Tool Calling)
    if mensaje_modelo.tool_calls:
        tool_call = mensaje_modelo.tool_calls[0]
        nombre_funcion = tool_call.function.name
        argumentos = json.loads(tool_call.function.arguments)

        print(f"🤖 Agente Pensó: Necesito usar la herramienta '{nombre_funcion}' con los datos {argumentos}")

        # Paso C: Ejecutar la herramienta en Python (Execution)
        if nombre_funcion == "calculadora":
            resultado = calculadora(**argumentos)
            print(f"⚙️ Python Ejecutó: Resultado = {resultado}")

            # Paso D: Enviar el resultado de vuelta al modelo para la respuesta final
            mensajes.append(mensaje_modelo)
            mensajes.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(resultado)
            })

            respuesta_final = client.chat.completions.create(
                model=MODELO,
                messages=mensajes
            )
            print(f"🤖 Agente Respuesta Final: {respuesta_final.choices[0].message.content}")
    else:
        print(f"🤖 Agente Respuesta Directa: {mensaje_modelo.content}")

# 5. PRUEBA DEL AGENTE
if __name__ == "__main__":
    ejecutar_agente("¿Cuánto es 345 multiplicado por 12?")
