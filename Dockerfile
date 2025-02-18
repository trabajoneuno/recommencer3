FROM python:3.9

WORKDIR /app

# Instalar dependencias primero para aprovechar el caché de Docker
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copiar el código y archivos necesarios
COPY . .

# Asegúrate de que tus archivos productos.csv y recomendacion.tflite estén en el mismo directorio

# Exponer el puerto que usa tu aplicación
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["python", "main.py"]
