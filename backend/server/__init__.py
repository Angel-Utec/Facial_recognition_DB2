# -*- coding: utf-8 -*-
#Para ejecutar:
#1. Ir a la carpeta backend
#2. SET FLASK_APP=server
#3. SET FLASK_ENV=development
#4. flask run

from unicodedata import name
from flask import(
    Flask,
    abort,
    jsonify,
    request
)
import backend
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import face_recognition
from models import setup_db, Imagen
import numpy as np

#Paginación
db = SQLAlchemy()
TODOS_PER_PAGE=1000000

def paginate(request,selection,isDescendent):
    if isDescendent:                #Cuando se crea con CURL, muestra solo los últimos recursos creados
        start = len(selection)-TODOS_PER_PAGE
        end = len(selection)
    
    else:
        page=request.args.get('page',1,type=int)  #request.args.get trae los recursos especificados por el usuario por query parameters (luego del '?' en el url viene los query parameters)
        start = (page-1)*TODOS_PER_PAGE
        end = start+TODOS_PER_PAGE
    
    resources = [resource.format() for resource in selection]
    current_resources= resources[start:end]
    return current_resources


#API

def create_app(test_config=None):
    app = Flask(__name__)
    setup_db(app)
    CORS(app, origins="*", max_age=10)
    
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, true')
        response.headers.add('Access-Control-Allow-Methods', 'GET, PATCH, POST, DELETE, OPTIONS')
        return response

    @app.route('/imagen', methods=['GET'])      
    def get_posts():
        selection=Imagen.query.order_by('id').all()
        imagenes=paginate(request,selection,False)
      
        if (len(imagenes))==0:
            abort(404) 

        return jsonify({    #Las claves estarán ordenadas en orden alfabético
            'success': True,
            'imagenes': imagenes,
            'total_imagenes': len(selection)
        })

    @app.route('/posts', methods=['POST'])
    def create_post():
        Imagen.query.delete()
        db.session.commit()
        imagen_res = request.files['imagen']
        top_k = int(request.form['top_k'])
        imagen = face_recognition.load_image_file(imagen_res)
        vector = face_recognition.face_encodings(imagen)
        res = backend.knn_sequential(vector,int(top_k),backend.carpeta_salida_seq)
        for distance, file_name in res:
            nombre_archivo = file_name[18:-4]
            nombre = file_name[18:-9]
            foto_direccion = file_name[18:-4] + ".jpg"
            print(f"Archivo: {file_name}, Distancia: {distance}")
            distance_res = -distance
            tipo = "knn"
            imagen = Imagen(tipo=tipo,nombre=nombre,nombre_archivo=nombre_archivo,foto_espec=foto_direccion,distancia=distance_res)
            imagen.insert()
        res2 = backend.KNN_Ktree(vector[0],top_k)
        for img in res2:
            nombre_archivo_ktree = img[:-9]
            nombre_ktree = img[:-4]
            foto_direccion_ktree = img[:-4] + ".jpg"
            tipo_Ktree = "Ktree"
            imagen = Imagen(tipo=tipo_Ktree,nombre=nombre_archivo_ktree,nombre_archivo=nombre_ktree,foto_espec=foto_direccion_ktree)
            imagen.insert()
        res3 = backend.KNN_HighD(vector[0],top_k)
        for file_name_High, distance_high in res3:
            nombre_archivo_high = file_name_High[:-9]
            nombre_high = file_name_High[:-4]
            foto_direccion_high = file_name_High[:-4] + ".jpg"
            tipo_high = "HighD"
            distance_highk = float(distance_high)
            imagen = Imagen(tipo=tipo_high,nombre=nombre_archivo_high,nombre_archivo=nombre_high,foto_espec=foto_direccion_high,distancia=distance_highk)
            imagen.insert()
        
        return jsonify({
            'success': True,
            # 'created': new_imagen_id,
            # 'tipo': new_imagen_tipo,
            # 'imagenes': current_images,
            # 'total_posts': len(selection)
        })
        
    #Manejo de errores

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'success': False,
            'code': 404,
            'message': 'resource not found'
        }),404

    @app.errorhandler(422)
    def unprocessable (error):
        return jsonify({
            'success': False,
            'code': 422,
            'message': 'Unprocessable'
        }), 422

    @app.errorhandler(500)
    def not_found(error):
        return jsonify({
            'success': False,
            'code': 500,
            'message': 'internal server error'
        }),500

    @app.errorhandler(403)
    def not_found(error):
        return jsonify({
            'success': False,
            'code': 403,
            'message': 'forbidden'
        }),403
    

    return app