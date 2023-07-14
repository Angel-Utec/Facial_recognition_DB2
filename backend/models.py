# -*- coding: utf-8 -*-
from flask_sqlalchemy import SQLAlchemy 
from datetime import datetime 


database_path='postgresql://postgres:postgres@localhost:5432/BD23'

#Configuracion
db=SQLAlchemy()

def setup_db(app, database_path=database_path):
    app.config['SQLALCHEMY_DATABASE_URI']=database_path
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False
    db.app=app
    db.init_app(app)
    with app.app_context():
        db.create_all()

#Modelos
class Imagen(db.Model):
    __tablename__ = 'imagen'
    id = db.Column(db.Integer, primary_key = True)
    tipo = db.Column(db.Text, nullable = False)
    nombre = db.Column(db.Text,nullable = False)
    foto_espec = db.Column(db.Text,nullable = False)
    distancia = db.Column(db.Text,nullable = False)

    #Metodo que formatee el objeto a json para devolverlo a mi API y que no de errores
    def format(self):
        return {
            'id':self.id,
            'tipo': self.tipo,
            'nombre': self.nombre,         
            'foto_espec':self.foto_espec,
            'distancia':self.distancia
        }

    #Metodo que permite la inserción de un post a través de nuestra API
    def insert(self):
        db.session.add(self)
        db.session.commit()

    def update(self):
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()