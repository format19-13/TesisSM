
PASOS:

1) Configurar ambiente en configs\settings.py:

	DIR_PREFIX="/Users/verouy" #MAC
	#DIR_PREFIX="/home/vero" #LINUX

2) Iniciar el servicio mongo
	brew services start mongodb 
	service mongod restart


BACKUP DB: mongodump --collection users --db tesisdb
RESTORE DB: mongorestore dump-2013-10-25/
--236?
