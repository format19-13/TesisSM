# TesisVT

#Para iniciar el servicio mongo
brew services start mongodb 

BACKUP DB: mongodump --collection users --db tesisdb
RESTORE DB: mongorestore dump-2013-10-25/