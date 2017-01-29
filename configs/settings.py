import os,sys
import os.path

sys.path.append(os.path.abspath(os.pardir))

from enum import Enum

DIR_PREFIX="/Users/verouy" #MAC
#DIR_PREFIX="/home/vero" #LINUX

############################
# LOGGER
############################
LOGGING_BASE_PATH = '/var/log/twitter_streamer/'
LOGGING_ROOT_NAME = 'logfile'

############################

############################
# STREAMER
############################
TWITTER_ACCESS_KEYS = {
    "app_key": "HdmtYkvu7gqv64RAE6DTYhCYs",
    "app_secret": "F4NMWUsBgSnW4d4bzXsG2hOEFxBVXBhZHAim6bx8s7zvU5cjuV",
    "app_access_token": "780859625448939521-XZ0Pz1XH4fXBQuTZwYjoL49fTLMQSbK",
    "app_access_token_secret": "fiWErePI0a0KiKlwMvu9VRXXYN7ZdrAmFMiGhFouHsAWE",
    "description": "Twitter access key"
}

TRACK_TERMS = 'uruguay'

# Cuentas a escuchar: Lista de IDs de cuentas de Twitter en un string separado por comas
FOLLOWS_IDS = '2304818876, 74452681'

# Montevideo
BOUNDING_BOXES = [
                    -56.96411, -35.09295,
                    -53.53638, -33.68778
]

############################

PLACES_COUNTRY_IDS = ['UY']

# EXTRACTOR
CANT_WORKERS = 2


class EnumSource(Enum):
    GEOLOCATION = "geolocation"
    TRACKTERMS = "trackterms"
    FOLLOW = "follow"

SOURCE = [EnumSource.GEOLOCATION, EnumSource.TRACKTERMS, EnumSource.FOLLOW]


class EnumSource(Enum):
    SCREEN_NAMES = "screen_names"

SOURCE = [EnumSource.SCREEN_NAMES]

#pauroda,croibal,jdelagrana
SCREEN_NAMES="RalphALMuser,pachogruas,mrtfernandez,rrebolledo,maria_vega_,gemaconga,olgarodriguezfr,VigelaCoaching,AlvaroGarciaP90,mmendizabal1,NaturalVoyager,Beltran_Lola,h_torres,isanseba,ncampos,MistralS,ManushGalvan,mfeijoo,juancarcubeiro,A_Valenzuela,srouvier,AntonioRull,abarceloh25,juanma_nieves,Alberto_Blanco,AlexandraaBerr,ser_seo,gomezdelpozuelo,AngelMGTech,marta_dominguez,maripuchi,JSanchezFranco,Adela_Micha,sboxcoaching,Naruedyoh,anablanco_tve,Rhmedia_es,juantorreslopez,GarciaAller,MariTriniGiner,mabelbullon,jmgomezperez,LuisTomala,chgarciacortes,javierbri,sorianomarina,ojecua,jesusferre,shora,helenaancos"
SCREEN_NAMES2="claudiogarcia82,Cati_Nicolau,MaiteFinch,RafaDiaz1,MarielaBejar,carlosecue,oscargcervella,nagodelos,segundaopor,hawbi,emiliodebenito,RealQueenPink,emoragues,espinosa_albert,ginesalarcon,mmmendieta,Mar_Malabo,perezreverte,felixbernet,MOVEITER,CesarDalmau,MateoDomenech,prodigy_man,yvonic55,FjaureguiC,vanis,DiegoIsabel1,mikomonik,mariupaolini,JoseTrecet,silviacobo,THFFernando,pfcdgayo,titatorro,helenaresano,hectormilla,jgarridoo,biogeocarlos,Sandra_GarciaF,Pep_Jimenez,joancmarch,heteronima,etnika,sebasmuriel,lopezdoriga,luislopezcuenca,eapesteguia,diegocobelo,GlobalBrand_,pabloformoso,ivanpuente,quiquemateuvlc,julietabolullo,mrgomezponce,JesusEncinar,jordimirobruix,RobCarballo,pepecuesta,CarmeChaparro,Juancmselma,juliagomezcora,morenobarber,yoleidycarvajal,monicamoro,mranera,soozmoody,juanant_galindo,evacolladoduran,LaCiencibilidad,copano,JalGuerrero,nuriarocagranel,alfonsomarco,david_carnicer,elenhitaES,lutxana,Carmen_RRHH,pilarroch,globaltattooma,ohcarool,ovibarcelo,Acedotor,kicorangel,reygarrido,cesarpiqueras,jgonzalez_es,rosamariaartal,EBDTraining,Ramirofortea,Galachitaa,petezin,worldreaders,susannagriso,SalvaRBI,Elenavama,clementealvarez,LuvSayal,albertoartero,cesarfitness,joanplanas,Victoriabioque,milaximenez,webempresa20,Hector_Cabrera_,anamariallopis,JesusGallent,fernand0,NataliaGomez_es,virgula,lourdesmunoz,leonardpera,kenshin23,PilarAJEV,corallarrosa,ytzvan,Rozyo_,energica,juanmerodio,almarro1,granero,mgdelpozuelo,catyforteza,miss_belmont,AliciaPomares,ChapoyPati,acoronadoc,andreurobuste,CarlosBecler,anavidalegea,MsConcu,gABiplanas_rrhh,billiesastre,ArangoCoaching,PCVillarcayo,diegolleo,cris_villanueva,belenboville,javiersalinas00,carlosguadian,sergeyodintsov,tereamor,eblanperis,LourdesVerger,factoriagris,respetocanas,CharoIzquierdo,perejoanmitjans,CarolinaGavilan,_ANAMILAN_,wellcomm,rodriguezbraun,GomezRufo,albeitar,ManuelGimnez,Elissambura,El_Pakozoico,luisdiazdeldedo,mlaudisio,luisgosalbez,Acliment,EtcoMusic,EsperanzAguirre,E_Valero,cutefilm,cdelamorTVE,flowergirona,carmengelices,Pablo_astudillo,sarabarass,crisalonsofran,auroraferrer,Pedrobiurrun,Alesonline,juanlusanchez,seeseuno,lourdesmoralesg,miguel_a_alonso,ejoana,monicadeza,juancbarcelo,AdelaidaUMA,vdgracia,teresa_perales,HacheRomero,JuanjoMorodo,luismiguelcas,jogartra,ingeosolum,AmiGerpe,tinofernandez,bibiladi,iescolar,socialsmr,csaavedrasexta,CarmenAlcayde,ElviraLindo,MartaRodriguezA"

SCREEN_NAMES="THFFernando,mabelbullon,MsConcu,mrgomezponce,LaCiencibilidad,MateoDomenech,sboxcoaching,ingeosolum,Carmen_RRHH,juancarcubeiro,pabloformoso,Pep_Jimenez,flowergirona,leonardpera,pilarroch,reygarrido,Alberto_Blanco,ArangoCoaching,andreurobuste,_ANAMILAN_,jgonzalez_es,tereamor,acoronadoc,Alesonline,maripuchi,mlaudisio,kenshin23,ginesalarcon,marta_dominguez,A_Valenzuela,nagodelos,respetocanas,carlosguadian,El_Pakozoico,croibal,belenboville,mmendizabal1,Victoriabioque,lourdesmunoz,ncampos,shora,helenaresano,helenaancos,RalphALMuser,anavidalegea,joancmarch,joanplanas,MOVEITER,Elissambura,ytzvan,rodriguezbraun,HacheRomero,pachogruas,nuriarocagranel,silviacobo,Acliment,oscargcervella,Naruedyoh,vanis,GarciaAller,Adela_Micha,mfeijoo,miss_belmont,etnika,worldreaders,almarro1,AdelaidaUMA,Beltran_Lola,ElviraLindo,susannagriso,SalvaRBI,JesusEncinar,ChapoyPati,juancbarcelo,isanseba,mrtfernandez,biogeocarlos,albeitar,gomezdelpozuelo,corallarrosa,NataliaGomez_es,anamariallopis,yvonic55,juanlusanchez,AntonioRull,luisgosalbez,cesarpiqueras,AngelMGTech,julietabolullo,soozmoody,socialsmr,rosamariaartal,catyforteza,JuanjoMorodo,ejoana,fernand0,teresa_perales,chgarciacortes,heteronima,jmgomezperez,AlvaroGarciaP90,PCVillarcayo,felixbernet,segundaopor,claudiogarcia82,EBDTraining,cdelamorTVE,AliciaPomares,globaltattooma,kicorangel,jesusferre,MariTriniGiner,lourdesmoralesg,luismiguelcas,cesarfitness,eblanperis,evacolladoduran,jogartra,pfcdgayo,Ramirofortea,AlexandraaBerr,diegocobelo,ManushGalvan,Rhmedia_es,LourdesVerger,RealQueenPink,hawbi,david_carnicer,srouvier,sergeyodintsov,bibiladi,RobCarballo,prodigy_man,Cati_Nicolau,juanant_galindo,virgula,Pablo_astudillo,vdgracia,ManuelGimnez,anablanco_tve,ivanpuente,Mar_Malabo,eapesteguia,LuisTomala,CesarDalmau,crisalonsofran,lopezdoriga,Rozyo_,albertoartero,sebasmuriel,morenobarber,rrebolledo,CarmenAlcayde,jdelagrana,javiersalinas00,yoleidycarvajal,carlosecue,jgarridoo,lutxana,copano,sorianomarina,juliagomezcora,espinosa_albert,perezreverte,h_torres,auroraferrer,juantorreslopez,ojecua,titatorro,olgarodriguezfr,EsperanzAguirre,MistralS,maria_vega_,pepecuesta,elenhitaES,CarolinaGavilan,granero,billiesastre,cris_villanueva,energica,VigelaCoaching,cutefilm,monicamoro,ohcarool,LuvSayal,ser_seo,sarabarass,alfonsomarco,juanmerodio,gABiplanas_rrhh,mranera,carmengelices,Elenavama,luislopezcuenca,seeseuno,Galachitaa,FjaureguiC,emoragues,Juancmselma,pauroda,RafaDiaz1,Acedotor,PilarAJEV,hectormilla,JoseTrecet,luisdiazdeldedo,csaavedrasexta,DiegoIsabel1,clementealvarez,jordimirobruix,Sandra_GarciaF,JSanchezFranco,juanma_nieves,javierbri,CharoIzquierdo,Pedrobiurrun,AmiGerpe,webempresa20,wellcomm,JesusGallent,gemaconga,NaturalVoyager,mgdelpozuelo,E_Valero,GlobalBrand_,quiquemateuvlc,MartaRodriguezA,EtcoMusic,Hector_Cabrera_,petezin,mmmendieta,iescolar,factoriagris,tinofernandez,CarmeChaparro,ovibarcelo,GomezRufo,milaximenez,emiliodebenito,JalGuerrero,MaiteFinch,CarlosBecler,miguel_a_alonso,monicadeza,mariupaolini,MarielaBejar,mikomonik,perejoanmitjans,diegolleo,abarceloh25"
SCREEN_NAMES="THFFernando,mabelbullon,MsConcu,mrgomezponce,LaCiencibilidad,MateoDomenech,sboxcoaching,ingeosolum,Carmen_RRHH,juancarcubeiro,pabloformoso,Pep_Jimenez,flowergirona,leonardpera,pilarroch,reygarrido,Alberto_Blanco,ArangoCoaching,andreurobuste,_ANAMILAN_,jgonzalez_es,tereamor,acoronadoc,Alesonline,maripuchi,mlaudisio,kenshin23,ginesalarcon,marta_dominguez,A_Valenzuela,nagodelos,respetocanas,carlosguadian,El_Pakozoico,croibal,belenboville,mmendizabal1,Victoriabioque,lourdesmunoz,ncampos,shora,helenaresano,helenaancos,RalphALMuser,anavidalegea,joancmarch,joanplanas,MOVEITER,Elissambura,ytzvan,rodriguezbraun,HacheRomero,pachogruas,nuriarocagranel,silviacobo,Acliment,oscargcervella,Naruedyoh,vanis,GarciaAller,Adela_Micha,mfeijoo,miss_belmont,etnika,worldreaders,almarro1,AdelaidaUMA,Beltran_Lola,ElviraLindo,susannagriso,SalvaRBI,JesusEncinar,ChapoyPati,juancbarcelo,isanseba,mrtfernandez,biogeocarlos,albeitar,gomezdelpozuelo,corallarrosa,NataliaGomez_es,anamariallopis,yvonic55,juanlusanchez,AntonioRull,luisgosalbez,cesarpiqueras,AngelMGTech,julietabolullo,soozmoody,socialsmr,rosamariaartal,catyforteza,JuanjoMorodo,ejoana,fernand0,teresa_perales,chgarciacortes,heteronima,jmgomezperez,AlvaroGarciaP90,PCVillarcayo,felixbernet,segundaopor,claudiogarcia82,EBDTraining,cdelamorTVE,AliciaPomares,globaltattooma,kicorangel,jesusferre,MariTriniGiner,lourdesmoralesg,luismiguelcas,cesarfitness,eblanperis,evacolladoduran,jogartra,pfcdgayo,Ramirofortea,AlexandraaBerr,diegocobelo,ManushGalvan,Rhmedia_es,LourdesVerger,RealQueenPink,hawbi,david_carnicer,srouvier,sergeyodintsov,bibiladi,RobCarballo,prodigy_man,Cati_Nicolau,juanant_galindo,virgula,Pablo_astudillo,vdgracia,ManuelGimnez,anablanco_tve,ivanpuente,Mar_Malabo,eapesteguia,LuisTomala,CesarDalmau,crisalonsofran,lopezdoriga,Rozyo_,albertoartero,sebasmuriel,morenobarber,rrebolledo,CarmenAlcayde,jdelagrana,javiersalinas00,yoleidycarvajal,carlosecue,jgarridoo,lutxana,copano,sorianomarina,juliagomezcora,espinosa_albert,perezreverte,h_torres,auroraferrer,juantorreslopez,ojecua,titatorro,olgarodriguezfr,EsperanzAguirre,MistralS,maria_vega_,pepecuesta,elenhitaES,CarolinaGavilan,granero,billiesastre,cris_villanueva,energica,VigelaCoaching,cutefilm,monicamoro,ohcarool,LuvSayal,ser_seo,sarabarass,alfonsomarco,juanmerodio,gABiplanas_rrhh,mranera,carmengelices,Elenavama,luislopezcuenca,seeseuno,Galachitaa,FjaureguiC,emoragues,Juancmselma,pauroda,RafaDiaz1,Acedotor,PilarAJEV,hectormilla,JoseTrecet,luisdiazdeldedo,csaavedrasexta,DiegoIsabel1,clementealvarez,jordimirobruix,Sandra_GarciaF,JSanchezFranco,juanma_nieves,javierbri,CharoIzquierdo,Pedrobiurrun,AmiGerpe,webempresa20,wellcomm,JesusGallent,gemaconga,NaturalVoyager,mgdelpozuelo,E_Valero,GlobalBrand_,quiquemateuvlc,MartaRodriguezA,EtcoMusic,Hector_Cabrera_,petezin,mmmendieta,iescolar,factoriagris,tinofernandez,CarmeChaparro,ovibarcelo,GomezRufo,milaximenez,emiliodebenito,JalGuerrero,MaiteFinch,CarlosBecler,miguel_a_alonso,monicadeza,mariupaolini,MarielaBejar,mikomonik,perejoanmitjans,diegolleo,abarceloh25"
