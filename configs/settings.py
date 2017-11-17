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
    "app_key": "U20LyEZpsROJSo1hE3GdSigC6",
    "app_secret": "u75pnyuv6nBvsdTpeC75j7OdXobgNqRoZekPyP8l8GlYYO6Rfm",
    "app_access_token": "780859625448939521-XZ0Pz1XH4fXBQuTZwYjoL49fTLMQSbK",
    "app_access_token_secret": "fiWErePI0a0KiKlwMvu9VRXXYN7ZdrAmFMiGhFouHsAWE",
    "description": "Twitter access key"
}

TRACK_TERMS = ''

# Cuentas a escuchar: Lista de IDs de cuentas de Twitter en un string separado por comas
FOLLOWS_IDS =''

# Montevideo
BOUNDING_BOXES = [-56.96411, -35.09295,-53.53638, -33.68778]
#-71.54296875,-52.8558641779 -66.005859375,-23.926013033 -53.4375,-32.9533681458 -71.54296875,-52.8558641779]
#                    -56.96411, -35.09295,
#                   -53.53638, -33.68778


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

SCREEN_NAMES="ralphalmuser,pachogruas,mrtfernandez,rrebolledo,maria_vega_,gemaconga,olgarodriguezfr,vigelacoaching,alvarogarciap90,mmendizabal1,naturalvoyager,beltran_lola,h_torres,isanseba,ncampos,mistrals,manushgalvan,mfeijoo,juancarcubeiro,a_valenzuela,srouvier,antoniorull,abarceloh25,juanma_nieves,alberto_blanco,alexandraaberr,ser_seo,gomezdelpozuelo,angelmgtech,marta_dominguez,maripuchi,jsanchezfranco,adela_micha,sboxcoaching,naruedyoh,anablanco_tve,rhmedia_es,juantorreslopez,garciaaller,maritriniginer,mabelbullon,jmgomezperez,luistomala,chgarciacortes,javierbri,sorianomarina,ojecua,jesusferre,shora,helenaancos,claudiogarcia82,cati_nicolau,maitefinch,rafadiaz1,marielabejar,carlosecue,oscargcervella,nagodelos,segundaopor,hawbi,emiliodebenito,realqueenpink,emoragues,espinosa_albert,ginesalarcon,mmmendieta,mar_malabo,perezreverte,felixbernet,moveiter,cesardalmau,mateodomenech,prodigy_man,yvonic55,fjaureguic,vanis,diegoisabel1,mikomonik,mariupaolini,josetrecet,silviacobo,thffernando,pfcdgayo,titatorro,helenaresano,hectormilla,jgarridoo,biogeocarlos,sandra_garciaf,pep_jimenez,joancmarch,heteronima,etnika,sebasmuriel,lopezdoriga, diegocobelo,globalbrand_,pabloformoso,ivanpuente,quiquemateuvlc,julietabolullo,mrgomezponce,jesusencinar,jordimirobruix,robcarballo,pepecuesta,carmechaparro,juancmselma,juliagomezcora,morenobarber,yoleidycarvajal,monicamoro,mranera,soozmoody,juanant_galindo,evacolladoduran,laciencibilidad,copano,jalguerrero,nuriarocagranel,alfonsomarco,david_carnicer,elenhitaes,lutxana,carmen_rrhh,pilarroch,globaltattooma,ohcarool,ovibarcelo,acedotor,kicorangel,cesarpiqueras,jgonzalez_es,rosamariaartal,ebdtraining,ramirofortea,galachitaa,petezin,worldreaders,susannagriso,salvarbi, elenavama,clementealvarez,luvsayal,albertoartero,cesarfitness,joanplanas,victoriabioque,milaximenez,webempresa20,hector_cabrera_,anamariallopis,jesusgallent,fernand0,nataliagomez_es,virgula,lourdesmunoz,leonardpera,kenshin23,pilarajev,corallarrosa,ytzvan,rozyo_,energica,juanmerodio,almarro1,granero,mgdelpozuelo,catyforteza,miss_belmont,aliciapomares,chapoypati,acoronadoc,andreurobuste,carlosbecler,anavidalegea,msconcu,gabiplanas_rrhh,billiesastre,arangocoaching,pcvillarcayo,diegolleo,cris_villanueva,belenboville,javiersalinas00,carlosguadian,sergeyodintsov,tereamor,eblanperis,lourdesverger,factoriagris,respetocanas,charoizquierdo,perejoanmitjans,carolinagavilan,_anamilan_,wellcomm,rodriguezbraun,gomezrufo,albeitar,manuelgimnez,elissambura,el_pakozoico,luisdiazdeldedo,mlaudisio,luisgosalbez,acliment,etcomusic,esperanzaguirre,e_valero,cutefilm,cdelamortve,flowergirona,carmengelices,pablo_astudillo,sarabarass,crisalonsofran,auroraferrer,pedrobiurrun,alesonline,juanlusanchez,seeseuno,lourdesmoralesg,miguel_a_alonso,ejoana,monicadeza,juancbarcelo,adelaidauma,vdgracia,teresa_perales,hacheromero,juanjomorodo,luismiguelcas,jogartra,ingeosolum,amigerpe,tinofernandez,bibiladi,iescolar,socialsmr,csaavedrasexta,carmenalcayde,elviralindo,martarodrigueza,,eapesteguia,luislopezcuenca"
#STREAMER RETURNS ERROR FOR THESE USERS: pauroda,croibal,jdelagrana,reygarrido
#USERS NO LONGER ACTIVE="cati_nicolau,csaavedrasexta,etcomusic,hawbi,laciencibilidad,mar_malabo,realqueenpink, rozyo_, salvarbi, sarabarass, seeseuno, socialsmr, yvonic55"
#TOTAL USERS IN DB: 233 (might be needed to run again extractUsers to add "ohcarool")