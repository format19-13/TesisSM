from enum import Enum

############################
# LOGGER
############################
LOGGING_BASE_PATH = '/var/log/twitter_streamer/'
LOGGING_ROOT_NAME = 'logfile'

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

class EnumSource(Enum):
    SCREEN_NAMES = "screen_names"

SOURCE = [EnumSource.SCREEN_NAMES]

#pauroda,croibal,jdelagrana
SCREEN_NAMES="RalphALMuser,pachogruas,mrtfernandez,rrebolledo,maria_vega_,gemaconga,olgarodriguezfr,VigelaCoaching,AlvaroGarciaP90,mmendizabal1,NaturalVoyager,Beltran_Lola,h_torres,isanseba,ncampos,MistralS,ManushGalvan,mfeijoo,juancarcubeiro,A_Valenzuela,srouvier,AntonioRull,abarceloh25,juanma_nieves,Alberto_Blanco,AlexandraaBerr,ser_seo,gomezdelpozuelo,AngelMGTech,marta_dominguez,maripuchi,JSanchezFranco,Adela_Micha,sboxcoaching,Naruedyoh,anablanco_tve,Rhmedia_es,juantorreslopez,GarciaAller,MariTriniGiner,mabelbullon,jmgomezperez,LuisTomala,chgarciacortes,javierbri,sorianomarina,ojecua,jesusferre,shora,helenaancos"
SCREEN_NAMES2="claudiogarcia82,Cati_Nicolau,MaiteFinch,RafaDiaz1,MarielaBejar,carlosecue,oscargcervella,nagodelos,segundaopor,hawbi,emiliodebenito,RealQueenPink,emoragues,espinosa_albert,ginesalarcon,mmmendieta,Mar_Malabo,perezreverte,felixbernet,MOVEITER,CesarDalmau,MateoDomenech,prodigy_man,yvonic55,FjaureguiC,vanis,DiegoIsabel1,mikomonik,mariupaolini,JoseTrecet,silviacobo,THFFernando,pfcdgayo,titatorro,helenaresano,hectormilla,jgarridoo,biogeocarlos,Sandra_GarciaF,Pep_Jimenez,joancmarch,heteronima,etnika,sebasmuriel,lopezdoriga,luislopezcuenca,eapesteguia,diegocobelo,GlobalBrand_,pabloformoso,ivanpuente,quiquemateuvlc,julietabolullo,mrgomezponce,JesusEncinar,jordimirobruix,RobCarballo,pepecuesta,CarmeChaparro,Juancmselma,juliagomezcora,morenobarber,yoleidycarvajal,monicamoro,mranera,soozmoody,juanant_galindo,evacolladoduran,LaCiencibilidad,copano,JalGuerrero,nuriarocagranel,alfonsomarco,david_carnicer,elenhitaES,lutxana,Carmen_RRHH,pilarroch,globaltattooma,ohcarool,ovibarcelo,Acedotor,kicorangel,reygarrido,cesarpiqueras,jgonzalez_es,rosamariaartal,EBDTraining,Ramirofortea,Galachitaa,petezin,worldreaders,susannagriso,SalvaRBI,Elenavama,clementealvarez,LuvSayal,albertoartero,cesarfitness,joanplanas,Victoriabioque,milaximenez,webempresa20,Hector_Cabrera_,anamariallopis,JesusGallent,fernand0,NataliaGomez_es,virgula,lourdesmunoz,leonardpera,kenshin23,PilarAJEV,corallarrosa,ytzvan,Rozyo_,energica,juanmerodio,almarro1,granero,mgdelpozuelo,catyforteza,miss_belmont,AliciaPomares,ChapoyPati,acoronadoc,andreurobuste,CarlosBecler,anavidalegea,MsConcu,gABiplanas_rrhh,billiesastre,ArangoCoaching,PCVillarcayo,diegolleo,cris_villanueva,belenboville,javiersalinas00,carlosguadian,sergeyodintsov,tereamor,eblanperis,LourdesVerger,factoriagris,respetocanas,CharoIzquierdo,perejoanmitjans,CarolinaGavilan,_ANAMILAN_,wellcomm,rodriguezbraun,GomezRufo,albeitar,ManuelGimnez,Elissambura,El_Pakozoico,luisdiazdeldedo,mlaudisio,luisgosalbez,Acliment,EtcoMusic,EsperanzAguirre,E_Valero,cutefilm,cdelamorTVE,flowergirona,carmengelices,Pablo_astudillo,sarabarass,crisalonsofran,auroraferrer,Pedrobiurrun,Alesonline,juanlusanchez,seeseuno,lourdesmoralesg,miguel_a_alonso,ejoana,monicadeza,juancbarcelo,AdelaidaUMA,vdgracia,teresa_perales,HacheRomero,JuanjoMorodo,luismiguelcas,jogartra,ingeosolum,AmiGerpe,tinofernandez,bibiladi,iescolar,socialsmr,csaavedrasexta,CarmenAlcayde,ElviraLindo,MartaRodriguezA"
