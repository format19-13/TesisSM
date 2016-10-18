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
    "app_key": "HdmtYkvu7gqv64RAE6DTYhCYs",
    "app_secret": "F4NMWUsBgSnW4d4bzXsG2hOEFxBVXBhZHAim6bx8s7zvU5cjuV",
    "app_access_token": "780859625448939521-XZ0Pz1XH4fXBQuTZwYjoL49fTLMQSbK",
    "app_access_token_secret": "fiWErePI0a0KiKlwMvu9VRXXYN7ZdrAmFMiGhFouHsAWE",
    "description": "Twitter access key"
}

class EnumSource(Enum):
    SCREEN_NAMES1 = "screen_names1"
    SCREEN_NAMES2 = "screen_names2"
    SCREEN_NAMES3 = "screen_names3"
    SCREEN_NAMES4 = "screen_names4"
    SCREEN_NAMES5 = "screen_names5"

SOURCE = [EnumSource.SCREEN_NAMES1,EnumSource.SCREEN_NAMES2,EnumSource.SCREEN_NAMES3,EnumSource.SCREEN_NAMES4,EnumSource.SCREEN_NAMES5]


SCREEN_NAMES1="ssportart,LizOMakeup,sorianomarina,angga_dm,Bodhi1,,ovibarcelo,dkpisces,jose_alias,mgozalbo,caff,ildefons,msheshtawy,seeseuno,wellcomm,Studiolit,monsalvenrique,TheoHuibers,Emers28,BernabeMurguia,MrMarkZamora,chloesor,Victoriabioque,mrlindsayjones,not_eggabo,djmooney,balancedbites,ruben3d,jamesnkerr,globaltattooma,ShelliMartineau,mgdelpozuelo,alorza,edsalvato,jordimatas,sbonet,belkheldar,LauraHarders,teresa_perales,anushka_sen,Day_NightEvents,julieannluna,alexhwilliams,LisaSanderson,ash_nicholson,skernes,amomstake,tover_banda,C_reinwald,DonnaBaierStein,belll07,VinnyVerelli,MegMoynahan,mfeijoo,TyVelde,dujkan,lourdesmunoz,kevinmclarke,ksmithdc,angchip,SEOWritersUK,jimmykastner,StephensHeather,edchi,BerriDesign,pollen8,LauriePace,luipauli,bassdiety,sgschneider,GerryDawes,pabloformoso,ArangoCoaching,gamkt,gattardi,helenaancos,LiamKenniford,soozmoody,reygarrido,eCare_Diary,MarkWhiteley,CFAIco,MarielaBejar,SEXYCHEFLIZ,Timbahood,usamaf,TheRealJoshPaul,Craigrobinson,dianahuth,momblogger,fabrizio,DragonflyReiki,hectormilla,laniar,TORYCREATIVE,yvonic55,SueAnello"
SCREEN_NAMES2="suehawkins,RichWriting,srouvier,jogartra,cindycatz,lawyerbychoice,Adrimartin,menchhu,heyitsfree,michaelwhitney,kenbrand,EtcoMusic,TwittingMarina,lutxana,tgottron,maninhavana,matterofbiz,beatricewhelan,eldestran,fernand0,AlephBlog,csaavedrasexta,MartaRodriguezA,LegalNewsToUse,WWequinerescue,dtunkelang,ReissSudden,davelee,LaxmicharaChele,everyueveryme,Dede_Watson,adtothebone,pachogruas,suzee2253,ElviraLindo,milesj,misellebergonia,nathanielstern,skooloflife,mrtfernandez,blogawardsie,justShauntelle,emilybratkovich,corallarrosa,giacomoinches,ClickForSEO_GB,LozzerOfHampton,DonPhilbin,RealQueenPink,VMaryAbraham,CragHutchinson,AliciaPomares,seo01,josejuanagudo,maripuchi,hawbi,gABiplanas_rrhh,E_Valero,zblunt,Jerri_Cook,marctorrens,anamariallopis,mariupaolini,IR_oldie,Pablo_astudillo,MegScarborough,Adela_Micha,rjmbasket08,elenhitaES,ABurgermeister,weissallison,cbrew,sojoner,yjsoon,marcelopont,survivingstores,BoulderSuZ,LIZPADILLA,andyhodder,ebaste,Slniecko16,danichain_,jmschroeder,olgarodriguezfr,pnhart,eilidhdickson,KurtisSeid,tereamor,RZYGA,jmiguelbenitez,lettersfrom500,asenkut,_AaronKlein_,intven,JOECORKSCREW,titatorro,nagodelos,Agent350,katjahofmann,abbebuck"
SCREEN_NAMES3="chadrem,dutchboyinohio,angga_55,callkathy,susanshapiro,PakoRho,LindseyLou84,2DGraphicDesign,mendeley_com,gomezdelpozuelo,CharoIzquierdo,RalphALMuser,edgarmeij,pbaleta,srstrong,stranded_aoife,metzlerd,clarke_adrian,ofalk,TriciaKean,silviacobo,El_Pakozoico,Annabellvdb,alison,AdweekMelissa,borkurdotnet,angelcustodio,alicialeeke,jordimirobruix,IHCCSTEM,PHLane,Karishma_tondon,krulwich,KTrerotola,adammarkus,rubenjans,deanabb,meganschmit,TheMeganAnn,mrgomezponce,pilarroch,etnika,MarsTweep,javier_artiles,perlani,xamat,HansInglish,alliehunt_,A_Valenzuela,Hwatterworth,StefanGroschupf,hsutiffany,atjamie,MaiteFinch,zeldman,KRTLANE,jochenleidner,kohlschuetter,jc_gonzalez,Juancmselma,GuyCresearchIT,susan_ringley,Remetey,auroraferrer,aandreup,virginiog,JeffTincher,canos,martinhermsen,iraszl,Ruohoska,NataliaGomez_es,liquene,KimVallee,MattZaheen,Heimsath,PaulBaranda,cbrewster,dlavery62,Schumenu,mikomonik,JulieOni,britt_benson_,Major_Grooves,ashwinram,AFPChrisGraham,AntonGBR,Clarice_Brough,ErnieVelas,JiveCork,juliagomezcora,Nick_Software,sclopit,Susan_Kayne,CareerToolboxUS,GarciaAller,annieblogs,goobisgoofy,abduzeedo,jamiejmann"
SCREEN_NAMES4="DarrellMorrison,crisalonsofran,SueFernPhillips,SueMcCauley,mounialalmas,nattiyak,doctorcasado,monicamoro,worldreaders,danayoung,krasicki,megzozo,LuvSayal,aiemeelow,TheAmyTucker,honorharger,Centralpt,JuanVillamayor,ThinkFY,zkellyq,MiaPleasant,gauravshukla,Kbedosky,Galachitaa,marcus_carlsson,JoannaRees,rafalopezdiez,bmbartel,davidbodenham,pedrofdezdec,adland,marcedavis,SamAntar,fkschmidt,trailguide411,JDoughtry,CHRISbodenner,jimhesson,FrnkNlsn,mhafez,tiferetjournal,ANTONY_MELVIN,rosamariaartal,awadallah,venturestudio,eBAstatsGroup,vanis,carkas,NiallWharton,mathowie,marta_dominguez,edans,Beejer,amandalynferri,pogil,acmurillo,marieholm,xllinares,jjmerelo,RasmusAuction,miles_kehoe,FYANG,ojecua,josek_net,mattstephens,MistralS,rrebolledo,LaCleoraOrtiz,holtkampw,ImpeccableCoach,kavango,petezin,accidentalgreen,IncomeMC,griner,DeanDeBiase,BeltwayBargain,frugalretailing,sramana,MarDixon,Krule,sharonwegner,susan4ps,mabelbullon,GE_Miller,raemond,webempresa20,LourdesVerger,michibusch,AOMConnect,jasonhoyt,Radhouani,abellavida,KMNbooks,luisnomad,PeterLaBarbera,royale,yoleidycarvajal,MsConcu,prodigy_man"
SCREEN_NAMES5="AntonioConde,mlaudisio,ejoana,ovaismehboob,yenialvarez,AngelooC,bassoforhire,christyxcore,karinalegzdins,carmengelices,LeadingWomen,mranera,hautebeauty,sparCKL,AmySoundDesign,AmaMocci,susieques,lostinsecurity,SagBizAssocLLC,perejoanmitjans,lugonlo,ChaToX,ian_soboroff,mihailupunet,ITECS,jgroc,bmorearchitect,marcoslarena,RichardDevine,monicadeza,amominred,jothejrno,bi0xid,EWineNV,coachireneg,cloudchloe,Jobs_SEO_UK,Scotthull"

#hscribner,larsbeck
