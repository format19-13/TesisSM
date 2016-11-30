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

#royale,weissallison,larsbeck,hscribner, alexhwilliams,angchip da 401 unauthorized

SCREEN_NAMES="RichWriting,AntonGBR,angga_55,dkpisces,LizOMakeup,sclopit,RalphALMuser,pachogruas,mrtfernandez,rrebolledo,C_reinwald,luipauli,olgarodriguezfr,ReissSudden,ABurgermeister,clarke_adrian,edsalvato,anushka_sen,belkheldar,msheshtawy,Major_Grooves,GE_Miller,nathanielstern,survivingstores,DragonflyReiki,pbaleta,StefanGroschupf,DonPhilbin,IncomeMC,MistralS,mfeijoo,miles_kehoe,A_Valenzuela,PHLane,adland,justShauntelle,carkas,srouvier,suehawkins,LisaSanderson,Scotthull,gomezdelpozuelo,marta_dominguez,LiamKenniford,maripuchi,FYANG,griner,kenbrand,Adela_Micha,DonnaBaierStein,lostinsecurity,sbonet,pollen8,BernabeMurguia,megzozo,SEOWritersUK,Jerri_Cook,GarciaAller,mounialalmas,abellavida,raemond,davidbodenham,bassoforhire,Slniecko16,christyxcore,AntonioConde,mabelbullon,eBAstatsGroup,sojoner,2DGraphicDesign,nattiyak,gamkt,Agent350,sorianomarina,ojecua,mhafez,rubenjans,canos,helenaancos,sharonwegner,seo01,dlavery62,MaiteFinch,_AaronKlein_,MarielaBejar,marcoslarena,LindseyLou84,TheAmyTucker,Studiolit,jasonhoyt,borkurdotnet,danayoung,skernes,fkschmidt,Nick_Software,nagodelos,tover_banda,hawbi,xamat,TheoHuibers,Heimsath,RealQueenPink,Clarice_Brough,rafalopezdiez,KRTLANE,PaulBaranda,marieholm,jothejrno,prodigy_man,britt_benson_,IR_oldie,frugalretailing,yvonic55,FrnkNlsn,hsutiffany"

SCREEN_NAMES4="vanis,blogawardsie,mikomonik,mariupaolini,ruben3d,danichain_,belll07,silviacobo,momblogger,djmooney,aandreup,bmorearchitect,KurtisSeid,titatorro,cloudchloe,DarrellMorrison,accidentalgreen,ebaste,andyhodder,hectormilla,SamAntar,JiveCork,LauraHarders,Centralpt,martinhermsen,etnika,fabrizio,marcedavis,AOMConnect,dujkan,Karishma_tondon,jamiejmann,alison,StephensHeather,sramana,ashwinram,abbebuck,JoannaRees,giacomoinches,pabloformoso,MegMoynahan,lugonlo,Adrimartin,mrgomezponce,TheRealJoshPaul,rjmbasket08,jamesnkerr,ssportart,skooloflife,jordimirobruix,krasicki,Annabellvdb,holtkampw,AngelooC,ovaismehboob,yenialvarez,JuanVillamayor,MarsTweep,perlani,edans,jmiguelbenitez,EWineNV,Juancmselma,juliagomezcora,MarDixon,JDoughtry,yoleidycarvajal,angelcustodio,krulwich,xllinares,lawyerbychoice,monicamoro,honorharger,amomstake,mranera,MegScarborough,LauriePace,aiemeelow,alliehunt_,mathowie,MsConcu,mattstephens,gABiplanas_rrhh,tiferetjournal,cindycatz,pedrofdezdec,jochenleidner,ShelliMartineau,venturestudio,ArangoCoaching,LIZPADILLA,marcus_carlsson,Emers28,javier_artiles,jimhesson,MattZaheen,bassdiety,Bodhi1,ClickForSEO_GB,josek_net,zeldman,yjsoon,chadrem,ofalk,emilybratkovich,LaCleoraOrtiz,ash_nicholson,jc_gonzalez,SEXYCHEFLIZ,Hwatterworth,julieannluna,dianahuth,tereamor,SueFernPhillips,sparCKL,LaxmicharaChele,monsalvenrique,mgozalbo,metzlerd,kavango,LourdesVerger,abduzeedo,deanabb,KTrerotola,CharoIzquierdo,ChaToX,perejoanmitjans,zkellyq,eldestran,mendeley_com,kohlschuetter,matterofbiz,cbrew,Dede_Watson,adammarkus,gattardi,annieblogs,Schumenu,ian_soboroff,SueAnello,VMaryAbraham,gauravshukla,amominred,eilidhdickson,wellcomm,Ruohoska,AmaMocci,michaelwhitney,jgroc,MrMarkZamora,KimVallee,El_Pakozoico,CFAIco,mlaudisio,srstrong,EtcoMusic,E_Valero,WWequinerescue,awadallah"
SCREEN_NAMES3="RZYGA,GuyCresearchIT,soozmoody,alicialeeke,Beejer,Kbedosky,tgottron,BoulderSuZ,liquene,Susan_Kayne,SueMcCauley,alorza,usamaf,goobisgoofy,elenhitaES,milesj,dtunkelang,MiaPleasant,TwittingMarina,lutxana,pilarroch,globaltattooma,susanshapiro,Remetey,Radhouani,ovibarcelo,bmbartel,TORYCREATIVE,maninhavana,JOECORKSCREW,trailguide411,edchi,reygarrido,amandalynferri,rosamariaartal,adtothebone,TriciaKean,Galachitaa,petezin,worldreaders,coachireneg,jimmykastner,ANTONY_MELVIN,RichardDevine,callkathy,intven,jmschroeder,dutchboyinohio,BeltwayBargain,ThinkFY,marcelopont,CareerToolboxUS,menchhu,stranded_aoife,davelee,LuvSayal,Victoriabioque,JulieOni,TheMeganAnn,karinalegzdins,jjmerelo,luisnomad,jose_alias,webempresa20,SagBizAssocLLC,ImpeccableCoach,JeffTincher,anamariallopis,CHRISbodenner,CragHutchinson,fernand0,NataliaGomez_es,hautebeauty,eCare_Diary,heyitsfree,lourdesmunoz,misellebergonia,angga_dm,HansInglish,BerriDesign,bi0xid,AlephBlog,AFPChrisGraham,DeanDeBiase,cbrewster,suzee2253,GerryDawes,VinnyVerelli,mihailupunet,sgschneider,PakoRho,corallarrosa,LozzerOfHampton,kevinmclarke,ksmithdc,balancedbites,acmurillo,zblunt,LeadingWomen,IHCCSTEM,caff,jordimatas,susan_ringley,pogil,mgdelpozuelo,meganschmit,AliciaPomares,marctorrens"

SCREEN_NAMES2="asenkut,LegalNewsToUse,carmengelices,katjahofmann,Pablo_astudillo,crisalonsofran,NiallWharton,auroraferrer,doctorcasado,AmySoundDesign,MarkWhiteley,everyueveryme,seeseuno,edgarmeij,atjamie,lettersfrom500,ejoana,PeterLaBarbera,laniar,monicadeza,Jobs_SEO_UK,ITECS,KMNbooks,Krule,teresa_perales,iraszl,susieques,virginiog,mrlindsayjones,Craigrobinson,not_eggabo,jogartra,Timbahood,beatricewhelan,susan4ps,RasmusAuction,ErnieVelas,chloesor,AdweekMelissa,TyVelde,Day_NightEvents,ildefons,josejuanagudo,csaavedrasexta,ElviraLindo,pnhart,michibusch,MartaRodriguezA"
