import nltk

from nltk.book import *

text1.concordance("monstrous") #devuelve las frases q contienen la palabra

text1.similar("monstrous")
text2.similar("monstrous") # palabras similares en base al contexto, dependen del texto

text2.common_contexts(["monstrous", "very"])

text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])
