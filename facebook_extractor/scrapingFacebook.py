
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support import ui
from selenium.webdriver.common.keys import Keys
import requests

def getFnac(user):
    url='https://www.facebook.com'
    urlInfoSection=url + '/'+user+ '/about?section=contact-info&pnref=about'

    # Opening the web browser
    driver = webdriver.PhantomJS()
    driver.get(url)

    fbUsername = "tesisvero@hotmail.com"
    fbPassword = "vero1234"
    emailFieldID = ".//*[@id='email']"
    passFieldID = ".//*[@id='pass']"
    loginButtonXPath = ".//input[@tabindex=4]"
    aboutButtonXPath = './/a[@data-tab-key="about"]'

    flLogoXpath = "(//a[contains(@href, 'logo')])[1]"
    emailFieldElement = WebDriverWait(driver, 10).until(lambda driver: driver.find_element_by_xpath(emailFieldID))
    passFieldElement = WebDriverWait(driver, 10).until(lambda driver: driver.find_element_by_xpath(passFieldID))
    loginButtonElement = WebDriverWait(driver, 10).until(lambda driver: driver.find_element_by_xpath(loginButtonXPath))
    emailFieldElement.click()
    emailFieldElement.clear()
    emailFieldElement.send_keys(fbUsername)

    passFieldElement.click()
    passFieldElement.clear()
    passFieldElement.send_keys(fbPassword)
    loginButtonElement.click()
    driver.get(urlInfoSection)

    section=driver.find_elements_by_id("pagelet_basic")
    salida=0

    for i in range(0, len(section)):

    	aux= section[i].text
        if "o de nacimiento" in aux:
            salida=int(aux.split("o de nacimiento")[1][:5]) 
        else:
            try:
    	        sectionXPath = './/li[@data-privacy-fbid="8787510733"]'
    	        elem=section[i].find_elements_by_xpath(sectionXPath)
 
                for t in range(0, len(elem)):
    	            elem2=elem[t].find_element_by_tag_name("div")
    	            lstDivs= elem2.find_elements_by_tag_name("div")  
                    #print driver.page_source
    	            for x in range(0, len(lstDivs)):
                        fnacText= lstDivs[x].text
                        if "Fecha de nacimiento" in fnacText:
                            aux=lstDivs[x+1].text.split("de")
                            if len(aux)==3:
                            	salida=int(aux[2])

            except:
                return "error"
    print salida            
    return salida    

def getEdad(user):
    fnac=getFnac(user)
    edad=2016-fnac
    rango=''
    if edad>0:
        if edad<=24:
            rango="18-24"
        elif edad <=34:
            rango="25-34"
        elif edad <=49:
            rango="35-49"
        elif edad <=64:
            rango="50-64"
        else:
            rango="65-xx"
    return rango

print getEdad("mcopes")            
#getEdad("100014311837101")