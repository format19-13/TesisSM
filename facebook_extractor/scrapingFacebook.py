
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support import ui

url1='https://www.facebook.com/profile.php?id=100014311837101&sk=about'
url2='https://www.facebook.com/mcopes/about'
url=url1
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

aboutButtonElement = WebDriverWait(driver, 10).until(lambda driver: driver.find_element_by_xpath(aboutButtonXPath))
aboutButtonElement.click()

fnac=driver.find_element_by_class_name("_c24").text

if "Fecha de nacimiento" in fnac: 
    print fnac.split('Fecha de nacimiento')[1]


