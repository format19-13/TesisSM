# coding=utf-8
import urllib2
import re
from lxml import html
from HTMLParser import HTMLParser
import requests
import sys
import pycurl
import cookielib
import mechanize
import getpass
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support import ui

#driver = webdriver.PhantomJS()#executable_path=r'/home/vero/dev/phantomjs-2.1.1-linux-x86_64/bin/phantomjs')

#driver.get("https://www.facebook.com/profile.php?id=100012688690284&sk=about&section=contact-info&pnref=about")

#print driver.find_elements_by_xpath("//*[contains(text(), 'Viudo')]")

#soup = BeautifulSoup(driver.page_source, "html.parser")
#print soup.prettify()

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import unittest

class LoginTest(unittest.TestCase):
    def setUp(self):
        self.driver = webdriver.PhantomJS()
        self.driver.get("https://www.facebook.com/")

    def test_Login(self):
        driver = self.driver
        fbUsername = "tesisvero@hotmail.com"
        fbPassword = "vero1234"
        emailFieldID = ".//*[@id='email']"
        passFieldID = ".//*[@id='pass']"
        loginButtonXPath = ".//input[@value='Log In']"
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
        WebDriverWait(driver, 10).until(lambda driver: driver.find_element_by_xpath(flLogoXpath))
        print "vero"
        print driver.find_elements_by_xpath("//*[contains(text(), 'junio')]")

    def tearDown(self):
        self.driver.quit()

if __name__ == '__main__':
    unittest.main()